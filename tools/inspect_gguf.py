"""Read GGUF metadata/tensor dimensions without allocating or executing weights."""
import argparse
import json
import math
from pathlib import Path
import struct

FIXED={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
SELECTED={'general.architecture','general.name','general.size_label','general.file_type',
          'llama.block_count','llama.embedding_length','llama.attention.head_count',
          'tokenizer.ggml.add_bos_token'}


def inspect(path):
    with path.open('rb') as file:
        def number(code):
            size=struct.calcsize('<'+code)
            data=file.read(size)
            if len(data)!=size:
                raise ValueError('Truncated GGUF header')
            return struct.unpack('<'+code,data)[0]
        def string():
            size=number('Q')
            if size>8*1024**2:
                raise ValueError('Oversized GGUF string')
            data=file.read(size)
            if len(data)!=size:
                raise ValueError('Truncated GGUF string')
            return data.decode('utf-8')
        def value(kind,depth=0):
            if depth>2:
                raise ValueError('Unexpected nested metadata')
            if kind in FIXED:
                return number(FIXED[kind])
            if kind==8:
                return string()
            if kind==9:
                subtype,count=number('I'),number('Q')
                if count>2_000_000:
                    raise ValueError('Oversized metadata array')
                for _ in range(count):
                    value(subtype,depth+1)
                return {'array_elements':count,'element_type':subtype}
            raise ValueError('Unsupported metadata type')
        if file.read(4)!=b'GGUF':
            raise ValueError('GGUF required')
        version=number('I')
        if version!=3:
            raise ValueError('Only the recorded GGUF v3 format is supported')
        tensors,entries=number('Q'),number('Q')
        if max(tensors,entries)>100_000:
            raise ValueError('Oversized header')
        metadata={}
        for _ in range(entries):
            key,kind=string(),number('I')
            parsed=value(kind)
            if key in SELECTED:
                metadata[key]=parsed
        elements=0
        for _ in range(tensors):
            name=string();dimensions=number('I')
            if not 1<=dimensions<=8:
                raise ValueError('Unsupported tensor dimensions')
            shape=[number('Q') for _ in range(dimensions)]
            number('I');number('Q')
            elements+=math.prod(shape)
    return {'gguf_version':version,'tensor_count':tensors,'stored_tensor_elements':elements,
            'metadata':metadata,'file_bytes':path.stat().st_size,
            'method':'Header and tensor dimensions only; not a second execution or a replacement for the full-file checksum.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    result=inspect(args.model)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
