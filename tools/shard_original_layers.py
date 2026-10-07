"""Repackage original BF16 tensors by layer without changing payload bytes.

Prepares files for future streaming experiments. This does not run a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

from run_original_precision import inspect


def shard(model, output):
    if output.exists():
        raise ValueError('Fresh output directory required')
    provenance = inspect(model)
    source = model / 'model.safetensors'
    with source.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
        data_start = 8+length
        groups = {}
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            match = re.match(r'model\.layers\.(\d+)\.', name)
            group = f'layer-{int(match.group(1)):03d}' if match else 'shared'
            groups.setdefault(group, {})[name] = tensor
        output.mkdir(parents=True)
        manifest = {'source': provenance, 'payload_conversion': False, 'shards': []}
        for group, tensors in sorted(groups.items()):
            new, offset = {}, 0
            for name, tensor in tensors.items():
                start, end = tensor['data_offsets']
                new[name] = {**tensor, 'data_offsets': [offset, offset+end-start]}
                offset += end-start
            encoded = json.dumps(new, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            target = output / (group+'.safetensors')
            hashes = {}
            with target.open('xb') as dest:
                dest.write(struct.pack('<Q', len(encoded)))
                dest.write(encoded)
                for name, tensor in tensors.items():
                    start, end = tensor['data_offsets']
                    f.seek(data_start+start)
                    remaining, digest = end-start, hashlib.sha256()
                    while remaining:
                        chunk = f.read(min(remaining, 8*1024**2))
                        if not chunk:
                            raise ValueError('Truncated input tensor')
                        dest.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                    hashes[name] = digest.hexdigest()
            # Read back every output tensor and compare hashes against source payload.
            with target.open('rb') as dest:
                n = struct.unpack('<Q', dest.read(8))[0]
                h = json.loads(dest.read(n))
                for name, tensor in h.items():
                    start, end = tensor['data_offsets']
                    dest.seek(8+n+start)
                    remaining, digest = end-start, hashlib.sha256()
                    while remaining:
                        chunk = dest.read(min(remaining, 8*1024**2))
                        if not chunk:
                            raise ValueError('Truncated output tensor')
                        digest.update(chunk)
                        remaining -= len(chunk)
                    if digest.hexdigest() != hashes[name]:
                        raise ValueError('Output tensor payload differs from original')
                dest.seek(0)
                file_hash = hashlib.file_digest(dest, 'sha256').hexdigest()
            manifest['shards'].append({'file': target.name, 'bytes': target.stat().st_size,
                'sha256': file_hash, 'tensor_payload_sha256': hashes})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'shards': len(groups), 'verified_unchanged_tensor_payloads':
                      sum(len(t) for t in groups.values())}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    shard(a.model.resolve(), a.output.resolve())
