"""Install the checksum-pinned official Apple Silicon runtime into a fresh folder."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import httpx

URL='https://github.com/ggml-org/llama.cpp/releases/download/b11457/llama-b11457-bin-macos-arm64.tar.gz'
SHA256='e234070cbde0c8b0d30f79fa08ff8246abe22c72acb752619349b8d9c46f7839'
BYTES=12007659


def install(output, archive=None):
    output=output.resolve()
    if output.exists():
        raise ValueError('Choose a fresh runtime directory')
    if archive is None:
        archive=output.parent/(output.name+'.tar.gz')
        if archive.exists():
            raise ValueError('Archive destination already exists; use --archive to verify it')
        archive.parent.mkdir(parents=True,exist_ok=True)
        received=0
        with httpx.stream('GET',URL,follow_redirects=True,timeout=120) as response:
            response.raise_for_status()
            with archive.open('xb') as target:
                for block in response.iter_bytes(1<<20):
                    received+=len(block)
                    if received>BYTES:
                        raise ValueError('Release archive exceeds its pinned size')
                    target.write(block)
    if archive.stat().st_size!=BYTES:
        raise ValueError('Unexpected release size')
    with archive.open('rb') as source:
        digest=hashlib.file_digest(source,'sha256').hexdigest()
    if digest!=SHA256:
        raise ValueError('Official release checksum mismatch')
    output.mkdir(parents=True)
    with tarfile.open(archive) as handle:
        handle.extractall(output,filter='data')
    runtime=output/'llama-b11457/llama-completion'
    if not runtime.is_file():
        raise ValueError('Expected official executable absent after extraction')
    (output/'provenance.json').write_text(json.dumps({'release':'b11457','url':URL,
       'archive_bytes':BYTES,'sha256':digest,'executable':'llama-b11457/llama-completion'},indent=2)+'\n')
    print('Verified local runtime:',runtime)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args()
    install(args.output,args.archive)
