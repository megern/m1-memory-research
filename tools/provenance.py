"""Pin the base of this study before attributing results to an upstream model."""
import hashlib
import json
from pathlib import Path

def identify_base(directory):
    directory=Path(directory)
    digest=hashlib.sha256((directory/'config.json').read_bytes()).hexdigest()
    root=Path(__file__).resolve().parents[1]/'models'
    for name in ['qwen3-0.6b.json','qwen3-1.7b.json']:
        path=root/name
        if not path.is_file():
            continue
        manifest=json.loads(path.read_text())
        if digest==manifest['files']['config.json']['sha256']:
            return manifest
    raise ValueError('Model configuration is not one of this study\'s pinned bases')

def verify_base(directory):
    directory=Path(directory)
    manifest=identify_base(directory)
    expected={name:manifest['files'][name]['sha256'] for name in ['config.json','model.safetensors']}
    for name, digest in expected.items():
        with (directory/name).open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != digest:
            raise ValueError('Pinned base checksum mismatch in '+name)
    return expected
