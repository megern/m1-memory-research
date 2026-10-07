"""Explicitly acquire the pinned public base, or verify local files offline."""
import argparse
import hashlib
import json
from pathlib import Path
from download_model import download


def acquire(manifest, output, verify_only=False):
    for name, metadata in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Only flat model files are permitted')
        local = output / name
        if verify_only:
            if not local.is_file() or local.stat().st_size != metadata['bytes']:
                raise ValueError(f'Missing or wrong-size local file: {name}')
            with local.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if digest != metadata['sha256']:
                raise ValueError(f'Local checksum mismatch: {name}')
        else:
            pinned = {'repository': manifest['repository'], 'revision': manifest['revision'],
                      'filename': name, 'bytes': metadata['bytes'],
                      'sha256': metadata['sha256'], 'license': manifest['license']}
            download(pinned, local, workers=2, chunk_mib=32)
    print('Every pinned base file verified; no model code was executed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'models/qwen3-0.6b.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    acquire(json.loads(args.manifest.read_text()), args.output.resolve(), args.verify_only)
