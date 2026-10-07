"""Acquire pinned original files with resumable ranges; never executes model code."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from download_model import download


def acquire(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for name, entry in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Flat filenames required')
        local = output/name
        if local.exists():
            with local.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if local.stat().st_size != entry['bytes'] or digest != entry['sha256']:
                raise ValueError('Existing file differs; refusing replacement')
        else:
            pending.append(name)
    state = {'repository': manifest['repository'], 'revision': manifest['revision'],
             'status': 'downloading', 'pending_files': pending.copy(), 'completed_files': []}
    def save():
        p = output/'acquisition-state.json'
        temporary = p.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(p)
    save()
    def fetch(name):
        entry = manifest['files'][name]
        spec = {'repository': manifest['repository'], 'revision': manifest['revision'],
                'filename': name, 'license': manifest['license'], **entry}
        download(spec, output/name, workers=8, chunk_mib=8)
        return name
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(fetch, name) for name in pending]):
                name = future.result()
                state['completed_files'].append(name)
                state['pending_files'].remove(name)
                save()
        state['status'] = 'verified_complete'
        save()
    except Exception as exc:
        state['status'], state['error_type'] = 'failed_resumable', type(exc).__name__
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    acquire(a.manifest.resolve(), a.output.resolve())
