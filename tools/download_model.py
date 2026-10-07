"""Resumable, checksum-verified public model downloads; no model execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import quote
import httpx

def sha_region(fd, start, size):
    h = hashlib.sha256()
    while size:
        data = os.pread(fd, min(size, 1 << 20), start)
        if not data:
            raise ValueError('Incomplete local chunk')
        h.update(data)
        start += len(data)
        size -= len(data)
    return h.hexdigest()

def download(manifest, output, workers=4, chunk_mib=32):
    if not 1 <= workers <= 16 or not 1 <= chunk_mib <= 128:
        raise ValueError('Workers must be 1..16 and chunks 1..128 MiB')
    size, expected = manifest['bytes'], manifest['sha256']
    if not isinstance(size, int) or not 0 < size <= 32 * 1024**3:
        raise ValueError('This experiment permits model downloads up to 32 GiB')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if output.stat().st_size != size or digest != expected:
            raise ValueError('Existing final file does not match; refusing to overwrite')
        print('Already verified', flush=True)
        return
    part = output.with_suffix(output.suffix + '.part')
    state_path = output.with_suffix(output.suffix + '.resume.json')
    chunk = chunk_mib * 1024**2
    identity = {'manifest': manifest, 'chunk_bytes': chunk}
    state = {'identity': identity, 'complete': {}}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['identity'] != identity:
            raise ValueError('Resume identity differs; refusing mixed downloads')
    elif part.exists():
        raise ValueError('Partial file lacks its resume manifest')
    if shutil.disk_usage(output.parent).free < size + 4 * 1024**3:
        raise ValueError('At least model-size + 4 GiB of free storage required')
    fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    def save_state():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2))
        os.replace(temp, state_path)
    save_state()
    for index, digest in list(state['complete'].items()):
        start = int(index) * chunk
        if sha_region(fd, start, min(chunk, size-start)) != digest:
            del state['complete'][index]
    save_state()
    base = 'https://huggingface.co/' + manifest['repository'] + '/resolve/' + manifest['revision'] + '/' + quote(manifest['filename'], safe='/')
    total = math.ceil(size / chunk)
    started = time.monotonic()
    def fetch(index):
        start, end = index * chunk, min(size, (index+1) * chunk) - 1
        for attempt in range(6):
            try:
                digest = hashlib.sha256()
                offset = start
                with httpx.stream('GET', base + '?download=true&range_start=' + str(start),
                                  headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                  follow_redirects=True, timeout=120) as response:
                    response.raise_for_status()
                    required = f'bytes {start}-{end}/{size}'
                    if response.status_code != 206 or response.headers.get('content-range') != required:
                        raise ValueError('Server did not honor the exact requested range')
                    for data in response.iter_bytes(1 << 20):
                        if offset + len(data) > end + 1:
                            raise ValueError('Response exceeded its requested range')
                        written = os.pwrite(fd, data, offset)
                        if written != len(data):
                            raise OSError('Short write')
                        digest.update(data)
                        offset += len(data)
                if offset != end + 1:
                    raise ValueError('Short range response')
                with lock:
                    os.fsync(fd)
                    state['complete'][str(index)] = digest.hexdigest()
                    save_state()
                return
            except (httpx.HTTPError, OSError, ValueError):
                if attempt == 5:
                    raise RuntimeError(f'Chunk {index} failed after retries') from None
                time.sleep(min(2**attempt, 15))
    try:
        pending = [i for i in range(total) if str(i) not in state['complete']]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, i) for i in pending]
            last = 0
            for future in as_completed(futures):
                future.result()
                now = time.monotonic()
                if now - last >= 20:
                    complete_bytes = sum(min(chunk, size-int(i)*chunk) for i in state['complete'])
                    print(json.dumps({'complete_bytes': complete_bytes, 'total_bytes': size,
                                      'complete_chunks': len(state['complete']), 'total_chunks': total,
                                      'elapsed_seconds': round(now-started, 1)}), flush=True)
                    last = now
        digest = sha_region(fd, 0, size)
        if digest != expected:
            raise ValueError('Full-file SHA256 mismatch; partial retained for inspection')
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(part, output)
    print(json.dumps({'status': 'verified', 'bytes': size, 'sha256': digest}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk-mib', type=int, default=32)
    a = p.parse_args()
    download(json.loads(a.manifest.read_text()), a.output, a.workers, a.chunk_mib)
