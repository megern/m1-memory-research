"""Bounded CPU file prefetch; MLX work remains on the calling thread."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time


class LayerReader:
    def __init__(self, directory, layers, ahead):
        if ahead not in (0, 1, 2) or layers < 1:
            raise ValueError('Supported lookahead is 0, 1 or 2 layers')
        self.directory, self.layers, self.ahead = Path(directory), layers, ahead
        self.pool = ThreadPoolExecutor(max_workers=1) if ahead else None
        self.pending, self.next_index = {}, 0

    def _read(self, index):
        started = time.monotonic()
        data = (self.directory/f'layer-{index:03d}.safetensors').read_bytes()
        return data, time.monotonic()-started

    def begin(self):
        if self.pending:
            raise ValueError('Previous pass was not fully consumed')
        self.next_index = 0
        if self.pool:
            for index in range(min(self.layers, self.ahead+1)):
                self.pending[index] = self.pool.submit(self._read, index)

    def take(self, index):
        if index != self.next_index or index >= self.layers:
            raise ValueError('Layers must be consumed once in order')
        started = time.monotonic()
        if self.pool:
            data, read_seconds = self.pending.pop(index).result()
            wait_seconds = time.monotonic()-started
            following = index+self.ahead+1
            if following < self.layers:
                self.pending[following] = self.pool.submit(self._read, following)
        else:
            data, read_seconds = self._read(index)
            wait_seconds = time.monotonic()-started
        self.next_index += 1
        return data, read_seconds, wait_seconds

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
