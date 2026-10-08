"""Read original BF16 tensors/rows from pinned safetensors without repackaging.

CPU metadata is retained; evaluated MLX tensors are not retained by the store.
Selective lazy loading uses existing MLX functionality, not a novelty claim.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


def header(path):
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated safetensors prefix')
        length = struct.unpack('<Q', raw)[0]
        if not 0 < length <= 16*1024**2:
            raise ValueError('Invalid safetensors header length')
        encoded = f.read(length)
        if len(encoded) != length:
            raise ValueError('Truncated safetensors header')
    data = json.loads(encoded)
    start, size = 8+length, path.stat().st_size
    result, intervals = {}, []
    for name, spec in data.items():
        if name == '__metadata__':
            continue
        if spec['dtype'] != 'BF16':
            raise ValueError('Only original BF16 tensors supported')
        shape, offsets = spec['shape'], spec['data_offsets']
        if any(type(v) is not int or v < 0 for v in shape+offsets) or len(offsets) != 2:
            raise ValueError('Invalid shape or offset')
        low, high = offsets
        if high-low != 2*math.prod(shape) or not 0 <= low <= high <= size-start:
            raise ValueError('Tensor byte count or range differs from shape')
        result[name] = {'dtype': 'BF16', 'shape': shape, 'file': path.name,
                        'offset': start+low, 'bytes': high-low}
        intervals.append((low, high))
    previous = 0
    for low, high in sorted(intervals):
        if low != previous:
            raise ValueError('Tensor payload contains gaps or overlapping ranges')
        previous = high
    if start+previous != size:
        raise ValueError('Unexpected bytes after tensor payload')
    return result


def verify_files(model, manifest, *, nocache=False, observe=None, workers=1):
    if nocache and sys.platform != 'darwin':
        raise ValueError('Verification F_NOCACHE requires macOS')
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Verification workers must be 1..8')
    abort = Event()
    def verify_one(name, item):
        if Path(name).name != name:
            raise ValueError('Flat filenames required')
        p = model/name
        if p.resolve().parent != model.resolve():
            raise ValueError('Model file escapes local directory')
        before = p.stat()
        if before.st_size != item['bytes']:
            raise ValueError('Model file size differs')
        with p.open('rb') as f:
            if nocache:
                import fcntl
                fcntl.fcntl(f.fileno(),48,1)
            hasher = hashlib.sha256()
            while data := f.read(8*1024**2):
                if abort.is_set():
                    raise ValueError('Verification cancelled after another file failed')
                hasher.update(data)
                if observe:
                    observe()
            digest = hasher.hexdigest()
        after = p.stat()
        if digest != item['sha256'] or (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Model hash differs or file changed during verification')
        return [after.st_size, after.st_mtime_ns, after.st_ino]
    if workers == 1:
        return {name:verify_one(name,item) for name,item in manifest['files'].items()}
    identities = {}
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='original-hash') as pool:
        futures={pool.submit(verify_one,name,item):name for name,item in manifest['files'].items()}
        try:
            for future in as_completed(futures):
                identities[futures[future]]=future.result()
        except BaseException:
            abort.set()
            for future in futures:future.cancel()
            raise
    return {name:identities[name] for name in manifest['files']}


class OriginalTensorStore:
    def __init__(self, model, manifest, verify=True, verified_identities=None, io_mode='native'):
        if io_mode not in {'native', 'raw-cached', 'raw-nocache'}:
            raise ValueError('Unknown original-weight I/O mode')
        if io_mode == 'raw-nocache' and sys.platform != 'darwin':
            raise ValueError('Per-descriptor F_NOCACHE requires macOS')
        self.io_mode = io_mode
        self.f_nocache_descriptors = 0
        self.model, self.manifest = Path(model).resolve(), manifest
        if manifest.get('quantization') is not False or manifest.get('weight_dtype') != 'BF16':
            raise ValueError('Pinned original BF16 manifest required')
        self.identities = verify_files(self.model, manifest) if verify else verified_identities
        if self.identities is None:
            raise ValueError('Supply full verification or previously verified file identities')
        self.tensors = {}
        for name in manifest['files']:
            self._check(name)
            if name.endswith('.safetensors'):
                tensors = header(self.model/name)
                if set(tensors) & set(self.tensors):
                    raise ValueError('Duplicate tensor names across source files')
                self.tensors.update(tensors)
        if not self.tensors:
            raise ValueError('No original tensors found')
        index = self.model/'model.safetensors.index.json'
        if index.exists():
            if index.name not in manifest['files']:
                raise ValueError('Unverified safetensors index')
            mapping = json.loads(index.read_text())['weight_map']
            if mapping != {name: info['file'] for name,info in self.tensors.items()}:
                raise ValueError('Index disagrees with original tensor headers')
        self.selected_tensor_bytes_requested = 0
        self.row_bytes_read = 0
        self.layer_groups_requested = 0

    def _check(self, name):
        p = self.model/name
        st = p.stat()
        if p.resolve().parent != self.model or [st.st_size,st.st_mtime_ns,st.st_ino] != self.identities[name]:
            raise ValueError('Verified model file changed')

    def load(self, names):
        import mlx.core as mx
        if self.io_mode != 'native':
            selected = {}
            for name in names:
                selected[name] = self._raw_tensor(name)
            self.selected_tensor_bytes_requested += sum(self.tensors[n]['bytes'] for n in names)
            return selected
        groups = defaultdict(list)
        for name in names:
            groups[self.tensors[name]['file']].append(name)
        selected = {}
        for file, keys in groups.items():
            self._check(file)
            arrays = mx.load(str(self.model/file))
            selected.update({name: arrays[name] for name in keys})
            del arrays
        # Never keep the source dictionary: doing so can retain evaluated layers.
        mx.eval(selected)
        if {str(t.dtype) for t in selected.values()} != {'mlx.core.bfloat16'}:
            raise ValueError('MLX changed BF16 dtype')
        self.selected_tensor_bytes_requested += sum(self.tensors[n]['bytes'] for n in names)
        return selected

    def layer(self, index):
        names = [name for name in self.tensors if name.startswith(f'model.layers.{index}.')]
        if not names:
            raise ValueError('Missing layer')
        self.layer_groups_requested += 1
        return self.load(names)

    def layer_raw(self, index):
        if self.io_mode not in {'raw-cached', 'raw-nocache'}:
            raise ValueError('Raw read-ahead requires a raw I/O mode')
        names = [n for n in self.tensors if n.startswith(f'model.layers.{index}.')]
        if not names:
            raise ValueError('Missing layer')
        return {n: self.read_bytes(self.tensors[n]['file'], self.tensors[n]['offset'],
                                  self.tensors[n]['bytes']) for n in names}

    def layer_prefix(self, index, budget):
        if self.io_mode not in {'raw-cached','raw-nocache'} or type(budget) is not int or budget < 0:
            raise ValueError('Raw I/O and nonnegative byte budget required')
        names = [n for n in self.tensors if n.startswith(f'model.layers.{index}.')]
        if not names:
            raise ValueError('Missing layer')
        result = {}
        remaining = budget
        for name in names:
            if not remaining:break
            spec = self.tensors[name]
            size = min(remaining,spec['bytes'])
            result[name] = self.read_bytes(spec['file'],spec['offset'],size)
            remaining -= size
        return result

    def materialize_prefix_layer(self, index, prefix):
        import mlx.core as mx
        import numpy as np
        if sys.byteorder != 'little':raise ValueError('Little-endian required')
        names = [n for n in self.tensors if n.startswith(f'model.layers.{index}.')]
        if not names or not set(prefix) <= set(names):raise ValueError('Invalid layer prefix')
        selected = {}
        for name in names:
            spec = self.tensors[name]
            first = prefix.pop(name,b'')
            if len(first)>spec['bytes']:raise ValueError('Oversized prefix')
            rest = self.read_bytes(spec['file'],spec['offset']+len(first),spec['bytes']-len(first)) if len(first)<spec['bytes'] else b''
            raw = first+rest
            del first,rest
            bits = np.frombuffer(raw,dtype='<u2').reshape(spec['shape'])
            selected[name] = mx.array(bits).view(mx.bfloat16)
            mx.eval(selected[name])
            del raw,bits
        self.layer_groups_requested += 1
        self.selected_tensor_bytes_requested += sum(self.tensors[n]['bytes'] for n in names)
        return selected

    def materialize_layer(self, raw):
        import mlx.core as mx
        import numpy as np
        if sys.byteorder != 'little':
            raise ValueError('Little-endian host required for raw BF16 transport')
        selected = {}
        for name, payload in raw.items():
            spec = self.tensors[name]
            if len(payload) != spec['bytes']:
                raise ValueError('Truncated prefetched tensor')
            bits = np.frombuffer(payload, dtype='<u2').reshape(spec['shape'])
            selected[name] = mx.array(bits).view(mx.bfloat16)
            mx.eval(selected[name])
        self.layer_groups_requested += 1
        self.selected_tensor_bytes_requested += sum(self.tensors[n]['bytes'] for n in raw)
        return selected

    def rows(self, name, start, end):
        import mlx.core as mx
        import numpy as np
        if sys.byteorder != 'little':
            raise ValueError('Little-endian host required for raw BF16 transport')
        spec = self.tensors[name]
        if len(spec['shape']) != 2 or type(start) is not int or type(end) is not int or not 0 <= start < end <= spec['shape'][0]:
            raise ValueError('Invalid matrix row range')
        self._check(spec['file'])
        columns = spec['shape'][1]
        size = (end-start)*columns*2
        raw = self.read_bytes(spec['file'], spec['offset']+start*columns*2, size)
        self.row_bytes_read += size
        # Bit reinterpretation, never conversion through float32 or quantization.
        bits = np.frombuffer(raw, dtype='<u2').reshape(end-start, columns)
        values = mx.array(bits).view(mx.bfloat16)
        mx.eval(values)
        return values

    def read_bytes(self, file, offset, size):
        """Exact checked read; F_NOCACHE applies only to this opened descriptor."""
        self._check(file)
        with (self.model/file).open('rb') as f:
            before = os.fstat(f.fileno())
            identity = [before.st_size, before.st_mtime_ns, before.st_ino]
            if identity != self.identities[file]:
                raise ValueError('Opened file differs from verified identity')
            if offset < 0 or size < 0 or offset+size > before.st_size:
                raise ValueError('Read exceeds verified file')
            if self.io_mode == 'raw-nocache':
                import fcntl
                # F_NOCACHE=48 from the public macOS SDK sys/fcntl.h.
                # Do not use F_GLOBAL_NOCACHE or alter machine-wide cache state.
                fcntl.fcntl(f.fileno(), 48, 1)
                self.f_nocache_descriptors += 1
            raw = os.pread(f.fileno(), size, offset)
            if len(raw) != size:
                raise ValueError('Truncated original tensor read')
            after = os.fstat(f.fileno())
            if [after.st_size,after.st_mtime_ns,after.st_ino] != identity:
                raise ValueError('Original file changed while reading')
        return raw

    def _raw_tensor(self, name):
        import mlx.core as mx
        import numpy as np
        if sys.byteorder != 'little':
            raise ValueError('Little-endian host required for raw BF16 transport')
        spec = self.tensors[name]
        raw = self.read_bytes(spec['file'],spec['offset'],spec['bytes'])
        bits = np.frombuffer(raw,dtype='<u2').reshape(spec['shape'])
        values = mx.array(bits).view(mx.bfloat16)
        mx.eval(values)
        return values


class LayerReadAhead:
    """One future of checked raw bytes; MLX work remains on the caller thread.

    Consumed futures are cleared before submitting the next read. Closing joins
    the only reader, including when materialization or model computation fails.
    """
    def __init__(self, store, count, prefix_bytes=0):
        self.store, self.count = store, count
        self.prefix_bytes = prefix_bytes
        self.pool = self.future = None
        self.next_index = 0

    def __enter__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='original-read')
        self.future = self._submit(0)
        return self

    def _submit(self, index):
        if self.prefix_bytes:
            return self.pool.submit(self.store.layer_prefix,index,self.prefix_bytes)
        return self.pool.submit(self.store.layer_raw,index)

    def take(self, index):
        if index != self.next_index or self.future is None:
            raise ValueError('Read-ahead layers must be consumed exactly once in order')
        raw = self.future.result()
        self.future = None
        self.next_index += 1
        if self.next_index < self.count:
            self.future = self._submit(self.next_index)
        return self.store.materialize_prefix_layer(index,raw) if self.prefix_bytes else self.store.materialize_layer(raw)

    def __exit__(self, *exception):
        if self.future is not None:
            self.future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.future = None
        return False
