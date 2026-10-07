"""Integrity failures must stop before using original tensor bytes."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from original_tensor_store import OriginalTensorStore, header, verify_files


class OriginalTensorStoreTests(unittest.TestCase):
    def make_file(self, root, name='model.safetensors', spec=None, payload=b'\x80\x3f\x00\xc0'):
        spec = spec or {'matrix':{'dtype':'BF16','shape':[2,1],'data_offsets':[0,4]}}
        encoded = json.dumps(spec).encode()
        path = root/name
        path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
        return path

    def manifest(self, paths):
        return {'quantization':False,'weight_dtype':'BF16','files':
                {p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in paths}}

    def test_parallel_hashing_matches_serial_and_overlaps_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            files=[self.make_file(root,f'part-{i}.safetensors') for i in range(4)]
            manifest=self.manifest(files);barrier=threading.Barrier(4)
            serial=verify_files(root,manifest)
            parallel=verify_files(root,manifest,workers=4,observe=lambda:barrier.wait(timeout=5))
            self.assertEqual(serial,parallel)

    def test_parallel_corrupt_file_never_returns_partial_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=[self.make_file(root,f'part-{i}.safetensors') for i in range(4)]
            manifest=self.manifest(files);files[2].write_bytes(files[2].read_bytes()[:-1]+b'x')
            with self.assertRaises(ValueError):verify_files(root,manifest,workers=4)

    def test_parallel_observer_stop_propagates_and_joins_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=[self.make_file(root,f'part-{i}.safetensors') for i in range(4)]
            def stop():raise RuntimeError('pressure-stop-fixture')
            with self.assertRaisesRegex(RuntimeError,'pressure-stop-fixture'):
                verify_files(root,self.manifest(files),workers=4,observe=stop)
            self.assertFalse(any(t.name.startswith('original-hash') for t in threading.enumerate()))

    def test_parallel_invalid_worker_count_rejected_before_file_access(self):
        for workers in [0,9,True]:
            with self.assertRaises(ValueError):verify_files(Path('/does/not/exist'),{},workers=workers)

    def test_header_preserves_exact_bf16_byte_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.make_file(Path(tmp)); info=header(p)['matrix']
            self.assertEqual(info['shape'],[2,1])
            self.assertEqual(p.read_bytes()[info['offset']:info['offset']+info['bytes']],b'\x80\x3f\x00\xc0')

    def test_reject_wrong_dtype_shape_and_overlapping_payload(self):
        invalid = [
            {'matrix':{'dtype':'F16','shape':[2,1],'data_offsets':[0,4]}},
            {'matrix':{'dtype':'BF16','shape':[3,1],'data_offsets':[0,4]}},
            {'a':{'dtype':'BF16','shape':[1],'data_offsets':[0,2]},
             'b':{'dtype':'BF16','shape':[1],'data_offsets':[0,2]}},
            {'matrix':{'dtype':'BF16','shape':[1],'data_offsets':[2,4]}}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for spec in invalid:
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    header(self.make_file(Path(tmp),spec=spec))

    def test_hash_and_post_verification_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make_file(root);manifest=self.manifest([p])
            store=OriginalTensorStore(root,manifest)
            p.write_bytes(p.read_bytes()[:-1]+b'\xff')
            with self.assertRaises(ValueError): verify_files(root,manifest)
            with self.assertRaises(ValueError): store._check(p.name)

    def test_correct_shard_index_and_wrong_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make_file(root,name='model-00001.safetensors')
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'matrix':p.name}}))
            store=OriginalTensorStore(root,self.manifest([p,index]))
            self.assertEqual(store.tensors['matrix']['file'],p.name)
            index.write_text(json.dumps({'weight_map':{'matrix':'missing.safetensors'}}))
            with self.assertRaises(ValueError): OriginalTensorStore(root,self.manifest([p,index]))

    def test_duplicates_and_unverified_index_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.make_file(root,'a.safetensors');b=self.make_file(root,'b.safetensors')
            with self.assertRaises(ValueError): OriginalTensorStore(root,self.manifest([a,b]))
            (root/'model.safetensors.index.json').write_text('{}')
            with self.assertRaises(ValueError): OriginalTensorStore(root,self.manifest([a]))

    def test_symlink_outside_model_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();p=self.make_file(root)
            (model/p.name).symlink_to(p)
            with self.assertRaises(ValueError): verify_files(model,self.manifest([p]))

    def test_exact_reads_and_truncation_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make_file(root)
            store=OriginalTensorStore(root,self.manifest([p]),io_mode='raw-cached')
            info=store.tensors['matrix']
            self.assertEqual(store.read_bytes(p.name,info['offset'],4),b'\x80\x3f\x00\xc0')
            with self.assertRaises(ValueError): store.read_bytes(p.name,p.stat().st_size,4)
            with patch('original_tensor_store.os.pread',return_value=b'\x80'):
                with self.assertRaises(ValueError): store.read_bytes(p.name,info['offset'],4)

    def test_nocache_sets_only_descriptor_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make_file(root)
            with patch('original_tensor_store.sys.platform','darwin'):
                store=OriginalTensorStore(root,self.manifest([p]),io_mode='raw-nocache')
            with patch('fcntl.fcntl',return_value=0) as control:
                info=store.tensors['matrix']
                self.assertEqual(store.read_bytes(p.name,info['offset'],4),b'\x80\x3f\x00\xc0')
            self.assertEqual(control.call_args.args[1:],(48,1))
            self.assertEqual(store.f_nocache_descriptors,1)

    def test_invalid_mode_and_platform_fail(self):
        with self.assertRaises(ValueError): OriginalTensorStore('.',{},io_mode='invalid')
        with patch('original_tensor_store.sys.platform','linux'),self.assertRaises(ValueError):
            OriginalTensorStore('.',{},io_mode='raw-nocache')

    def test_hash_verification_nocache_keeps_digest_and_observes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make_file(root)
            with patch('original_tensor_store.sys.platform','darwin'),patch('fcntl.fcntl',return_value=0) as control,patch('builtins.print'):
                observer=unittest.mock.Mock()
                identities=verify_files(root,self.manifest([p]),nocache=True,observe=observer)
            self.assertIn(p.name,identities)
            observer.assert_called_once()
            self.assertEqual(control.call_args.args[1:],(48,1))


if __name__ == '__main__': unittest.main()
