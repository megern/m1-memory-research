"""Prefix byte budgets, reconstruction and guarded once-only verification."""
import hashlib,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from original_tensor_store import OriginalTensorStore,LayerReadAhead
from run_large_model_cache_pair import verify

class PrefixTests(unittest.TestCase):
    def test_prefix_bounds_exact_offsets_and_missing_layer(self):
        store=OriginalTensorStore.__new__(OriginalTensorStore);store.io_mode='raw-cached'
        store.tensors={'model.layers.0.a':{'file':'a','offset':10,'bytes':6},'model.layers.0.b':{'file':'a','offset':16,'bytes':8},'model.layers.1.a':{'file':'a','offset':24,'bytes':8}}
        calls=[]
        def read(file,offset,size):calls.append((offset,size));return bytes(range(offset,offset+size))
        store.read_bytes=read
        prefix=store.layer_prefix(0,10)
        self.assertEqual(sum(map(len,prefix.values())),10)
        self.assertEqual(calls,[(10,6),(16,4)])
        self.assertEqual(store.layer_prefix(0,0),{})
        with self.assertRaises(ValueError):store.layer_prefix(2,10)
        with self.assertRaises(ValueError):store.layer_prefix(0,-1)

    def test_prefix_reader_routes_original_index_and_budget(self):
        calls=[]
        def read(index,budget):calls.append((index,budget));return {'data':bytes(8)}
        store=SimpleNamespace(layer_prefix=read,materialize_prefix_layer=lambda index,raw:(index,len(raw['data'])))
        with LayerReadAhead(store,2,8) as reader:
            self.assertEqual(reader.take(0),(0,8));self.assertEqual(reader.take(1),(1,8))
        self.assertEqual(calls,[(0,8),(1,8)])

    def test_once_only_verification_detects_identity_and_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights';path.write_bytes(b'original bytes')
            manifest={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch('run_large_model_cache_pair.psutil.virtual_memory',return_value=SimpleNamespace(available=2**30)),patch('run_large_model_cache_pair.psutil.swap_memory',return_value=SimpleNamespace(used=0)):
                result=verify(path,manifest,0)
                self.assertEqual(result['sha256'],manifest['sha256'])
                def mutate(sample):path.write_bytes(b'changed bytes!')
                with self.assertRaises(ValueError):verify(path,manifest,0,mutate)
            path.write_bytes(b'original bytes')
            with patch('run_large_model_cache_pair.psutil.virtual_memory',return_value=SimpleNamespace(available=2**30)),patch('run_large_model_cache_pair.psutil.swap_memory',return_value=SimpleNamespace(used=2**30)):
                with self.assertRaises(MemoryError):verify(path,manifest,0)
