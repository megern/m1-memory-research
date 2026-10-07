"""Bounded byte-reader lifetime and failure propagation; MLX checked by real audit."""
import sys
from pathlib import Path
from threading import Event, enumerate as threads
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from original_tensor_store import LayerReadAhead

class ReadAheadTests(unittest.TestCase):
    def test_one_future_in_order_and_reader_overlaps_consumer(self):
        entered,release=Event(),Event()
        reads=[]
        def read(index):
            reads.append(index)
            if index==1:
                entered.set()
                if not release.wait(5):raise RuntimeError('Test reader timed out')
            return {'index':index}
        store=SimpleNamespace(layer_raw=read,materialize_layer=lambda raw:raw['index'])
        try:
            with LayerReadAhead(store,3) as reader:
                self.assertEqual(reader.take(0),0)
                self.assertTrue(entered.wait(5))
                self.assertEqual(reads,[0,1])
                with self.assertRaises(ValueError):reader.take(0)
                release.set()
                self.assertEqual(reader.take(1),1)
                self.assertEqual(reader.take(2),2)
                self.assertIsNone(reader.future)
                with self.assertRaises(ValueError):reader.take(3)
        finally:release.set()
        self.assertFalse(any(t.name.startswith('original-read') for t in threads()))

    def test_read_failure_propagates_and_reader_is_joined(self):
        def read(index):raise ValueError('Changed original file')
        store=SimpleNamespace(layer_raw=read,materialize_layer=lambda raw:raw)
        reader=LayerReadAhead(store,2)
        with self.assertRaisesRegex(ValueError,'Changed original file'):
            with reader:reader.take(0)
        self.assertIsNone(reader.future)
        self.assertFalse(any(t.name.startswith('original-read') for t in threads()))

    def test_materialization_failure_closes_pending_reader(self):
        def convert(raw):raise MemoryError('Budget exceeded')
        store=SimpleNamespace(layer_raw=lambda index:bytes(8),materialize_layer=convert)
        reader=LayerReadAhead(store,2)
        with self.assertRaises(MemoryError):
            with reader:reader.take(0)
        self.assertIsNone(reader.future)
        self.assertFalse(any(t.name.startswith('original-read') for t in threads()))
