"""Chunk orchestration boundaries; numerical correctness uses real model audits."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from direct_original_engine import DirectOriginalEngine
from run_direct_original import compare_reference_logits

class ChunkTests(unittest.TestCase):
    def test_reference_greedy_is_not_teacher_forced_input(self):
        scores=np.array([[1.,5.,2.]])
        result=compare_reference_logits(scores,scores,1,2,0)
        self.assertTrue(result['greedy_token_equal'])
        self.assertEqual(result['teacher_forced_token_id'],2)
        self.assertEqual(result['reference_greedy_token_id'],1)
        self.assertFalse(compare_reference_logits(scores,scores,2,2,0)['greedy_token_equal'])

    def test_invalid_logits_cannot_pass_comparison(self):
        with self.assertRaises(ValueError):compare_reference_logits(np.array([[float('nan')]]),np.array([[0.]]),0,0,0)
        with self.assertRaises(ValueError):compare_reference_logits(np.zeros((1,2)),np.zeros((1,3)),0,0,0)

    def engine(self,size):
        engine=DirectOriginalEngine.__new__(DirectOriginalEngine)
        engine.args=SimpleNamespace(num_hidden_layers=2)
        engine.prefill_schedule="chunk-major"
        engine.prefill_chunk_size=size
        engine.prefill_chunks=0
        engine.mx=None
        self.calls=[]
        def forward(inputs,cache,project,layer_chunk_size=None):
            self.layer_chunk_size=layer_chunk_size
            self.calls.append((inputs.tolist(),[c.offset for c in cache],project))
            for c in cache:c.offset+=inputs.shape[1]
            return 'final logits' if project else None
        engine._forward_chunk=forward
        return engine

    def test_nondivisible_prompt_keeps_order_offsets_and_one_final_projection(self):
        engine=self.engine(3);cache=[SimpleNamespace(offset=5) for _ in range(2)]
        self.assertEqual(engine(np.array([[10,11,12,13,14,15,16]]),cache),'final logits')
        self.assertEqual(self.calls,[([[10,11,12]],[5,5],False),([[13,14,15]],[8,8],False),([[16]],[11,11],True)])
        self.assertEqual(engine.prefill_chunks,3)
        self.assertEqual([c.offset for c in cache],[12,12])

    def test_layer_major_sends_whole_prompt_to_one_layer_stream(self):
        engine=self.engine(3);engine.prefill_schedule='layer-major'
        cache=[SimpleNamespace(offset=5) for _ in range(2)]
        self.assertEqual(engine(np.array([[10,11,12,13,14,15,16]]),cache),'final logits')
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.layer_chunk_size,3)
        self.assertEqual(engine.prefill_chunks,3)
        self.assertEqual([c.offset for c in cache],[12,12])

    def test_decode_is_single_pass_and_not_counted_as_prefill(self):
        engine=self.engine(8)
        engine(np.array([[9]]),[SimpleNamespace(offset=30) for _ in range(2)])
        self.assertEqual(len(self.calls),1)
        self.assertTrue(self.calls[0][2])
        self.assertEqual(engine.prefill_chunks,0)

    def test_misaligned_layer_caches_rejected_before_work(self):
        engine=self.engine(8)
        with self.assertRaises(ValueError):engine(np.array([[9]]),[SimpleNamespace(offset=2),SimpleNamespace(offset=3)])
        self.assertEqual(self.calls,[])

    def test_empty_prompt_and_context_overflow_rejected(self):
        engine=self.engine(8)
        for inputs,offset in [(np.empty((1,0),dtype=int),0),(np.ones((1,128),dtype=int),129)]:
            with self.assertRaises(ValueError):engine(inputs,[SimpleNamespace(offset=offset) for _ in range(2)])
        self.assertEqual(self.calls,[])
