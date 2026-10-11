import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('resource_audit', Path(__file__).resolve().parents[1] / 'tools/audit_training_resources.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ResourceAuditTests(unittest.TestCase):
    def report(self):
        return {'identity': {'train_sha256': 'a' * 64, 'valid_sha256': 'b' * 64,
                 'model_files': {'weights': 'c' * 64}, 'versions': {'mlx': 'fixture'},
                 'config': {'seed': 1, 'rank': 4, 'lora_scale': 1, 'learning_rate': .001,
                            'microbatch_size': 1, 'accumulation_steps': 2, 'max_seq_length': 32}},
                'peak_optimizer_resident_bytes': 100, 'mlx_peak_bytes_this_process_segment': 1000,
                'process_rss_high_water_bytes': 1200, 'end_to_end_seconds_this_process_segment': 10,
                'examples_consumed': 16, 'tokens': 32, 'heldout_nll_before': 2, 'heldout_nll_after': 1,
                'layer_update_coverage': [8] * 4}

    def test_optimizer_saving_not_confused_with_total_allocation(self):
        left = self.report();right = copy.deepcopy(left)
        right['peak_optimizer_resident_bytes'] = 25
        right['mlx_peak_bytes_this_process_segment'] = 990
        result = audit.compare(left, right)
        self.assertEqual(result['measurements']['peak_optimizer_resident_bytes']['reduction_percent'], 75)
        self.assertAlmostEqual(result['measurements']['mlx_peak_bytes_this_process_segment']['reduction_percent'], 1)
        self.assertTrue(result['fairness']['same_layer_update_coverage'])

    def test_unequal_coverage_remains_explicit(self):
        right = self.report();right['layer_update_coverage'] = [2] * 4
        self.assertFalse(audit.compare(self.report(), right)['fairness']['same_layer_update_coverage'])

    def test_unmatched_inputs_and_seed_rejected(self):
        for field in ('train_sha256', 'valid_sha256', 'model_files', 'versions'):
            right = self.report();right['identity'][field] = 'different'
            with self.assertRaisesRegex(ValueError, 'Unmatched'):
                audit.compare(self.report(), right)
        right = self.report();right['identity']['config']['seed'] = 2
        with self.assertRaisesRegex(ValueError, 'seed'):
            audit.compare(self.report(), right)

    def test_nan_boolean_and_bad_coverage_rejected(self):
        for value in (float('nan'), True, 0):
            right = self.report();right['mlx_peak_bytes_this_process_segment'] = value
            with self.assertRaises(ValueError):audit.compare(self.report(), right)
        right = self.report();right['layer_update_coverage'] = [True] * 4
        with self.assertRaises(ValueError):audit.compare(self.report(), right)
