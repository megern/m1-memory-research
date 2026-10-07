from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from budgeted_original_engine import cache_plan


class CachePlanTest(unittest.TestCase):
    def test_budget_includes_replacement_and_reserve(self):
        plan = cache_plan(1000, 300, 100, 28, reserve=100)
        self.assertEqual(plan['cached_layers'], 4)
        self.assertLessEqual(plan['estimated_persistent_plus_replacement_bytes']+plan['reserved_bytes'], 1000)
        self.assertEqual(cache_plan(1000, 300, 100, 28, cap=0, reserve=100)['cached_layers'], 0)
        self.assertEqual(cache_plan(100000, 300, 100, 28, reserve=100)['cached_layers'], 28)
        with self.assertRaises(ValueError):
            cache_plan(599, 300, 100, 28, reserve=100)
