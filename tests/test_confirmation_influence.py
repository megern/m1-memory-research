import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from analyze_confirmation_influence import influence
class InfluenceTests(unittest.TestCase):
    def test_case_sensitivity_preserves_confirmation_direction(self):
        report=influence(ROOT/'results/sms-confirmation-v1')
        self.assertEqual(len(report['removed_cases']),128)
        self.assertAlmostEqual(report['baseline_two_seed_mean_effects']['iterations'],.2026625604295187)
        self.assertAlmostEqual(report['baseline_two_seed_mean_effects']['learning_rate'],-.10550907352590971)
        for factor,limits in report['leave_one_out'].items():
            self.assertEqual(limits['direction_changes'],0)
            self.assertLessEqual(limits['minimum'],limits['maximum'])
