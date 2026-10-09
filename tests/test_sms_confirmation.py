import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from plan_sms_confirmation import plan
class ConfirmationTests(unittest.TestCase):
 def test_complete_design(self):
  p=plan();self.assertEqual(len(p['cells']),16);self.assertEqual(len({c['id'] for c in p['cells']}),16)
  self.assertEqual({c['seed'] for c in p['cells']},{23,41})
  self.assertEqual(p['status'],'planned_not_executed');self.assertFalse(p['test_evaluation'])
 def test_update_counts(self):
  self.assertTrue(all(c['optimizer_updates']*c['gradient_accumulation_steps']==c['iterations'] for c in plan()['cells']))
