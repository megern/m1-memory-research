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
 def test_partial_seed_has_no_effect_estimate(self):
  import tempfile,json
  from run_sms_confirmation import summarize
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cells=plan()['cells'];(p/'protocol.json').write_text(json.dumps({'cells':cells,'limitations':'validation only'}))
   c=cells[0];(p/(c['id']+'-valid.json')).write_text(json.dumps({'complete':True,'metrics':{'macro_f1':.8}}))
   summarize(p);r=json.loads((p/'summary.json').read_text())
   self.assertEqual(r['completed_validation_cells'],1);self.assertEqual(r['per_seed_factor_effects'],{})
 def test_incomplete_retry_does_not_replace_complete(self):
  import tempfile,json
  from run_sms_confirmation import summarize
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cells=plan()['cells'];(p/'protocol.json').write_text(json.dumps({'cells':cells,'limitations':'validation only'}))
   c=cells[0];(p/(c['id']+'-valid.json')).write_text(json.dumps({'complete':True,'metrics':{'macro_f1':.7}}))
   (p/(c['id']+'-valid-retry.json')).write_text(json.dumps({'complete':True,'metrics':{'macro_f1':.99}}))
   summarize(p);r=json.loads((p/'summary.json').read_text())
   self.assertEqual(r['cells'][c['id']]['metrics']['macro_f1'],.7)
 def test_analysis_rejects_missing_cells(self):
  import tempfile,json
  from analyze_sms_confirmation import analyze
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'protocol.json').write_text(json.dumps({'cells':plan()['cells']}))
   with self.assertRaises(ValueError):analyze(p)
 def test_analysis_rejects_metric_not_matching_predictions(self):
  import tempfile,json
  from analyze_sms_confirmation import analyze
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cells=plan()['cells'];(p/'protocol.json').write_text(json.dumps({'cells':cells,'splits_sha256':{'valid':'data-hash'}}))
   rows=[{'id':'a','label':'ham','prediction':'ham'},{'id':'b','label':'spam','prediction':'spam'}]
   (p/(cells[0]['id']+'-valid.json')).write_text(json.dumps({'complete':True,'split':'valid','data_sha256':'data-hash','rows':rows,'metrics':{'macro_f1':.5}}))
   with self.assertRaises(ValueError):analyze(p)
 def test_complete_paired_analysis_has_zero_contrasts_for_identical_vectors(self):
  import tempfile,json
  from analyze_sms_confirmation import analyze
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cells=plan()['cells'];(p/'protocol.json').write_text(json.dumps({'cells':cells,'splits_sha256':{'valid':'data-hash'}}))
   rows=[{'id':str(i),'label':'spam' if i<13 else 'ham','prediction':'spam' if i<13 else 'ham'} for i in range(128)]
   for cell in cells:(p/(cell['id']+'-valid.json')).write_text(json.dumps({'complete':True,'split':'valid','data_sha256':'data-hash','rows':rows,'metrics':{'macro_f1':1.}}))
   analyze(p);result=json.loads((p/'confirmation-analysis.json').read_text())
   self.assertEqual(result['completed_cells'],16)
   self.assertEqual(len(result['contrasts']),18)
   for contrast in result['contrasts'].values():
    self.assertEqual(contrast['point'],0.);self.assertEqual(contrast['conditional_paired_case_ci95'],[0.,0.])
