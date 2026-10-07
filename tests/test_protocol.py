import json
import hashlib
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
from evaluate_specialist import parse_label,score
from run_large_model import run
from provenance import verify_base

class ProtocolTests(unittest.TestCase):
 def test_format_and_classification_are_distinct(self):
  output='```json\n{"label":"high"}\n```'
  self.assertIsNone(parse_label(output))
  self.assertEqual(parse_label(output,allow_fence=True),'high')
  for invalid in ['The answer is high.', '{"label":"high","action":"delete"}',
                  '{"label":"HIGH"}', '```json\n{"label":"high"}\n``` trailing']:
   self.assertIsNone(parse_label(invalid,allow_fence=True))
 def test_macro_f1_exposes_majority_collapse(self):
  rows=[{'expected':'low','prediction':'low','valid_json':True,'finish_reason':'stop'} for _ in range(8)]
  rows += [{'expected':'high','prediction':'low','valid_json':True,'finish_reason':'stop'} for _ in range(2)]
  summary=score(rows)
  self.assertEqual(summary['accuracy'],0.8)
  self.assertLess(summary['macro_f1'],0.5)
 def test_data_splits_and_class_balance_are_frozen(self):
  protocol=json.loads((root/'data'/'confirmatory-v2'/'protocol.json').read_text())
  for task,labels in [('auth',3),('workflow',2)]:
   seen=set()
   for split in ['train','valid','test']:
    path=root/'data'/'confirmatory-v2'/task/(split+'.jsonl')
    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),protocol['datasets'][task][split]['sha256'])
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    ids={r['id'] for r in rows}
    self.assertEqual(len(ids),len(rows))
    self.assertFalse(ids & seen)
    seen |= ids
    counts={label:sum(r['label']==label for r in rows) for label in {r['label'] for r in rows}}
    self.assertEqual(len(counts),labels)
    self.assertEqual(len(set(counts.values())),1)
 def test_projected_data_retains_raw_evidence_and_balances_languages(self):
  from normalized_auth import features, rule_label
  protocol=json.loads((root/'data/normalized-auth/protocol.json').read_text())
  for split in ['train','valid','test']:
   path=root/'data/normalized-auth/auth'/(split+'.jsonl')
   self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),protocol['datasets']['auth'][split]['sha256'])
   cases=[json.loads(line) for line in path.read_text().splitlines()]
   for case in cases:
    self.assertEqual(rule_label(case['raw_evidence']),case['label'])
    projected=json.loads(case['messages'][1]['content'].splitlines()[-1])
    self.assertEqual(projected,features(case['raw_evidence']))
   counts=[sum(r['language']==lang and r['label']==label for r in cases)
           for lang in ['ar','en'] for label in ['low','medium','high']]
   self.assertEqual(len(set(counts)),1)
  self.assertEqual(rule_label({'failed_logins':5,'window_seconds':300,'subsequent_success':True}),'high')
  self.assertEqual(rule_label({'failed_logins':5,'window_seconds':301,'subsequent_success':True}),'low')
 def test_unverified_weights_never_launch_runtime(self):
  with tempfile.TemporaryDirectory() as temp:
   directory=Path(temp)
   model=directory/'model.gguf'; model.write_bytes(b'not verified model weights')
   runtime=directory/'runtime'; runtime.write_text('unused')
   with self.assertRaisesRegex(ValueError,'checksum'):
    run(runtime,model,{'bytes':model.stat().st_size,'sha256':'0'*64},directory/'report')
   self.assertFalse((directory/'report').exists())
 def test_unknown_base_is_rejected_before_model_execution(self):
  with tempfile.TemporaryDirectory() as temp:
   directory=Path(temp)
   (directory/'config.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'not one of this study'):
    verify_base(directory)
 def test_regeneration_matches_recorded_data_and_preserves_existing_freeze(self):
  with tempfile.TemporaryDirectory() as temp:
   destination=Path(temp)/'data'
   command=[sys.executable,str(root/'tools/make_confirmatory_data.py'),'--output',str(destination)]
   subprocess.run(command,check=True,capture_output=True)
   protocol=json.loads((root/'data/confirmatory-v2/protocol.json').read_text())
   for task in ['auth','workflow']:
    for split in ['train','valid','test']:
     path=destination/task/(split+'.jsonl')
     self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),protocol['datasets'][task][split]['sha256'])
   before=(destination/'protocol.json').read_bytes()
   refusal=subprocess.run(command,capture_output=True)
   self.assertNotEqual(refusal.returncode,0)
   self.assertEqual((destination/'protocol.json').read_bytes(),before)

if __name__=='__main__':
 unittest.main()
