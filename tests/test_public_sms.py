import importlib.util
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prep=load('prepare_public_sms');study=load('public_sms_study')
analysis=load('analyze_public_sms')
planner=load('plan_sms_training_controls')
class PublicSMS(unittest.TestCase):
    def test_control_plan_cannot_be_created_after_new_test_results(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);(directory/'seed-17.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'must precede'):planner.freeze(directory,directory/'missing-controls')
    def test_template_duplicates_form_one_group(self):
        rows=[{'id':'a','text':'Claim your prize 123 at https://example.org/a','label':'spam'},{'id':'b','text':'Claim your prize 456 at https://example.org/b','label':'spam'},{'id':'c','text':'Meet me at the station tomorrow','label':'ham'}]
        clusters,_=prep.groups(rows)
        self.assertEqual(sorted(map(len,clusters)),[1,2])
    def test_conflicting_labels_remain_visible_in_group(self):
        clusters,_=prep.groups([{'text':'same text','label':'ham'},{'text':'same text','label':'spam'}])
        self.assertEqual({r['label'] for r in clusters[0]},{'ham','spam'})
    def test_macro_f1_not_majority_accuracy(self):
        rows=[{'label':'ham','prediction':'ham'}]*9+[{'label':'spam','prediction':'ham'}]
        self.assertAlmostEqual(study.metrics(rows)['accuracy'],.9)
        self.assertAlmostEqual(study.metrics(rows)['macro_f1'],9/19)
        self.assertEqual(study.metrics(rows)['spam_recall'],0)
    def test_lexical_fit_ignores_test_labels(self):
        row=lambda i,t,l:{'id':i,'label':l,'messages':prep.messages(t)}
        train=[row('a','win prize','spam'),row('b','meet tomorrow','ham')]
        first=study.lexical(train,[row('c','win prize','ham')])[0]
        second=study.lexical(train,[row('c','win prize','spam')])[0]
        self.assertEqual(first['margin'],second['margin']);self.assertEqual(first['prediction'],'spam')
    def test_zero_observed_errors_do_not_imply_certainty(self):
        self.assertGreater(analysis.wilson(0,230)[1],0)
        self.assertLess(analysis.wilson(26,26)[0],1)
        self.assertIsNone(analysis.wilson(0,0))
    def test_paired_identical_predictions_cancel_case_uncertainty(self):
        rows=[{'id':str(i),'label':label,'prediction':prediction} for i,(label,prediction) in enumerate([('spam','spam'),('spam','ham'),('ham','ham'),('ham','spam')])]
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for name in ('base','seed-17'):(directory/(name+'.json')).write_text(json.dumps({'complete':True,'rows':rows}))
            with contextlib.redirect_stdout(io.StringIO()):analysis.analyze(directory)
            report=json.loads((directory/'quality-analysis.json').read_text())
            self.assertEqual(report['paired_macro_f1_delta_vs_base']['seed-17']['ci95'],[0.,0.])
    def test_case_order_drift_is_rejected(self):
        rows=[{'id':'a','label':'spam','prediction':'spam'},{'id':'b','label':'ham','prediction':'ham'}]
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            (directory/'base.json').write_text(json.dumps({'complete':True,'rows':rows}))
            (directory/'seed-17.json').write_text(json.dumps({'complete':True,'rows':rows[::-1]}))
            with self.assertRaisesRegex(ValueError,'Pair mismatch'):analysis.analyze(directory)
    def test_fresh_test_excludes_all_previously_used_groups(self):
        old=json.loads((ROOT/'data/public-sms-v1/protocol.json').read_text())
        new=json.loads((ROOT/'data/public-sms-followup-v1/protocol.json').read_text())
        used={i for split in old['splits'].values() for group in split['source_group_ids'].values() for i in group}
        fresh={i for group in new['splits']['test']['source_group_ids'].values() for i in group}
        self.assertFalse(used&fresh)
        for split in ('train','valid'):self.assertEqual(old['splits'][split]['sha256'],new['splits'][split]['sha256'])
    def test_all_frozen_source_identities_match_exports(self):
        for name in ('public-sms-study-v1','public-sms-study-v2','public-sms-followup-v1','sms-factorial-pilot-v1','sms-confirmation-v1'):
            root=ROOT/'results'/name;protocol=json.loads((root/'protocol.json').read_text())
            for file,digest in protocol.get('frozen_source_sha256',protocol.get('frozen_sources',{})).items():
                with self.subTest(study=name,file=file):self.assertEqual(hashlib.sha256((root/'source'/file).read_bytes()).hexdigest(),digest)
    def test_export_contains_no_raw_messages_or_weight_files(self):
        for root in list((ROOT/'results').glob('public-sms-*'))+list((ROOT/'results').glob('sms-factorial-pilot-*'))+list((ROOT/'results').glob('sms-confirmation-*')):
            self.assertFalse(list(root.rglob('*.safetensors')))
            self.assertFalse(list(root.rglob('*.jsonl')))
            def check(value):
                if isinstance(value,dict):
                    self.assertFalse({'messages','sms','text'}&value.keys())
                    for child in value.values():check(child)
                elif isinstance(value,list):
                    for child in value:check(child)
            for file in root.rglob('*.json'):check(json.loads(file.read_text()))
if __name__=='__main__':unittest.main()
