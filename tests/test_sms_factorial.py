import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from run_sms_factorial_pilot import cells,summarize
from resume_sms_factorial_pilot import resume
from analyze_sms_factorial_effects import analyze
class FactorialPilot(unittest.TestCase):
    def test_shared_case_bootstrap_preserves_zero_policy_contrasts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);design=cells();(root/'protocol.json').write_text(json.dumps({'cells':design,'splits_sha256':{'valid':'shared-validation'}}))
            rows=[{'id':str(i),'label':label,'prediction':pred} for i,(label,pred) in enumerate([('ham','ham'),('ham','spam'),('spam','ham'),('spam','spam')])]
            for cell in design:(root/(cell['id']+'-valid.json')).write_text(json.dumps({'complete':True,'split':'valid','data_sha256':'shared-validation','rows':rows}))
            with contextlib.redirect_stdout(io.StringIO()):analyze(root)
            report=json.loads((root/'factor-effects-analysis.json').read_text())
            for contrast in report['contrasts'].values():
                self.assertAlmostEqual(contrast['point'],0)
                for bound in contrast['paired_case_ci95']:self.assertAlmostEqual(bound,0)
    def test_complete_balanced_factorial_and_optimizer_counts(self):
        design=cells();self.assertEqual(len(design),8)
        self.assertEqual(len({(r['iterations'],r['learning_rate'],r['gradient_accumulation_steps']) for r in design}),8)
        for row in design:self.assertEqual(row['optimizer_updates']*row['gradient_accumulation_steps'],row['iterations'])
        for key in ('iterations','learning_rate','gradient_accumulation_steps'):
            for level in {r[key] for r in design}:self.assertEqual(sum(r[key]==level for r in design),4)
    def test_incomplete_cells_do_not_produce_biased_factor_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);design=cells();(root/'protocol.json').write_text(json.dumps({'cells':design}));(root/'journal.json').write_text('{}')
            for index,row in enumerate(design):(root/(row['id']+'-valid.json')).write_text(json.dumps({'complete':index!=0,'metrics':{'macro_f1':.5}}))
            with contextlib.redirect_stdout(io.StringIO()):summarize(root)
            report=json.loads((root/'summary.json').read_text());self.assertEqual(report['completed_validation_cells'],7);self.assertEqual(report['descriptive_factor_effects'],{})
    def test_factor_effects_and_no_test_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);design=cells();(root/'protocol.json').write_text(json.dumps({'cells':design}));(root/'journal.json').write_text('{}');(root/'test.json').write_text('must not be read')
            for row in design:
                score=.4+.1*(row['iterations']==480)+.05*(row['learning_rate']==.0002)-.02*(row['gradient_accumulation_steps']==4)
                (root/(row['id']+'-valid.json')).write_text(json.dumps({'complete':True,'metrics':{'macro_f1':score}}))
            with contextlib.redirect_stdout(io.StringIO()):summarize(root)
            report=json.loads((root/'summary.json').read_text());effects=report['descriptive_factor_effects']
            self.assertAlmostEqual(effects['iterations']['second_minus_first'],.1)
            self.assertAlmostEqual(effects['gradient_accumulation_steps']['second_minus_first'],-.02)
            self.assertFalse(report['test_evaluated']);self.assertFalse(report['best_model_selected'])
    def test_no_favorable_retry_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row=cells()[0];(root/'protocol.json').write_text(json.dumps({'cells':[row]}));(root/'journal.json').write_text('{}')
            (root/(row['id']+'-valid.json')).write_text(json.dumps({'complete':True,'metrics':{'macro_f1':.4}}));(root/(row['id']+'-valid-retry.json')).write_text(json.dumps({'complete':True,'metrics':{'macro_f1':1.}}))
            with contextlib.redirect_stdout(io.StringIO()):summarize(root)
            report=json.loads((root/'summary.json').read_text());self.assertEqual(report['cells'][row['id']]['metrics']['macro_f1'],.4)
    def test_only_one_disclosed_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'protocol.json').write_text('{}');(root/'continuation-protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'One continuation'):resume(root,root,root)
if __name__=='__main__':unittest.main()
