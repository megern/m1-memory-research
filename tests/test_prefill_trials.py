"""Retain setup failures and refuse partial studies before speed comparisons."""
import sys,json,tempfile,hashlib
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_prefill_trials import order,summarize
from analyze_prefill_trials import analyze

class PrefillTrialTests(unittest.TestCase):
    def test_every_chunk_setting_once_per_round(self):
        p={'seed':8043,'rounds':2,'chunk_sizes':[128,32]}
        jobs=order(p)
        self.assertEqual(jobs,order(p))
        for r in range(2):self.assertEqual(sorted(t['chunk_size'] for t in jobs if t['round']==r),[32,128])

    def test_named_schedules_keep_all_variants(self):
        p={'seed':8044,'rounds':2,'variants':{'unchunked':{},'chunk-major32':{},'layer-major32':{}}}
        jobs=order(p)
        for r in range(2):self.assertEqual(sorted(t['variant'] for t in jobs if t['round']==r),sorted(p['variants']))

    def test_verification_failure_retains_setup_pressure(self):
        r={'completed':False,'stopped_by_guard':'verification_swap_growth_over_512MiB','worker_elapsed_seconds':0,'verification_seconds':3,'events':[],'measurements':[],'verification_measurements':[{'system_swap_growth_bytes':600*1024**2}],'sampled_process_tree_rss_peak_bytes':0}
        s=summarize(r)
        self.assertFalse(s['completed']);self.assertIsNone(s['cases'])
        self.assertEqual(s['max_setup_and_worker_swap_growth_bytes'],600*1024**2)
        self.assertIsNone(s['layer_calls'])

    def test_failed_attempt_never_ranked_as_fast_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=json.dumps({'chunk_sizes':[128,32]}).encode();(root/'protocol.json').write_bytes(protocol)
            base={'round':0,'forward_seconds':10,'peak_mlx_bytes':100,'sampled_rss_peak_bytes':200,'selected_tensor_bytes':300,'row_bytes_read':400,'max_setup_and_worker_swap_growth_bytes':0,'cases':[{'token_ids':[7]}]}
            trials=[{**base,'chunk_size':128,'completed':True},{**base,'chunk_size':32,'completed':False,'forward_seconds':1}]
            summary={'completed_series':True,'protocol_sha256':hashlib.sha256(protocol).hexdigest(),'source_sha256':{},'planned_order':trials,'trials':trials}
            (root/'summary.json').write_text(json.dumps(summary));a=analyze(root)
            self.assertIsNone(a['chunk_statistics']['32']['completed_only_median_forward_seconds'])
            self.assertIsNone(a['same_round_comparisons'][0]['greedy_token_sequences_equal'])
            summary['trials']=trials[:1];(root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):analyze(root)

if __name__=='__main__':unittest.main()
