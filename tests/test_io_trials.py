"""Trial planning preserves every mode; incomplete jobs never gain full metrics."""
import sys
from pathlib import Path
import unittest
import hashlib
import json
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_io_trials import order, summarize
from analyze_io_trials import analyze


class IoTrialTests(unittest.TestCase):
    def test_randomized_order_keeps_every_mode_in_each_round(self):
        protocol={'seed':4381,'rounds':2,'modes':['native','raw-cached','raw-nocache']}
        jobs=order(protocol)
        self.assertEqual(jobs,order(protocol))
        for round_index in [0,1]:
            self.assertEqual(sorted(j['mode'] for j in jobs if j['round']==round_index),sorted(protocol['modes']))

    def test_partial_summary_keeps_stop_and_observed_peak(self):
        report={'completed':False,'stopped_by_guard':'swap_growth_over_512MiB',
                'worker_elapsed_seconds':20,'verification_seconds':3,
                'sampled_process_tree_rss_peak_bytes':123,
                'events':[{'event':'step','case':0,'step':0,'token_id':7,
                           'elapsed_seconds':10,'mlx_peak_bytes_so_far':456}],
                'measurements':[{'system_swap_growth_bytes':789}]}
        s=summarize(report)
        self.assertFalse(s['completed'])
        self.assertEqual(s['guard'],'swap_growth_over_512MiB')
        self.assertEqual(s['peak_mlx_bytes'],456)
        self.assertIsNone(s['cases'])
        self.assertEqual(s['peak_mlx_scope'],'observed completed steps only')

    def test_analysis_keeps_failures_beside_conditional_medians(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=json.dumps({'modes':['native','raw-cached','raw-nocache']}).encode()
            (root/'protocol.json').write_bytes(protocol)
            base={'round':0,'forward_seconds':10,'peak_mlx_bytes':100,
                  'sampled_rss_peak_bytes':200,'max_system_swap_growth_bytes':0,
                  'verification_swap_delta_bytes':0,'cases':[{'token_ids':[7]}]}
            trials=[{**base,'mode':'native','completed':True},
                    {**base,'mode':'raw-cached','completed':False,'forward_seconds':1},
                    {**base,'mode':'raw-nocache','completed':True}]
            (root/'summary.json').write_text(json.dumps({'completed_series':True,'source_sha256':{},
                'protocol_sha256':hashlib.sha256(protocol).hexdigest(),
                'planned_order':[{}, {}, {}],'trials':trials}))
            result=analyze(root)
            self.assertIsNone(result['mode_statistics']['raw-cached']['completed_only_median_forward_seconds'])
            self.assertEqual(result['mode_statistics']['raw-cached']['stopped'],1)
            self.assertIsNone(result['same_round_comparisons'][0]['greedy_token_sequences_equal'])
            self.assertTrue(result['same_round_comparisons'][1]['greedy_token_sequences_equal'])


if __name__=='__main__': unittest.main()
