"""Setup pressure must stop before launching a model; keep the original baseline."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_direct_original import run, verify_guarded


class VerificationGuardTests(unittest.TestCase):
    def test_initial_swap_spike_stops_before_hashing(self):
        with patch('run_direct_original.psutil.swap_memory',side_effect=[SimpleNamespace(used=0),SimpleNamespace(used=600*1024**2)]),patch('run_direct_original.psutil.virtual_memory',return_value=SimpleNamespace(available=2*1024**3)),patch('run_direct_original.psutil.Process') as process,patch('run_direct_original.verify_files') as verify:
            process.return_value.memory_info.return_value.rss=1024
            process.return_value.cpu_times.return_value=(0,0)
            process.return_value.num_threads.return_value=1
            identities,phase=verify_guarded(Path('.'),{},900)
        self.assertIsNone(identities)
        self.assertEqual(phase['stopped_by_guard'],'verification_swap_growth_over_512MiB')
        verify.assert_not_called()

    def test_completed_verification_retains_pre_hash_baseline(self):
        with patch('run_direct_original.psutil.swap_memory',side_effect=[SimpleNamespace(used=10),SimpleNamespace(used=20),SimpleNamespace(used=30)]),patch('run_direct_original.psutil.virtual_memory',return_value=SimpleNamespace(available=2*1024**3)),patch('run_direct_original.psutil.Process') as process,patch('run_direct_original.verify_files',return_value={'fixture':[1,2,3]}):
            process.return_value.memory_info.return_value.rss=1024
            process.return_value.cpu_times.return_value=(0,0)
            process.return_value.num_threads.return_value=1
            identities,phase=verify_guarded(Path('.'),{},900)
        self.assertEqual(identities,{'fixture':[1,2,3]})
        self.assertEqual(phase['baseline_system_swap_bytes'],10)
        self.assertEqual(phase['samples'][-1]['system_swap_growth_bytes'],20)

    def test_stopped_verification_writes_report_never_launches_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'repository':'fixture/never-executed','revision':'test'}))
            phase={'seconds':1,'stopped_by_guard':'verification_swap_growth_over_512MiB',
                   'error_type':None,'io_policy':'cached','samples':[]}
            with patch('run_direct_original.verify_guarded',return_value=(None,phase)),patch('run_direct_original.subprocess.Popen') as launch:
                report=run(root,manifest,root/'output',['fixture'])
            launch.assert_not_called()
            self.assertFalse(report['model_started'])
            self.assertFalse(report['completed'])
            self.assertTrue((root/'output/report.json').exists())


if __name__=='__main__':unittest.main()
