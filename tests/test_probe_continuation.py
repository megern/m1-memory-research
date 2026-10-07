"""Failed acquisitions must not launch a model; execute only the frozen source."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from wait_original_probe import continue_probe


class ContinuationTests(unittest.TestCase):
    def fixture(self, root, status):
        model=root/'model';model.mkdir()
        identity={'repository':'fixture/never-executed','revision':'test'}
        manifest=root/'manifest.json';manifest.write_text(json.dumps(identity))
        protocol=root/'protocol.json'
        protocol.write_text(json.dumps({**identity,'tokens_per_case':4,'budget_mib':1024,
                                       'head_rows':2048,'timeout_seconds':900,'prompts':['fixture']}))
        (model/'acquisition-state.json').write_text(json.dumps({**identity,'status':status}))
        return model,manifest,protocol,root/'output'

    def test_failed_download_never_launches_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),'failed_resumable')
            with patch('subprocess.call') as call,self.assertRaises(RuntimeError):
                continue_probe(*args)
            call.assert_not_called()
            self.assertEqual(json.loads((args[-1]/'status.json').read_text())['stage'],'failed')

    def test_completed_acquisition_invokes_frozen_source_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),'verified_complete')
            with patch('subprocess.call',return_value=1) as call:
                self.assertEqual(continue_probe(*args),1)
            command=call.call_args.args[0]
            self.assertEqual(command[1],str(args[-1]/'source/run_direct_original.py'))
            self.assertIn(str(args[-1]/'manifest.json'),command)
            state=json.loads((args[-1]/'status.json').read_text())
            self.assertFalse(state['completed'])
            self.assertEqual(state['stage'],'stopped_or_failed')


if __name__ == '__main__': unittest.main()
