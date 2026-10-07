from pathlib import Path
import subprocess
import sys
import unittest


class OriginalCliBoundsTest(unittest.TestCase):
    def test_invalid_bounds_fail_before_model_loading(self):
        script = Path(__file__).resolve().parents[1]/'tools/local_original_inference.py'
        for flags in (['--tokens', '65'], ['--budget-mib', '2049']):
            result = subprocess.run([sys.executable, str(script), '--model', '/missing-model', *flags],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Budget must be', result.stderr)
            self.assertNotIn('Traceback', result.stderr)
