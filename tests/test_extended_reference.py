"""Invalid reference dependencies must fail before hashing or GPU model loading."""
import json,sys,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from make_extended_reference import run

class ExtendedReferenceTests(unittest.TestCase):
    def files(self,root):
        manifest={'repository':'Qwen/Qwen3-0.6B','revision':'fixture'}
        (root/'manifest.json').write_text(json.dumps(manifest));(root/'protocol.json').write_text(json.dumps(manifest))
        return root/'manifest.json',root/'protocol.json'

    def test_missing_baseline_stops_before_verification_and_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,protocol=self.files(root)
            with patch('make_extended_reference.verify_guarded') as verify:
                with self.assertRaises(FileNotFoundError):run(root,manifest,protocol,root/'output',32,root/'missing')
                verify.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_different_model_protocol_stops_before_resource_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,protocol=self.files(root)
            protocol.write_text(json.dumps({'repository':'Qwen/Qwen3-0.6B','revision':'different'}))
            with patch('make_extended_reference.verify_guarded') as verify:
                with self.assertRaises(ValueError):run(root,manifest,protocol,root/'output',1024)
                verify.assert_not_called()

if __name__=='__main__':unittest.main()
