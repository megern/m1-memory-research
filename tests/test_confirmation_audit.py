import hashlib, json, shutil, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from audit_sms_confirmation import audit

class ConfirmationAuditTests(unittest.TestCase):
    def test_published_export(self):
        self.assertEqual(audit(ROOT / 'results/sms-confirmation-v1')['cells_verified'], 16)
    def test_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'export'
            shutil.copytree(ROOT/'results/sms-confirmation-v1', target)
            (target/'protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): audit(target)
    def test_rehashed_false_metric_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'export'
            shutil.copytree(ROOT/'results/sms-confirmation-v1', target)
            name='seed23-n120-lr0.0001-a1-valid.json';path=target/name
            data=json.loads(path.read_text());data['metrics']['accuracy']=.123
            path.write_text(json.dumps(data))
            manifest=json.loads((target/'export-manifest.json').read_text())
            manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            (target/'export-manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Metrics disagree'): audit(target)
    def test_unmanifested_selected_report_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'export'
            shutil.copytree(ROOT/'results/sms-confirmation-v1',target)
            path=target/'export-manifest.json';manifest=json.loads(path.read_text())
            del manifest['seed23-n120-lr0.0001-a1-valid.json'];path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'missing from manifest'):audit(target)
    def test_complete_report_retry_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'export'
            shutil.copytree(ROOT/'results/sms-confirmation-v1',target)
            shutil.copy2(target/'seed23-n120-lr0.0001-a1-valid.json',target/'seed23-n120-lr0.0001-a1-valid-retry.json')
            with self.assertRaisesRegex(ValueError,'cannot be replaced'):audit(target)
