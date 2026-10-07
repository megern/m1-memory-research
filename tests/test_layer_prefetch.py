import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from layer_prefetch import LayerReader


class PrefetchTest(unittest.TestCase):
    def test_order_budget_and_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(5):
                (root/f'layer-{index:03d}.safetensors').write_bytes(bytes([index])*1024)
            for ahead in (0, 1, 2):
                with LayerReader(root, 5, ahead) as reader:
                    for _ in range(2):
                        reader.begin()
                        with self.assertRaises(ValueError):
                            reader.take(1)
                        for index in range(5):
                            data, read, wait = reader.take(index)
                            self.assertEqual(data, bytes([index])*1024)
                            self.assertGreaterEqual(read, 0)
                            self.assertGreaterEqual(wait, 0)
                            self.assertLessEqual(len(reader.pending), ahead+1)
                        self.assertFalse(reader.pending)
                        with self.assertRaises(ValueError):
                            reader.take(5)

    def test_failure_does_not_hang(self):
        with tempfile.TemporaryDirectory() as temporary:
            with LayerReader(temporary, 2, 1) as reader:
                reader.begin()
                with self.assertRaises(FileNotFoundError):
                    reader.take(0)
            self.assertFalse(reader.pending)


if __name__ == '__main__':
    unittest.main()
