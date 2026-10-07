"""Verify that layer packaging preserves tensor bytes and prevents overwrite."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from shard_original_layers import shard


class OriginalShardsTest(unittest.TestCase):
    def test_byte_preserving_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'model'
            source.mkdir()
            tensors = {
                'model.embed_tokens.weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
                'model.layers.0.self_attn.q_proj.weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [4, 8]},
                'model.layers.1.mlp.down_proj.weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [8, 12]},
            }
            header = json.dumps(tensors).encode()
            payload = bytes(range(12))
            (source/'model.safetensors').write_bytes(struct.pack('<Q', len(header))+header+payload)
            output = root/'shards'
            # Synthetic transport fixture bypasses only upstream identity verification.
            # No synthetic model execution or benchmark result is claimed.
            with patch('shard_original_layers.inspect', return_value={'fixture': True}):
                shard(source, output)
            recovered = {}
            for path in output.glob('*.safetensors'):
                with path.open('rb') as f:
                    n = struct.unpack('<Q', f.read(8))[0]
                    h = json.loads(f.read(n))
                    raw = f.read()
                for name, tensor in h.items():
                    a, b = tensor['data_offsets']
                    recovered[name] = raw[a:b]
                    self.assertEqual(tensor['dtype'], tensors[name]['dtype'])
                    self.assertEqual(tensor['shape'], tensors[name]['shape'])
            self.assertEqual(set(recovered), set(tensors))
            for name, tensor in tensors.items():
                a, b = tensor['data_offsets']
                self.assertEqual(recovered[name], payload[a:b])
            self.assertEqual(len(json.loads((output/'manifest.json').read_text())['shards']), 3)
            with self.assertRaises(ValueError):
                shard(source, output)


if __name__ == '__main__':
    unittest.main()
