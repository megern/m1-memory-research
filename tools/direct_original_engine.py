"""Original Qwen3 BF16 streaming with row lookups and tiled output projection.

Does not duplicate weights, quantize, skip layers or approximate vocabulary scores.
Uses known lazy loading and tiling techniques; novelty is not established.
"""
import json


class DirectOriginalEngine:
    def __init__(self, store, budget_mib=1024, head_rows=2048):
        import mlx.core as mx
        from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock
        if not 128 <= head_rows <= 8192 or not 128 <= budget_mib <= 4096:
            raise ValueError('Head rows must be 128..8192 and MLX budget 128..4096 MiB')
        config = json.loads((store.model/'config.json').read_text())
        if config.get('model_type') != 'qwen3' or config.get('quantization') or config.get('quantization_config'):
            raise ValueError('Original dense Qwen3 only')
        self.args = ModelArgs.from_dict(config)
        self.store, self.mx, self.Block = store, mx, TransformerBlock
        self.budget, self.head_rows = budget_mib*1024**2, head_rows
        self.head_name = 'model.embed_tokens.weight' if self.args.tie_word_embeddings else 'lm_head.weight'
        for name in ['model.embed_tokens.weight', self.head_name]:
            if store.tensors[name]['shape'] != [self.args.vocab_size, self.args.hidden_size]:
                raise ValueError('Embedding/head shape differs from original architecture')
        largest = max(sum(v['bytes'] for name,v in store.tensors.items()
                        if name.startswith(f'model.layers.{i}.')) for i in range(self.args.num_hidden_layers))
        self.plan = {'budget_bytes': self.budget, 'largest_layer_bytes': largest,
            'head_tile_bytes': head_rows*self.args.hidden_size*2, 'head_rows': head_rows,
            'cached_layers': 0, 'duplicate_weight_storage_bytes': 0,
            'embedding_mode': 'Exact requested rows from original BF16 payload',
            'head_mode': 'All vocabulary rows in tiles, original BF16; no top-k pruning',
            'scope': 'Measured active MLX limit, not total RAM; transient peaks reported separately'}
        if largest+128*1024**2+2*self.plan['head_tile_bytes'] > self.budget:
            raise ValueError('Budget too small for one original layer plus reserve and head tiles')
        self.norm = store.load(['model.norm.weight'])['model.norm.weight']
        self.layer_calls = 0

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(self.args.num_hidden_layers)]

    def _check_budget(self):
        if self.mx.get_active_memory() > self.budget:
            raise MemoryError('Measured active MLX allocation exceeds budget')

    def __call__(self, inputs, cache=None):
        from mlx_lm.models.base import create_attention_mask
        mx = self.mx
        if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] > 128:
            raise ValueError('Prototype: one sequence, at most 128 input tokens')
        cache = cache if cache is not None else self.make_cache()
        if len(cache) != self.args.num_hidden_layers or cache[0].offset+inputs.shape[1] > 256:
            raise ValueError('Prototype total context limit is 256 tokens')
        ids = inputs[0].tolist()
        rows = [self.store.rows('model.embed_tokens.weight', token, token+1) for token in ids]
        h = mx.concatenate(rows, axis=0)[None, :, :]
        mx.eval(h)
        del rows
        mask = create_attention_mask(h, cache[0])
        for index in range(self.args.num_hidden_layers):
            weights = self.store.layer(index)
            block = self.Block(self.args)
            prefix = f'model.layers.{index}.'
            block.load_weights([(name.removeprefix(prefix), value) for name,value in weights.items()], strict=True)
            h = block(h, mask, cache[index])
            mx.eval(h, cache[index].state)
            self.layer_calls += 1
            self._check_budget()
            del block, weights
            mx.clear_cache()
        # RMSNorm is token-wise. Earlier prompt logits are not required for greedy
        # decoding; compare the last-position scores against the resident model.
        last = mx.fast.rms_norm(h[:, -1:, :], self.norm, self.args.rms_norm_eps)
        mx.eval(last)
        del h
        tiles = []
        for start in range(0, self.args.vocab_size, self.head_rows):
            matrix = self.store.rows(self.head_name, start, min(start+self.head_rows, self.args.vocab_size))
            logits = last @ matrix.T
            mx.eval(logits)
            tiles.append(logits)
            self._check_budget()
            del matrix, logits
            mx.clear_cache()
        result = mx.concatenate(tiles, axis=-1)
        mx.eval(result)
        self._check_budget()
        return result
