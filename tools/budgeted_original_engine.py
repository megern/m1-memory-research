"""Experimental exact-BF16 Qwen3 engine: reusable blocks and bounded layer cache.

Known offloading/caching techniques; this prototype does not establish novelty.
All MLX work stays on the caller thread. Shared embedding and KV cache are resident.
"""
import hashlib
import json
from pathlib import Path

from run_original_precision import WEIGHT_SHA


def cache_plan(budget, shared_bytes, layer_bytes, layer_count, cap=None,
               reserve=64*1024**2):
    if min(budget, shared_bytes, layer_bytes, layer_count) <= 0 or reserve < 0:
        raise ValueError('Positive sizes and nonnegative reserve required')
    # Reserve two layer-sized allocations while replacing a reusable block's
    # weights. This estimate is not a total-RAM or activation-memory guarantee.
    remaining = budget-shared_bytes-2*layer_bytes-reserve
    if remaining < 0:
        raise ValueError('Budget too small for shared weights, replacement and reserve')
    count = min(layer_count, remaining//layer_bytes)
    if cap is not None:
        if cap < 0:
            raise ValueError('Cache cap must be nonnegative')
        count = min(count, cap)
    return {'budget_bytes': budget, 'shared_bytes': shared_bytes,
        'largest_layer_file_bytes': layer_bytes, 'reserved_bytes': reserve,
        'cached_layers': count, 'estimated_persistent_plus_replacement_bytes':
        shared_bytes+(count+2)*layer_bytes,
        'scope': 'MLX allocation heuristic, not total RAM; measured limit checked during forward'}


class BudgetedOriginalEngine:
    def __init__(self, model, budget_mib=640, cache_cap=None, verify=True):
        import mlx.core as mx
        from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock
        self.mx, self.Block = mx, TransformerBlock
        self.model_path = Path(model)
        self.directory = self.model_path.parent/(self.model_path.name+'-layers')
        manifest = json.loads((self.directory/'manifest.json').read_text())
        if manifest['source']['weight_sha256'] != WEIGHT_SHA:
            raise ValueError('Verified original checkpoint required')
        if verify:
            for shard in manifest['shards']:
                with (self.directory/shard['file']).open('rb') as f:
                    if hashlib.file_digest(f, 'sha256').hexdigest() != shard['sha256']:
                        raise ValueError('Shard checksum mismatch')
        config = json.loads((self.model_path/'config.json').read_text())
        if config.get('quantization') or config.get('quantization_config'):
            raise ValueError('Quantized models forbidden')
        self.args = ModelArgs.from_dict(config)
        if not self.args.tie_word_embeddings:
            raise ValueError('Pinned tied-embedding Qwen3 only')
        shared = mx.load(str(self.directory/'shared.safetensors'))
        self.embedding, self.norm = shared['model.embed_tokens.weight'], shared['model.norm.weight']
        del shared
        mx.eval(self.embedding, self.norm)
        if str(self.embedding.dtype) != 'mlx.core.bfloat16' or str(self.norm.dtype) != 'mlx.core.bfloat16':
            raise ValueError('Original BF16 required')
        largest = max(s['bytes'] for s in manifest['shards'] if s['file'].startswith('layer-'))
        self.plan = cache_plan(budget_mib*1024**2, self.embedding.nbytes+self.norm.nbytes,
                               largest, self.args.num_hidden_layers, cache_cap)
        self.cached = {}
        for index in range(self.plan['cached_layers']):
            block = self.Block(self.args)
            self._replace(block, index)
            self.cached[index] = block
        self.reusable = self.Block(self.args)
        self.layer_calls, self.file_loads, self.cache_hits = 0, 0, 0

    def _replace(self, block, index):
        weights = self.mx.load(str(self.directory/f'layer-{index:03d}.safetensors'))
        if {str(t.dtype) for t in weights.values()} != {'mlx.core.bfloat16'}:
            raise ValueError('Layer precision changed')
        self.mx.eval(weights)
        prefix = f'model.layers.{index}.'
        block.load_weights([(name.removeprefix(prefix), tensor)
                            for name, tensor in weights.items()], strict=True)

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(self.args.num_hidden_layers)]

    def __call__(self, inputs, cache=None):
        from mlx_lm.models.base import create_attention_mask
        mx = self.mx
        if inputs.shape[1] > 128:
            raise ValueError('Prototype prompt limit is 128 tokens')
        if cache is None:
            cache = self.make_cache()
        if len(cache) != self.args.num_hidden_layers or cache[0].offset+inputs.shape[1] > 256:
            raise ValueError('Prototype total context limit is 256 tokens')
        h = self.embedding[inputs]
        mask = create_attention_mask(h, cache[0])
        for index in range(self.args.num_hidden_layers):
            if index in self.cached:
                block = self.cached[index]
                self.cache_hits += 1
            else:
                block = self.reusable
                self._replace(block, index)
                self.file_loads += 1
            h = block(h, mask, cache[index])
            mx.eval(h, cache[index].state)
            self.layer_calls += 1
            if mx.get_active_memory() > self.plan['budget_bytes']:
                raise MemoryError('Measured MLX active allocation exceeded selected budget')
            mx.clear_cache()
        h = mx.fast.rms_norm(h, self.norm, self.args.rms_norm_eps)
        logits = h @ self.embedding.T
        mx.eval(logits)
        if mx.get_active_memory() > self.plan['budget_bytes']:
            raise MemoryError('Output allocation exceeded selected MLX budget')
        return logits
