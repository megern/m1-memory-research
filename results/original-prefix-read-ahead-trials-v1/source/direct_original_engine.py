"""Original Qwen3 BF16 streaming with row lookups and tiled output projection.

Does not duplicate weights, quantize, skip layers or approximate vocabulary scores.
Uses known lazy loading and tiling techniques; novelty is not established.
"""
import json


class DirectOriginalEngine:
    def __init__(self, store, budget_mib=1024, head_rows=2048, prefill_chunk_size=128, prefill_schedule="chunk-major", max_input_tokens=128, max_context_tokens=256, prefetch_layers=0, prefetch_prefix_mib=0):
        import mlx.core as mx
        from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock
        if not 128 <= head_rows <= 8192 or not 128 <= budget_mib <= 4096:
            raise ValueError('Head rows must be 128..8192 and MLX budget 128..4096 MiB')
        if type(prefill_chunk_size) is not int or not 1 <= prefill_chunk_size <= 1024:
            raise ValueError('Prefill chunk size must be 1..1024')
        if prefill_schedule not in {'chunk-major','layer-major'}:
            raise ValueError('Unknown prefill schedule')
        if not 1 <= max_input_tokens <= 1024 or not max_input_tokens <= max_context_tokens <= 2048:
            raise ValueError('Input/context limits must be 1..1024 and input..2048')
        if type(prefetch_layers) is not int or prefetch_layers not in {0,1}:
            raise ValueError('Read-ahead depth must be 0 or 1')
        if prefetch_layers and store.io_mode == 'native':
            raise ValueError('Byte read-ahead requires raw-cached or raw-nocache')
        if type(prefetch_prefix_mib) is not int or prefetch_prefix_mib not in {0,8,16,32,64,128}:
            raise ValueError('Prefix MiB must be 0,8,16,32,64,128')
        if prefetch_prefix_mib and store.io_mode == 'native':
            raise ValueError('Prefix streaming requires raw I/O')
        self.prefetch_prefix_bytes = prefetch_prefix_mib*1024**2
        self.prefetch_layers = prefetch_layers
        self.max_input_tokens, self.max_context_tokens = max_input_tokens, max_context_tokens
        self.prefill_schedule = prefill_schedule
        self.layer_weight_loads = 0
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_chunks = self.head_passes = 0
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
            'io_mode': store.io_mode, 'prefill_chunk_size': prefill_chunk_size, 'prefill_schedule': prefill_schedule,
            'head_tile_bytes': head_rows*self.args.hidden_size*2, 'head_rows': head_rows,
            'max_input_tokens':max_input_tokens,'max_context_tokens':max_context_tokens,
            'kv_cache_kind':'Original BF16 KVCache, no quantization',
            'prefetch_layers':prefetch_layers, 'prefetch_prefix_mib':prefetch_prefix_mib,
            'prefetch_host_payload_limit_bytes':min(largest,self.prefetch_prefix_bytes or largest)*prefetch_layers,
            'prefetch_policy':'One raw-byte reader; all MLX operations on main thread; prefix mode materializes one tensor at a time' ,
            'cached_layers': 0, 'duplicate_weight_storage_bytes': 0,
            'embedding_mode': 'Exact requested rows from original BF16 payload',
            'head_mode': 'All vocabulary rows in tiles, original BF16; no top-k pruning',
            'scope': 'Measured active MLX limit, not total RAM; transient peaks reported separately'}
        if largest+min(largest,self.prefetch_prefix_bytes or largest)*prefetch_layers+128*1024**2+2*self.plan['head_tile_bytes'] > self.budget:
            raise ValueError('Budget too small for current/prefetched layers plus reserve and head tiles')
        self.norm = store.load(['model.norm.weight'])['model.norm.weight']
        self.layer_calls = 0

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(self.args.num_hidden_layers)]

    def _check_budget(self):
        if self.mx.get_active_memory() > self.budget:
            raise MemoryError('Measured active MLX allocation exceeds budget')

    def __call__(self, inputs, cache=None):
        if inputs.ndim != 2 or inputs.shape[0] != 1 or not 1 <= inputs.shape[1] <= self.max_input_tokens:
            raise ValueError(f'Prototype: one sequence, at most {self.max_input_tokens} input tokens')
        cache = cache if cache is not None else self.make_cache()
        if len(cache) != self.args.num_hidden_layers or cache[0].offset+inputs.shape[1] > self.max_context_tokens:
            raise ValueError(f'Prototype total context limit is {self.max_context_tokens} tokens')
        offsets = [item.offset for item in cache]
        if len(set(offsets)) != 1:
            raise ValueError('All layer cache offsets must agree before a forward pass')
        result = None
        length = inputs.shape[1]
        if self.prefill_schedule == 'layer-major':
            result = self._forward_chunk(inputs, cache, project=True, layer_chunk_size=self.prefill_chunk_size)
            if length > 1:
                self.prefill_chunks += (length+self.prefill_chunk_size-1)//self.prefill_chunk_size
            return result
        for start in range(0, length, self.prefill_chunk_size):
            end = min(start+self.prefill_chunk_size, length)
            result = self._forward_chunk(inputs[:, start:end], cache, project=end == length)
            if length > 1:
                self.prefill_chunks += 1
        return result

    def _forward_chunk(self, inputs, cache, project, layer_chunk_size=None):
        from mlx_lm.models.base import create_attention_mask
        mx = self.mx
        ids = inputs[0].tolist()
        rows = [self.store.rows('model.embed_tokens.weight', token, token+1) for token in ids]
        h = mx.concatenate(rows, axis=0)[None, :, :]
        mx.eval(h)
        del rows
        from contextlib import nullcontext
        from original_tensor_store import LayerReadAhead
        manager = LayerReadAhead(self.store, self.args.num_hidden_layers,self.prefetch_prefix_bytes) if self.prefetch_layers else nullcontext(None)
        with manager as reader:
            for index in range(self.args.num_hidden_layers):
                weights = reader.take(index) if reader else (self.store.materialize_prefix_layer(index,self.store.layer_prefix(index,self.prefetch_prefix_bytes)) if self.prefetch_prefix_bytes else self.store.layer(index))
                self.layer_weight_loads += 1
                block = self.Block(self.args)
                prefix = f'model.layers.{index}.'
                block.load_weights([(name.removeprefix(prefix), value) for name,value in weights.items()], strict=True)
                if layer_chunk_size is not None and h.shape[1] > layer_chunk_size:
                    parts = []
                    for start in range(0,h.shape[1],layer_chunk_size):
                        piece = h[:, start:start+layer_chunk_size, :]
                        mask = create_attention_mask(piece,cache[index])
                        piece = block(piece,mask,cache[index])
                        mx.eval(piece,cache[index].state)
                        parts.append(piece)
                        self.layer_calls += 1
                        self._check_budget()
                    h = mx.concatenate(parts,axis=1)
                    mx.eval(h)
                    del parts,piece,mask
                else:
                    mask = create_attention_mask(h,cache[index])
                    h = block(h,mask,cache[index])
                    mx.eval(h,cache[index].state)
                    self.layer_calls += 1
                    self._check_budget()
                del block, weights
                mx.clear_cache()
        if not project:
            del h
            mx.clear_cache()
            return None
        self.head_passes += 1
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
