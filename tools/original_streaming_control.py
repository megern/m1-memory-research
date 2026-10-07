"""Two-step equivalence control for synchronous original-BF16 layer loading.

Reuses MLX Qwen3 blocks; layer loading is an established baseline, not novelty.
"""
import gc
import hashlib
import json
from pathlib import Path
import time


def worker(model):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_attention_mask
    from mlx.utils import tree_flatten

    mx.set_memory_limit(4*1024**3)
    mx.random.seed(17)
    directory = model.parent / (model.name+'-layers')
    manifest = json.loads((directory/'manifest.json').read_text())
    for item in manifest['shards']:
        with (directory/item['file']).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != item['sha256']:
                raise ValueError('Shard checksum mismatch')
    net, tokenizer = load(str(model), trust_remote_code=False,
        tokenizer_config={'local_files_only': True, 'trust_remote_code': False})
    if {str(t.dtype) for _, t in tree_flatten(net.parameters())} != {'mlx.core.bfloat16'}:
        raise ValueError('Resident model must preserve BF16')
    mx.eval(net.parameters())
    args = ModelArgs.from_dict(json.loads((model/'config.json').read_text()))
    if not args.tie_word_embeddings:
        raise ValueError('This control supports the pinned tied-embedding model only')
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content':
        'Reply with exactly the single English word naming the capital of France.'}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    caches = [KVCache() for _ in range(args.num_hidden_layers)]
    x, references, tokens = mx.array([prompt]), [], []
    begin = time.monotonic()
    for _ in range(2):
        logits = net(x, cache=caches)[:, -1, :]
        mx.eval(logits)
        references.append(np.asarray(logits.astype(mx.float32)).copy())
        token = int(mx.argmax(logits, axis=-1).item())
        tokens.append(token)
        x = mx.array([[token]])
    resident_seconds = time.monotonic()-begin
    resident_peak = mx.get_peak_memory()
    del net, caches, logits, x
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    shared = mx.load(str(directory/'shared.safetensors'))
    embedding = shared['model.embed_tokens.weight']
    norm = shared['model.norm.weight']
    # The upstream file stores a redundant head for tied embeddings; the model
    # sanitizes it as well. No calculation layer or used weight is omitted.
    del shared
    mx.eval(embedding, norm)
    caches = [KVCache() for _ in range(args.num_hidden_layers)]
    x, streamed, comparisons, layer_calls = mx.array([prompt]), [], [], 0
    begin = time.monotonic()
    for step in range(2):
        h = embedding[x]
        mask = create_attention_mask(h, caches[0])
        for index in range(args.num_hidden_layers):
            weights = mx.load(str(directory/f'layer-{index:03d}.safetensors'))
            if {str(t.dtype) for t in weights.values()} != {'mlx.core.bfloat16'}:
                raise ValueError('Streaming must preserve BF16')
            prefix = f'model.layers.{index}.'
            block = TransformerBlock(args)
            block.load_weights([(name.removeprefix(prefix), value)
                                for name, value in weights.items()], strict=True)
            h = block(h, mask, caches[index])
            mx.eval(h, caches[index].state)
            layer_calls += 1
            del block, weights
            mx.clear_cache()
        h = mx.fast.rms_norm(h, norm, args.rms_norm_eps)
        logits = (h @ embedding.T)[:, -1, :]
        mx.eval(logits)
        actual = np.asarray(logits.astype(mx.float32))
        token = int(mx.argmax(logits, axis=-1).item())
        streamed.append(token)
        comparisons.append({'step': step,
            'maximum_absolute_logit_difference': float(np.max(np.abs(actual-references[step]))),
            'allclose_atol_0p01_rtol_0p01': bool(np.allclose(actual, references[step], atol=.01, rtol=.01))})
        # Teacher-forced second input keeps logit comparisons on identical inputs,
        # even if a first greedy token differs. Token equality reported separately.
        x = mx.array([[tokens[step]]])
    print(json.dumps({'event': 'synchronous_streaming_control',
        'runtime_dtype': 'BF16', 'steps': 2, 'layers_executed': layer_calls,
        'resident_tokens': tokens, 'streaming_tokens': streamed,
        'token_ids_equal_on_identical_inputs': tokens == streamed,
        'resident_output': tokenizer.decode(tokens), 'streaming_output': tokenizer.decode(streamed),
        'logit_comparisons': comparisons,
        'resident_compute_seconds_excluding_model_load': resident_seconds,
        'streaming_seconds_including_layer_loads': time.monotonic()-begin,
        'resident_mlx_peak_allocated_bytes': resident_peak,
        'streaming_mlx_peak_allocated_bytes': mx.get_peak_memory(),
        'limitations': ['Single prompt, two steps; not a throughput or quality benchmark.',
            'OS caches warmed by hashes and resident run; no cold-cache result.',
            'Resident and streaming runs share a process; RSS peak cannot isolate them.',
            'Shared embedding and KV cache remain resident; only layers streamed.',
            'Uses established synchronous offloading; no new scheduling method yet.']}), flush=True)
