"""Frozen, isolated BF16 scheduler trials against resident and file baselines."""
import gc
import hashlib
import io
import json
from pathlib import Path
import time

from layer_prefetch import LayerReader

PROTOCOL = Path(__file__).resolve().parents[1]/'models/original-scheduler-protocol.json'


def worker(model, mode):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_attention_mask
    from mlx.utils import tree_flatten
    from transformers import AutoTokenizer

    mx.set_memory_limit(4*1024**3)
    mx.random.seed(17)
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    refs_dir = model.parent/(model.name+'-scheduler-reference')
    layers_dir = model.parent/(model.name+'-layers')
    manifest = json.loads((layers_dir/'manifest.json').read_text())
    for shard in manifest['shards']:
        with (layers_dir/shard['file']).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != shard['sha256']:
                raise ValueError('Shard checksum mismatch')
    args = ModelArgs.from_dict(json.loads((model/'config.json').read_text()))
    if not args.tie_word_embeddings:
        raise ValueError('Pinned tied-embedding model required')
    if mode in ('reference', 'resident'):
        net, tokenizer = load(str(model), trust_remote_code=False,
            tokenizer_config={'local_files_only': True, 'trust_remote_code': False})
        if {str(t.dtype) for _, t in tree_flatten(net.parameters())} != {'mlx.core.bfloat16'}:
            raise ValueError('Original BF16 required')
        mx.eval(net.parameters())
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        shared = mx.load(str(layers_dir/'shared.safetensors'))
        embedding, norm = shared['model.embed_tokens.weight'], shared['model.norm.weight']
        del shared
        mx.eval(embedding, norm)
    if mode == 'reference':
        if refs_dir.exists():
            raise ValueError('Frozen reference already exists; refusing overwrite')
        refs_dir.mkdir()
        references, ref_arrays = [], {}
    else:
        refs = json.loads((refs_dir/'reference.json').read_text())
        if refs['protocol_sha256'] != protocol_sha:
            raise ValueError('Reference protocol differs')
        with (refs_dir/'logits.npz').open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != refs['logits_sha256']:
                raise ValueError('Reference logits checksum differs')
        reference_arrays = np.load(refs_dir/'logits.npz', allow_pickle=False)
    ahead = {'prefetch1': 1, 'prefetch2': 2}.get(mode, 0)
    traces, results = [], []
    mx.reset_peak_memory()
    measured_start = time.monotonic()
    reader = LayerReader(layers_dir, args.num_hidden_layers, ahead)
    try:
        for case, text in enumerate(protocol['prompts']):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=False)
            caches = [KVCache() for _ in range(args.num_hidden_layers)]
            x, tokens, differences, token_seconds = mx.array([ids]), [], [], []
            maximum = protocol['maximum_generation_steps'] if mode == 'reference' else len(refs['cases'][case]['tokens'])
            for step in range(maximum):
                begin = time.monotonic()
                if mode in ('reference', 'resident'):
                    logits = net(x, cache=caches)[:, -1, :]
                    mx.eval(logits)
                else:
                    h = embedding[x]
                    mask = create_attention_mask(h, caches[0])
                    if mode != 'file':
                        reader.begin()
                    for index in range(args.num_hidden_layers):
                        started = time.monotonic()
                        if mode == 'file':
                            weights = mx.load(str(layers_dir/f'layer-{index:03d}.safetensors'))
                            mx.eval(weights)
                            read_seconds, wait_seconds = None, None
                            byte_count = next(s['bytes'] for s in manifest['shards']
                                if s['file'] == f'layer-{index:03d}.safetensors')
                        else:
                            data, read_seconds, wait_seconds = reader.take(index)
                            byte_count = len(data)
                            weights = mx.load(io.BytesIO(data), format='safetensors')
                            mx.eval(weights)
                            del data
                        ready_seconds = time.monotonic()-started
                        if {str(t.dtype) for t in weights.values()} != {'mlx.core.bfloat16'}:
                            raise ValueError('Streaming changed weight precision')
                        started = time.monotonic()
                        block = TransformerBlock(args)
                        prefix = f'model.layers.{index}.'
                        block.load_weights([(name.removeprefix(prefix), tensor)
                                            for name, tensor in weights.items()], strict=True)
                        h = block(h, mask, caches[index])
                        mx.eval(h, caches[index].state)
                        compute_seconds = time.monotonic()-started
                        del block, weights
                        mx.clear_cache()
                        traces.append({'case': case, 'step': step, 'layer': index,
                            'serialized_layer_bytes': byte_count,
                            'file_read_seconds': read_seconds, 'consumer_wait_seconds': wait_seconds,
                            'consumer_weight_ready_seconds': ready_seconds,
                            'block_construct_and_compute_seconds': compute_seconds})
                    h = mx.fast.rms_norm(h, norm, args.rms_norm_eps)
                    logits = (h @ embedding.T)[:, -1, :]
                    mx.eval(logits)
                token_seconds.append(time.monotonic()-begin)
                actual = np.asarray(logits.astype(mx.float32)).copy()
                token = int(mx.argmax(logits, axis=-1).item())
                tokens.append(token)
                if mode == 'reference':
                    ref_arrays[f'case{case}_step{step}'] = actual
                    next_token = token
                else:
                    reference = reference_arrays[f'case{case}_step{step}']
                    next_token = refs['cases'][case]['tokens'][step]
                    differences.append({'step': step, 'greedy_token_equal': token == next_token,
                        'maximum_absolute_difference': float(np.max(np.abs(actual-reference))),
                        'allclose': bool(np.allclose(actual, reference,
                            atol=protocol['logit_atol'], rtol=protocol['logit_rtol']))})
                x = mx.array([[next_token]])
                if mode == 'reference' and next_token == tokenizer.eos_token_id:
                    break
            result = {'case': case, 'prompt_tokens': len(ids), 'tokens': tokens,
                'raw_output': tokenizer.decode(tokens), 'token_seconds': token_seconds,
                'prefill_seconds': token_seconds[0], 'decode_seconds': sum(token_seconds[1:]),
                'comparisons': differences}
            results.append(result)
            if mode == 'reference':
                references.append({'case': case, 'prompt': text, 'tokens': tokens})
            del caches, x, logits
            if mode not in ('reference', 'resident'):
                del h
            gc.collect()
            mx.clear_cache()
    finally:
        reader.close()
    measured_seconds = time.monotonic()-measured_start
    if mode == 'reference':
        np.savez(refs_dir/'logits.npz', **ref_arrays)
        with (refs_dir/'logits.npz').open('rb') as f:
            ref_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        (refs_dir/'reference.json').write_text(json.dumps({'protocol_sha256': protocol_sha,
            'logits_sha256': ref_sha, 'cases': references}, ensure_ascii=False, indent=2)+'\n')
    else:
        ref_sha = refs['logits_sha256']
    print(json.dumps({'event': 'scheduler_trial', 'mode': mode,
        'protocol_sha256': protocol_sha, 'reference_logits_sha256': ref_sha,
        'runtime_dtype': 'BF16', 'measured_seconds_including_comparison_and_cleanup': measured_seconds,
        'forward_seconds': sum(sum(r['token_seconds']) for r in results),
        'prefill_seconds': sum(r['prefill_seconds'] for r in results),
        'decode_seconds': sum(r['decode_seconds'] for r in results),
        'greedy_steps': sum(len(r['tokens']) for r in results),
        'mlx_peak_allocated_bytes': mx.get_peak_memory(),
        'maximum_serialized_layer_bytes': max(s['bytes'] for s in manifest['shards'] if s['file'].startswith('layer-')),
        'buffer_budget_current_plus_pending_files': ahead+2 if ahead else 1,
        'cases': results, 'layer_traces': traces,
        'limitations': ['Four short prompts; at most eight greedy steps, not quality or steady throughput.',
            'Teacher-forced resident reference inputs; token equality checked separately.',
            'File hashes warm OS caches; no cold-SSD benchmark or SSD byte measurement.',
            'Buffered bytes reside outside MLX allocator; report RSS separately, do not add metrics.',
            'CPU thread reads bytes only; all MLX operations stay on the main thread.',
            'New process per trial; resident loading and shared-weight loading excluded from forward time.',
            'Prefetch is an established technique; no novelty claim.']}, ensure_ascii=False), flush=True)
