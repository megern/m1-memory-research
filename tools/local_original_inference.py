"""Experimental local BF16 inference under an MLX allocation budget.

Requires the verified original Qwen3-0.6B and unchanged layer shards locally.
No automatic downloads or network inference. Budget excludes total OS memory.
"""
import argparse
import json
import os
from pathlib import Path
import time


def worker(model):
    settings = json.loads(os.environ['MEGERN_ORIGINAL_INFERENCE_SETTINGS'])
    import mlx.core as mx
    from transformers import AutoTokenizer
    from budgeted_original_engine import BudgetedOriginalEngine
    mx.set_memory_limit(4*1024**3)
    mx.random.seed(17)
    started = time.monotonic()
    engine = BudgetedOriginalEngine(model, budget_mib=settings['budget_mib'])
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': settings['prompt']}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if len(prompt) > 128:
        raise ValueError('Shorten the prompt to at most 128 template tokens')
    setup_seconds = time.monotonic()-started
    mx.reset_peak_memory()
    cache, x, output, first, eos = engine.make_cache(), mx.array([prompt]), [], None, False
    started = time.monotonic()
    for _ in range(settings['tokens']):
        logits = engine(x, cache=cache)[:, -1, :]
        token = int(mx.argmax(logits, axis=-1).item())
        if first is None:
            first = time.monotonic()-started
        output.append(token)
        if token == tokenizer.eos_token_id:
            eos = True
            break
        x = mx.array([[token]])
    print(json.dumps({'event': 'bounded_original_inference', 'runtime_dtype': 'BF16',
        'engine_plan': engine.plan, 'prompt': settings['prompt'], 'prompt_tokens': len(prompt),
        'raw_generated_token_ids': output, 'text': tokenizer.decode(output, skip_special_tokens=True),
        'stopped_at_eos': eos, 'reached_token_limit': not eos and len(output) == settings['tokens'],
        'setup_seconds': setup_seconds, 'generation_seconds': time.monotonic()-started,
        'first_greedy_step_seconds': first, 'mlx_peak_allocated_bytes': mx.get_peak_memory(),
        'engine_layer_calls': engine.layer_calls, 'engine_cache_hits': engine.cache_hits,
        'engine_file_loads': engine.file_loads,
        'limitations': ['Prototype supports the pinned Qwen3-0.6B original BF16 checkpoint only.',
            'Budget governs measured MLX allocations, not total RAM or disk cache.',
            'Cached layers and block reuse are known methods; no proven algorithmic novelty.',
            'Generation quality depends on the base model; no production security guarantee.']},
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--prompt')
    p.add_argument('--budget-mib', type=int, default=640)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--experiment', default='local-original')
    a = p.parse_args()
    if not 384 <= a.budget_mib <= 2048 or not 1 <= a.tokens <= 64:
        p.error('Budget must be 384..2048 MiB and token limit 1..64')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    if a.worker:
        worker(a.model.resolve())
    elif a.output and a.prompt and len(a.prompt) <= 4096:
        os.environ['MEGERN_ORIGINAL_INFERENCE_SETTINGS'] = json.dumps({
            'prompt': a.prompt, 'budget_mib': a.budget_mib, 'tokens': a.tokens})
        from run_original_precision import run
        report = run(a.model.resolve(), a.output.resolve(), 'local-original', worker_script=Path(__file__).resolve())
        if report['exit_code'] != 0 or report['stopped_by_guard']:
            raise SystemExit(1)
    else:
        p.error('Provide --output and a nonempty --prompt of at most 4096 characters')
