"""Bounded local control with original Qwen3-0.6B BF16 weights; no quantization.

This is a resident baseline, not a new offloading algorithm or quality benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import psutil

REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
WEIGHT_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
PROMPTS = [
    'Reply with exactly the single English word naming the capital of France.',
    'What is 17 multiplied by 19? Give only the number.',
    'اشرح في جملة واحدة ما هي المصادقة متعددة العوامل.',
]


def inspect(model):
    weight = model / 'model.safetensors'
    with weight.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
        f.seek(0)
        length = struct.unpack('<Q', f.read(8))[0]
        if length > 16 * 1024**2:
            raise ValueError('Unexpected safetensors header size')
        header = json.loads(f.read(length))
    if digest != WEIGHT_SHA:
        raise ValueError('Original upstream weight hash mismatch')
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    dtypes = sorted({v['dtype'] for v in tensors.values()})
    if dtypes != ['BF16']:
        raise ValueError(f'Expected original BF16 tensors, received {dtypes}')
    config = json.loads((model / 'config.json').read_text())
    if config.get('quantization') or config.get('quantization_config'):
        raise ValueError('Quantized configuration is forbidden')
    hashes = {}
    for name in ['config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
        with (model / name).open('rb') as f:
            hashes[name] = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'repository': 'Qwen/Qwen3-0.6B', 'revision': REVISION,
            'weight_sha256': digest, 'weight_bytes': weight.stat().st_size,
            'file_dtypes': dtypes, 'tensor_count': len(tensors),
            'support_file_hashes': hashes}


def worker(model):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx.utils import tree_flatten
    mx.set_memory_limit(4 * 1024**3)
    mx.random.seed(17)
    started = time.monotonic()
    net, tokenizer = load(str(model), trust_remote_code=False,
                          tokenizer_config={'local_files_only': True,
                                            'trust_remote_code': False})
    params = tree_flatten(net.parameters())
    if any('scales' in name or 'biases' in name for name, _ in params):
        raise ValueError('Unexpected quantization parameters')
    dtypes = sorted({str(t.dtype) for _, t in params})
    if dtypes != ['mlx.core.bfloat16']:
        raise ValueError(f'Runtime must preserve BF16, got {dtypes}')
    mx.eval(net.parameters())
    print(json.dumps({'event': 'loaded', 'seconds': time.monotonic()-started,
                      'runtime_parameter_dtypes': dtypes,
                      'mlx_parameter_bytes': sum(t.nbytes for _, t in params)}), flush=True)
    for index, text in enumerate(PROMPTS):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
        begin = time.monotonic()
        first, output, count = None, '', 0
        for chunk in stream_generate(net, tokenizer, prompt=prompt,
                                      max_tokens=32, sampler=make_sampler(temp=0)):
            if first is None:
                first = time.monotonic()-begin
            output += chunk.text
            count += 1
        print(json.dumps({'event': 'prompt', 'index': index, 'prompt': text,
            'prompt_tokens': len(prompt), 'raw_output': output,
            'generation_callback_count': count,
            'first_callback_seconds': first, 'elapsed_seconds': time.monotonic()-begin,
            'mlx_peak_allocated_bytes': mx.get_peak_memory()}, ensure_ascii=False), flush=True)


def run(model, output, experiment='resident'):
    if output.exists():
        raise ValueError('Use a fresh result directory')
    verified = inspect(model)
    output.mkdir(parents=True)
    started, swap = time.monotonic(), psutil.swap_memory().used
    observations, stop, low_since = [], None, None
    with (output/'stdout.jsonl').open('w') as stdout, (output/'stderr.txt').open('w') as stderr:
        proc = subprocess.Popen([sys.executable, __file__, '--model', str(model), '--worker',
                                 '--experiment', experiment],
            stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            now = time.monotonic()
            try:
                rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
            except psutil.NoSuchProcess:
                rss = 0
            available = psutil.virtual_memory().available
            growth = psutil.swap_memory().used-swap
            observations.append({'seconds': now-started, 'process_tree_rss_bytes': rss,
                'system_available_bytes': available, 'system_swap_growth_bytes': growth})
            low_since = (low_since or now) if available < 512*1024**2 else None
            stop = ('timeout_180s' if now-started > 180 else
                    'rss_over_6GiB' if rss > 6*1024**3 else
                    'swap_growth_over_512MiB' if growth > 512*1024**2 else
                    'low_available_5s' if low_since and now-low_since > 5 else None)
            if stop:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(0.2)
        proc.wait()
    events = [json.loads(line) for line in (output/'stdout.jsonl').read_text().splitlines()]
    report = {'model': verified, 'experiment': experiment,
        'exit_code': proc.returncode, 'stopped_by_guard': stop,
        'elapsed_seconds': time.monotonic()-started, 'events': events,
        'physical_memory_bytes': psutil.virtual_memory().total,
        'sampled_process_tree_rss_peak_bytes': max(x['process_tree_rss_bytes'] for x in observations),
        'observations': observations,
        'limitations': ['Small-model control, not 70B execution or a new scheduling algorithm.',
            'Resident mode: three prompts, 32 tokens each. Streaming mode: two-step equivalence control.',
            'No conversion, quantization or adapter; runtime BF16 explicitly checked.',
            'Full-file hash precedes execution; OS cache state uncontrolled.',
            'RSS and MLX allocation are different overlapping metrics; do not add them.',
            'Existing system swap and background apps remain; no cold-cache claim.']}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['exit_code', 'stopped_by_guard',
        'elapsed_seconds', 'sampled_process_tree_rss_peak_bytes', 'events']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--experiment', choices=['resident', 'streaming'], default='resident')
    args = parser.parse_args()
    if args.worker:
        if args.experiment == 'streaming':
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
            from original_streaming_control import worker as streaming_worker
            streaming_worker(args.model.resolve())
        else:
            worker(args.model.resolve())
    elif args.output:
        run(args.model.resolve(), args.output.resolve(), args.experiment)
    else:
        parser.error('--output is required')
