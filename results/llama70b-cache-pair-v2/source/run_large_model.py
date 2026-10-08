"""Guarded local llama.cpp smoke run with raw output and process-tree telemetry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time
import psutil

def run(runtime, model, manifest, output, timeout=900, tokens=8, verified_file=None, swap_baseline=None):
    if not 1 <= tokens <= 32 or not 30 <= timeout <= 1800:
        raise ValueError('Bounded protocol requires 1..32 tokens and 30..1800 seconds')
    runtime, model, output = Path(runtime).resolve(), Path(model).resolve(), Path(output).resolve()
    if not runtime.is_file() or not model.is_file():
        raise ValueError('Existing local runtime and model required')
    if output.exists():
        raise ValueError('Choose a fresh report directory')
    if model.stat().st_size != manifest['bytes']:
        raise ValueError('Unexpected model size')
    identity = lambda: [model.stat().st_size, model.stat().st_mtime_ns, model.stat().st_ino]
    if verified_file is None:
        verification_started = time.monotonic()
        original_identity = identity()
        with model.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        verification_seconds = time.monotonic() - verification_started
        if digest != manifest['sha256'] or identity() != original_identity:
            raise ValueError('Model checksum or identity changed')
    else:
        if verified_file['sha256'] != manifest['sha256'] or verified_file['identity'] != identity():
            raise ValueError('Previously verified model identity differs')
        original_identity = verified_file['identity']
        digest, verification_seconds = verified_file['sha256'], 0.0
    output.mkdir(parents=True)
    user = 'Reply with exactly the single English word naming the capital of France.'
    prompt = '<|start_header_id|>user<|end_header_id|>\n\n' + user + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    flags = ['--offline', '--device', 'none', '--n-gpu-layers', '0', '--fit', 'off',
             '--load-mode', 'mmap', '--no-repack', '--no-op-offload', '--no-kv-offload',
             '--ctx-size', '256', '--batch-size', '32', '--ubatch-size', '8',
             '--threads', '4', '--n-predict', str(tokens), '--temp', '0', '--seed', '17',
             '--no-warmup', '--no-display-prompt', '--no-conversation', '--perf']
    command = [str(runtime), '-m', str(model), *flags, '-p', prompt]
    env = {key:value for key,value in os.environ.items()
           if not key.startswith(('LLAMA_ARG_', 'GGML_RPC_'))}
    env.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    before = psutil.swap_memory().used if swap_baseline is None else swap_baseline
    started = time.monotonic()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    monitor = psutil.Process(proc.pid)
    selector = selectors.DefaultSelector()
    for name, pipe in [('stdout', proc.stdout), ('stderr', proc.stderr)]:
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, name)
    raw = {'stdout': bytearray(), 'stderr': bytearray()}
    observations, first_output, stop_reason = [], None, None
    max_rss, low_memory_since, last_sample = 0, None, 0
    sampled_pageins, sampled_faults = None, None
    def stop(reason):
        nonlocal stop_reason
        stop_reason = reason
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    while selector.get_map() or proc.poll() is None:
        now = time.monotonic()
        for key, _ in selector.select(timeout=0.2):
            data = os.read(key.fileobj.fileno(), 65536)
            if not data:
                selector.unregister(key.fileobj)
                continue
            raw[key.data].extend(data)
            if key.data == 'stdout' and data.strip() and first_output is None:
                first_output = time.monotonic() - started
        if now - last_sample >= 0.5 and proc.poll() is None:
            try:
                processes = [monitor, *monitor.children(recursive=True)]
                root_info = monitor.memory_info()
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
                if hasattr(root_info, 'pageins'):
                    sampled_pageins = max(sampled_pageins or 0, root_info.pageins)
                if hasattr(root_info, 'pfaults'):
                    sampled_faults = max(sampled_faults or 0, root_info.pfaults)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            available = psutil.virtual_memory().available
            swap_growth = psutil.swap_memory().used - before
            max_rss = max(max_rss, rss)
            observations.append({'elapsed_seconds': round(now-started,3), 'process_tree_rss_bytes':rss,
                                 'system_available_bytes':available, 'system_swap_growth_bytes':swap_growth,
                                 'root_process_pageins_sampled_max':sampled_pageins,
                                 'root_process_faults_sampled_max':sampled_faults})
            if available < 512 * 1024**2:
                low_memory_since = low_memory_since or now
            else:
                low_memory_since = None
            reason = ('timeout' if now-started > timeout else
                      'process_tree_rss_over_12GiB' if rss > 12*1024**3 else
                      'system_swap_grew_over_512MiB' if swap_growth > 512*1024**2 else
                      'available_memory_under_512MiB_for_5s' if low_memory_since and now-low_memory_since > 5 else None)
            if reason:
                stop(reason)
            last_sample = now
    proc.wait()
    stderr, stdout = (raw[k].decode('utf-8', errors='replace') for k in ['stderr','stdout'])
    timings = []
    for line in stderr.splitlines():
        match = re.search(r'(prompt eval time|(?<!prompt )eval time)\s*=\s*([\d.]+) ms\s*/\s*(\d+) (runs|tokens).*?([\d.]+) tokens per second', line)
        if match:
            kind, milliseconds, count, unit, tps = match.groups()
            timings.append({'kind':kind, 'milliseconds':float(milliseconds),
                            'count':int(count), 'count_unit':unit, 'reported_tokens_per_second':float(tps)})
    report = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'model':manifest, 'runtime_release':'llama.cpp b11457',
              'runtime_archive_sha256':'e234070cbde0c8b0d30f79fa08ff8246abe22c72acb752619349b8d9c46f7839',
              'model_file_sha256_verified':digest, 'physical_memory_bytes':psutil.virtual_memory().total,
              'preflight_full_file_hash_seconds':verification_seconds,
              'cache_state':('Uncontrolled OS page cache; full-file checksum immediately precedes measured runtime.' if verified_file is None else 'Uncontrolled OS page cache; one series checksum precedes first process only, no full-file rehash between processes.'),
              'preverified_identity_used':verified_file is not None,
              'model_identity_unchanged_after_run':identity()==original_identity,
              'configuration':flags, 'user_prompt':user,
              'prompt_protocol':'Single BOS supplied by the model tokenizer; no manually duplicated begin_of_text token.', 'exit_code':proc.returncode,
              'remote_backend_configuration':'No RPC endpoints configured; inherited LLAMA_ARG_* and GGML_RPC_* cleared.',
              'stopped_by_guard':stop_reason, 'elapsed_seconds':time.monotonic()-started,
              'time_to_first_nonwhitespace_stdout_seconds':first_output,
              'observed_process_tree_rss_peak_bytes':max_rss, 'output':stdout,
              'root_process_pageins_sampled_max':sampled_pageins,
              'root_process_faults_sampled_max':sampled_faults,
              'parsed_timing_lines':timings,
              'generated_output_present':bool(stdout.strip()),
              'smoke_runtime_completed':proc.returncode == 0 and stop_reason is None and bool(stdout.strip()) and bool(timings) and identity()==original_identity,
              'exact_paris_answer':stdout.strip().strip('.').casefold() == 'paris',
              'measurements':observations,
              'limitations':['One short smoke prompt is not a quality benchmark.',
                             'CPU-only mmap relies on OS paging; no custom layer-streaming algorithm or novelty claim.',
                             'Output-byte latency includes runtime startup and is not a library token callback.',
                             'RSS is sampled every half second; true instantaneous peak may be higher.',
                             'macOS root-process fault counters are sampled before exit, not exact final totals or SSD byte counts.',
                             'Preflight or series checksum reads all weights and may warm page cache; this is not a cold-cache benchmark.',
                             'System swap observations include other apps; guard is conservative.',
                             'Local inference, not training a 70B model. IQ2_XXS differs from full-precision weights.']}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'stdout.txt').write_text(stdout)
    # Redact only local paths, preserving underlying model messages and timings.
    stderr = stderr.replace(str(model), '[local model file]').replace(str(runtime), '[local runtime]')
    (output/'stderr.txt').write_text(stderr)
    print(json.dumps({k:report[k] for k in ['exit_code','stopped_by_guard','elapsed_seconds','observed_process_tree_rss_peak_bytes','output','exact_paris_answer']},indent=2),flush=True)
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--tokens',type=int,default=8)
    a=p.parse_args()
    run(a.runtime,a.model,json.loads(a.manifest.read_text()),a.output,a.timeout,a.tokens)
