"""Guarded direct-original BF16 inference or small-model numerical audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
from original_tensor_store import verify_files


class VerificationStop(RuntimeError):
    pass


def verify_guarded(model, manifest, timeout, nocache=False):
    """Observe hashing itself; keep one swap baseline for setup and inference."""
    started,baseline = time.monotonic(),psutil.swap_memory().used
    samples,last,low_since = [],None,None
    def observe(force=False):
        nonlocal last,low_since
        now=time.monotonic()
        if not force and last is not None and now-last<.5:
            return
        last=now
        available=psutil.virtual_memory().available
        growth=psutil.swap_memory().used-baseline
        rss=psutil.Process().memory_info().rss
        samples.append({'seconds':now-started,'rss_bytes':rss,
                        'system_available_bytes':available,'system_swap_growth_bytes':growth})
        low_since=(low_since or now) if available<512*1024**2 else None
        reason=('verification_timeout' if now-started>timeout else
                'verification_rss_over_6GiB' if rss>6*1024**3 else
                'verification_swap_growth_over_512MiB' if growth>512*1024**2 else
                'verification_available_under_512MiB_for_5s' if low_since and now-low_since>5 else None)
        if reason:
            raise VerificationStop(reason)
    identities,reason,error_type=None,None,None
    try:
        observe(force=True)
        identities=verify_files(model,manifest,nocache=nocache,observe=observe)
        observe(force=True)
    except VerificationStop as exc:
        identities,reason=None,str(exc)
    except (ValueError,OSError) as exc:
        identities,reason,error_type=None,'verification_failed',type(exc).__name__
    return identities,{'started_monotonic':started,'seconds':time.monotonic()-started,
                       'baseline_system_swap_bytes':baseline,'samples':samples,
                       'stopped_by_guard':reason,'error_type':error_type,
                       'io_policy':'per-descriptor F_NOCACHE' if nocache else 'cached'}


def compare_reference_logits(actual, expected, token, teacher_forced_token, step):
    """Score equivalence is separate from the token chosen for the next input."""
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Reference scores require equal shapes and finite values')
    reference_token = int(np.argmax(expected, axis=-1).item())
    return {'step':step,'greedy_token_equal':token == reference_token,
            'teacher_forced_token_id':teacher_forced_token,'reference_greedy_token_id':reference_token,
            'maximum_absolute_logit_difference':float(np.max(np.abs(actual-expected))),
            'allclose_atol_0p01_rtol_0p01':bool(np.allclose(actual,expected,atol=.01,rtol=.01))}


def worker(model, settings):
    import mlx.core as mx
    import numpy as np
    from transformers import AutoTokenizer
    from original_tensor_store import OriginalTensorStore
    from direct_original_engine import DirectOriginalEngine
    mx.set_memory_limit(4*1024**3)
    mx.random.seed(17)
    manifest = settings['manifest']
    store = OriginalTensorStore(model, manifest, verify=False,
                               verified_identities=settings['file_identities'],
                               io_mode=settings.get('io_mode','native'))
    engine = DirectOriginalEngine(store, settings['budget_mib'], settings['head_rows'],
                                  settings.get('prefill_chunk_size',128),settings.get('prefill_schedule','chunk-major'))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    references, arrays = None, None
    if settings.get('reference'):
        directory = Path(settings['reference'])
        references = json.loads((directory/'reference.json').read_text())
        protocol = Path(__file__).resolve().parents[1]/'models/budgeted-engine-protocol.json'
        if manifest['repository'] != 'Qwen/Qwen3-0.6B' or references['protocol_sha256'] != hashlib.sha256(protocol.read_bytes()).hexdigest():
            raise ValueError('Only the fixed original 0.6B audit reference is supported')
        with (directory/'logits.npz').open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != references['logits_sha256']:
                raise ValueError('Reference logits hash differs')
        if references.get('reference_kind') and references.get('prefill_chunk_size') != settings.get('prefill_chunk_size',128):
            raise ValueError('Chunked upstream reference requires matching prefill size')
        arrays = np.load(directory/'logits.npz', allow_pickle=False)
        cases = [case['prompt'] for case in references['cases']]
    else:
        cases = settings['prompts']
    mx.reset_peak_memory()
    results = []
    for case, text in enumerate(cases):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if len(prompt) > 128:
            raise ValueError('Prompt exceeds 128 tokens')
        cache, x, tokens, comparisons, times = engine.make_cache(), mx.array([prompt]), [], [], []
        count = len(references['cases'][case]['tokens']) if references else settings['tokens']
        stopped_eos = False
        for step in range(count):
            started = time.monotonic()
            logits = engine(x, cache=cache)[:, -1, :]
            mx.eval(logits)
            seconds = time.monotonic()-started
            token = int(mx.argmax(logits, axis=-1).item())
            tokens.append(token)
            times.append(seconds)
            next_token = token
            if references:
                actual = np.asarray(logits.astype(mx.float32)).copy()
                expected = arrays[f'case{case}_step{step}']
                next_token = references['cases'][case]['tokens'][step]
                comparisons.append(compare_reference_logits(actual,expected,token,next_token,step))
            print(json.dumps({'event':'step','case':case,'step':step,'token_id':token,
                'elapsed_seconds':seconds,'mlx_active_bytes':mx.get_active_memory(),
                'mlx_peak_bytes_so_far':mx.get_peak_memory()}),flush=True)
            if not references and token == tokenizer.eos_token_id:
                stopped_eos = True
                break
            x = mx.array([[next_token]])
        results.append({'case':case,'prompt':text,'prompt_tokens':len(prompt),'token_ids':tokens,
            'text':tokenizer.decode(tokens,skip_special_tokens=True),'stopped_at_eos':stopped_eos,
            'reached_token_limit':not references and not stopped_eos and len(tokens)==count,
            'step_seconds':times,'comparisons':comparisons})
        del cache,x,logits
        mx.clear_cache()
    print(json.dumps({'event':'direct_original_result','reference_kind':references.get('reference_kind','original unchunked resident') if references else None,'model_repository':manifest['repository'],
        'revision':manifest['revision'],'stored_tensor_elements':sum(v['bytes']//2 for v in store.tensors.values()),
        'tensor_count':len(store.tensors),'all_tensor_file_dtypes':['BF16'],'engine_plan':engine.plan,
        'layer_calls':engine.layer_calls,'layer_weight_loads':engine.layer_weight_loads,'prefill_chunks':engine.prefill_chunks,
        'head_passes':engine.head_passes,'row_bytes_read_by_pread':store.row_bytes_read,
        'selected_native_tensor_bytes_requested':store.selected_tensor_bytes_requested,
        'f_nocache_descriptors':store.f_nocache_descriptors,
        'mlx_peak_allocated_bytes':mx.get_peak_memory(),'cases':results,
        'limitations':['Row-byte counters are logical process reads, not physical SSD traffic.',
            'Native selected tensor bytes are requested payload sizes, not SSD-byte telemetry.',
            'All vocabulary rows computed; head tiling may introduce numerical differences tested separately.',
            'Active MLX checks are not total-RAM caps; transient MLX peaks separately reported.',
            'OS caches uncontrolled; no cold-cache claim.',
            'Short numerical/functional controls do not establish general model quality or novelty.']},
        ensure_ascii=False),flush=True)


def run(model, manifest_path, output, prompts, tokens=4, budget_mib=1024, head_rows=2048,
        reference=None, timeout=900, io_mode='native', verification_nocache=False, prefill_chunk_size=128, prefill_schedule="chunk-major"):
    if output.exists():
        raise ValueError('Fresh output directory required')
    if prefill_schedule not in {'chunk-major','layer-major'}:
        raise ValueError('Unknown prefill schedule')
    if type(prefill_chunk_size) is not int or not 1 <= prefill_chunk_size <= 128:
        raise ValueError('Prefill chunk size must be 1..128')
    manifest = json.loads(manifest_path.read_text())
    if io_mode not in {'native','raw-cached','raw-nocache'}:
        raise ValueError('Unknown I/O mode')
    output.mkdir(parents=True)
    identities,verification=verify_guarded(model,manifest,timeout,verification_nocache)
    verification_seconds=verification['seconds']
    if identities is None:
        report={'repository':manifest['repository'],'revision':manifest['revision'],
                'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'completed':False,'model_started':False,'exit_code':None,
                'stopped_by_guard':verification['stopped_by_guard'],
                'verification_error_type':verification['error_type'],
                'verification_seconds':verification_seconds,'worker_elapsed_seconds':0,
                'verification_io_policy':verification['io_policy'],
                'verification_measurements':verification['samples'],
                'verification_system_swap_delta_bytes':verification['samples'][-1]['system_swap_growth_bytes'] if verification['samples'] else None,
                'swap_baseline_scope':'before verification, retained during inference',
                'numerical_audit_passed':None,'events':[],'measurements':[],
                'sampled_process_tree_rss_peak_bytes':0,
                'limitations':['Verification did not complete; model was never launched.']}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'completed':False,'model_started':False,'guard':report['stopped_by_guard']}),flush=True)
        return report
    settings={'manifest':manifest,'file_identities':identities,'budget_mib':budget_mib,
        'head_rows':head_rows,'reference':str(reference) if reference else None,'prompts':prompts,'tokens':tokens,
        'io_mode':io_mode,'prefill_chunk_size':prefill_chunk_size,'prefill_schedule':prefill_schedule}
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
               MEGERN_DIRECT_ORIGINAL_SETTINGS=json.dumps(settings))
    started, swap = time.monotonic(),verification['baseline_system_swap_bytes']
    observations, guard, low_since = [],None,None
    with (output/'stdout.jsonl').open('w') as stdout,(output/'stderr.txt').open('w') as stderr:
        child=subprocess.Popen([sys.executable,__file__,'--model',str(model),'--worker'],
            stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
        root=psutil.Process(child.pid)
        while child.poll() is None:
            now=time.monotonic()
            try:
                rss=sum(p.memory_info().rss for p in [root,*root.children(recursive=True)])
            except psutil.NoSuchProcess:
                rss=0
            available=psutil.virtual_memory().available
            growth=psutil.swap_memory().used-swap
            observations.append({'seconds':now-started,'rss_bytes':rss,'system_available_bytes':available,
                                 'system_swap_growth_bytes':growth})
            low_since=(low_since or now) if available<512*1024**2 else None
            guard=('timeout' if now-verification['started_monotonic']>timeout else 'rss_over_6GiB' if rss>6*1024**3 else
                   'swap_growth_over_512MiB' if growth>512*1024**2 else
                   'available_under_512MiB_for_5s' if low_since and now-low_since>5 else None)
            if guard:
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(.5)
        child.wait()
    events,invalid=[],[]
    for line in (output/'stdout.jsonl').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            invalid.append(line)
    result=next((e for e in events if e['event']=='direct_original_result'),None)
    comparisons=[v for case in result['cases'] for v in case['comparisons']] if result else []
    numerical_audit_passed=(all(v['greedy_token_equal'] and v['allclose_atol_0p01_rtol_0p01']
                                for v in comparisons) if comparisons else None)
    raw=(output/'stderr.txt').read_text()
    cleaned=raw.replace(str(Path(__file__).resolve().parents[1]),'[research source]')
    cleaned=cleaned.replace(str(model.parent),'[local models]').replace(sys.base_prefix,'[python runtime]')
    (output/'stderr.txt').write_text(cleaned)
    weight_bytes=sum(v['bytes'] for k,v in manifest['files'].items() if k.endswith('.safetensors'))
    report={'repository':manifest['repository'],'revision':manifest['revision'],
        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'weight_file_bytes':weight_bytes,'physical_memory_bytes':psutil.virtual_memory().total,
        'weights_exceed_physical_ram':weight_bytes>psutil.virtual_memory().total,
        'verification_seconds':verification_seconds,'worker_elapsed_seconds':time.monotonic()-started,
        'io_mode':io_mode,'verification_system_swap_delta_bytes':verification['samples'][-1]['system_swap_growth_bytes'],
        'verification_io_policy':verification['io_policy'],'verification_measurements':verification['samples'],
        'swap_baseline_scope':'before verification, retained during inference','model_started':True,
        'exit_code':child.returncode,'stopped_by_guard':guard,'completed':child.returncode==0 and guard is None and result is not None,
        'numerical_audit_passed':numerical_audit_passed,
        'sampled_process_tree_rss_peak_bytes':max((r['rss_bytes'] for r in observations),default=0),
        'stderr_raw_sha256_before_path_redaction':hashlib.sha256(raw.encode()).hexdigest(),
        'events':events,'invalid_stdout_lines':invalid,'measurements':observations,
        'limitations':['All original files hashed before worker timing; verification policy recorded, OS caches uncontrolled.',
            'Baseline system swap may already exist; observed deltas include other apps.',
            'RSS and MLX allocations overlap and use different accounting; never sum them.',
            'No copied layer files, quantization, cloud compute or missing weights substituted.']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'completed':report['completed'],'exit_code':child.returncode,'guard':guard,
        'elapsed_seconds':report['worker_elapsed_seconds'],'rss_peak_bytes':report['sampled_process_tree_rss_peak_bytes'],
        'weights_exceed_physical_ram':report['weights_exceed_physical_ram'],
        'mlx_peak_bytes':result['mlx_peak_allocated_bytes'] if result else None},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--prompt',action='append')
    p.add_argument('--tokens',type=int,default=4)
    p.add_argument('--budget-mib',type=int,default=1024)
    p.add_argument('--head-rows',type=int,default=2048)
    p.add_argument('--reference',type=Path)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--io-mode',choices=['native','raw-cached','raw-nocache'],default='native')
    p.add_argument('--prefill-schedule',choices=['chunk-major','layer-major'],default='chunk-major')
    p.add_argument('--prefill-chunk-size',type=int,default=128)
    p.add_argument('--verification-nocache',action='store_true')
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:
        worker(a.model.resolve(),json.loads(os.environ['MEGERN_DIRECT_ORIGINAL_SETTINGS']))
    else:
        if not a.manifest or not a.output or not 1<=a.tokens<=16 or not 30<=a.timeout<=900:
            p.error('Manifest/output required, tokens 1..16, timeout 30..900')
        report=run(a.model.resolve(),a.manifest.resolve(),a.output.resolve(),
            a.prompt or ['Reply with exactly the single English word naming the capital of France.'],
            a.tokens,a.budget_mib,a.head_rows,a.reference.resolve() if a.reference else None,a.timeout,a.io_mode,a.verification_nocache,a.prefill_chunk_size,a.prefill_schedule)
        if not report['completed'] or report['numerical_audit_passed'] is False:
            raise SystemExit(1)
