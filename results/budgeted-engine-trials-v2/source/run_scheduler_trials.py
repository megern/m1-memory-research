"""Run the frozen randomized schedule serially on this Mac, with safety guards."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import shutil
import statistics

from benchmark_original_scheduler import PROTOCOL
from run_original_precision import run


def main(model, output, protocol_path=None, worker_script=None,
         experiment_prefix='scheduler-', reference_suffix='scheduler-reference', extra_sources=()):
    if output.exists():
        raise ValueError('Use a fresh experiment directory')
    protocol_path = protocol_path or PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    refs_dir = model.parent/(model.name+'-'+reference_suffix)
    refs = json.loads((refs_dir/'reference.json').read_text())
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if refs['protocol_sha256'] != protocol_sha:
        raise ValueError('Reference and frozen protocol differ')
    paths = [Path(__file__), Path(__file__).with_name('benchmark_original_scheduler.py'),
             Path(__file__).with_name('layer_prefetch.py'),
             Path(__file__).with_name('run_original_precision.py'), *extra_sources]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rng, jobs = random.Random(protocol['order_seed']), []
    for round_index in range(protocol['rounds']):
        modes = protocol['modes'].copy()
        rng.shuffle(modes)
        jobs.extend({'round': round_index+1, 'mode': mode} for mode in modes)
    output.mkdir(parents=True)
    plan = {'protocol': protocol, 'protocol_sha256': protocol_sha, 'source_sha256': hashes,
        'reference_logits_sha256': refs['logits_sha256'], 'jobs': jobs,
        'environment': {p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm', 'numpy', 'psutil']},
        'status': 'running', 'completed': []}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    shutil.copyfile(refs_dir/'reference.json', output/'reference.json')
    rows = []
    for index, job in enumerate(jobs):
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name] for p in paths):
            raise ValueError('Source changed during frozen run; stop rather than mix implementations')
        name = f"round-{job['round']:02d}-{job['mode']}"
        with contextlib.redirect_stdout(io.StringIO()):
            report = run(model, output/name, experiment_prefix+job['mode'], worker_script=worker_script)
        event = report['events'][0] if report['events'] else None
        row = {**job, 'directory': name, 'exit_code': report['exit_code'],
            'guard': report['stopped_by_guard'], 'sampled_process_tree_rss_peak_bytes':
            report['sampled_process_tree_rss_peak_bytes']}
        if event:
            comparisons = [c for case in event['cases'] for c in case['comparisons']]
            row.update({k: event[k] for k in ['forward_seconds', 'prefill_seconds', 'decode_seconds',
                'greedy_steps', 'mlx_peak_allocated_bytes']})
            row.update(logits_allclose=all(c['allclose'] for c in comparisons),
                tokens_all_equal=all(c['greedy_token_equal'] for c in comparisons),
                maximum_absolute_logit_difference=max(c['maximum_absolute_difference'] for c in comparisons),
                comparison_count=len(comparisons),
                complete_layer_calls=len(event['layer_traces']))
            for key in ['engine_plan', 'engine_layer_calls', 'engine_file_loads', 'engine_cache_hits']:
                row[key] = event.get(key)
        rows.append(row)
        plan['completed'] = rows
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        print(json.dumps({'completed': index+1, 'total': len(jobs), **row}), flush=True)
    grouped = {}
    for mode in protocol['modes']:
        valid = [r for r in rows if r['mode'] == mode and r['exit_code'] == 0 and r['guard'] is None
                 and r.get('logits_allclose') and r.get('tokens_all_equal')
                 and r.get('comparison_count') == sum(len(c['tokens']) for c in refs['cases'])]
        metrics = {}
        for key in ['forward_seconds', 'prefill_seconds', 'decode_seconds',
                    'mlx_peak_allocated_bytes', 'sampled_process_tree_rss_peak_bytes']:
            values = [r[key] for r in valid]
            metrics[key] = {'median': statistics.median(values) if values else None,
                            'min': min(values) if values else None, 'max': max(values) if values else None}
        grouped[mode] = {'successful_equivalent_trials': len(valid), 'planned_trials': protocol['rounds'],
                         'metrics': metrics}
    summary = {'protocol_sha256': protocol_sha, 'source_sha256': hashes,
        'reference_logits_sha256': refs['logits_sha256'], 'trials': rows, 'modes': grouped,
        'limitations': protocol['claim_scope'], 'cache_protocol': protocol['cache_protocol'],
        'completed_jobs': len(rows), 'planned_jobs': len(jobs)}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plan['status'] = 'completed_with_failures' if any(r['exit_code'] != 0 or r['guard'] for r in rows) else 'completed'
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'status': plan['status'], 'modes': grouped}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.model.resolve(), a.output.resolve())
