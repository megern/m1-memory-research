"""Aggregate every frozen confirmatory seed, without selecting a winning model."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root):
    protocol = json.loads((root / 'data/confirmatory-v2/protocol.json').read_text())
    tasks = {}
    for task in ['auth', 'workflow']:
        entries = []
        for seed in [None, *protocol['training_seeds']]:
            name = 'base' if seed is None else f'seed{seed}'
            report = json.loads((root / 'results' / f'{task}-v2-{name}.json').read_text())
            if report['data_sha256'] != protocol['datasets'][task]['test']['sha256']:
                raise ValueError('Test report does not match frozen split')
            entry = {'variant': name, 'training_seed': seed,
                     'strict': report['summary'],
                     'outer_fence_allowed': report['classification_allowing_outer_fence'],
                     'by_language': report['by_language'],
                     'outer_fence_allowed_by_language': report['classification_allowing_outer_fence_by_language'],
                     'evaluation_mlx_peak_bytes': report['mlx_peak_bytes'],
                     'evaluation_rss_high_water_bytes': report['rss_high_water_bytes']}
            if seed is not None:
                adapter = root / 'artifacts' / f'{task}-v2-seed{seed}'
                weight = adapter / 'adapters.safetensors'
                digest = hashlib.sha256(weight.read_bytes()).hexdigest()
                if digest != report['adapter_sha256']:
                    raise ValueError('Adapter and report hashes disagree')
                entry.update(adapter_bytes=weight.stat().st_size, adapter_sha256=digest,
                             training=json.loads((adapter / 'training-summary.json').read_text()))
            entries.append(entry)
        aggregates = {}
        for scoring in ['strict', 'outer_fence_allowed']:
            aggregates[scoring] = {}
            for metric in ['accuracy', 'macro_f1']:
                values = [e[scoring][metric] for e in entries if e['training_seed'] is not None]
                aggregates[scoring][metric] = {'mean': statistics.mean(values),
                                             'min': min(values), 'max': max(values)}
        tasks[task] = {'majority_accuracy': 1/len(protocol['datasets'][task]['test']['labels']),
                       'entries': entries, 'adapter_aggregate': aggregates}
    return {'protocol': protocol, 'tasks': tasks,
            'release_seed': 17,
            'release_note': 'Seed 17 is the reference replication, not selected using test scores.',
            'limits': ['Original synthetic stated-rule tasks; no operational security claim.',
                       'Three seeds are reported together; ranges are not confidence intervals.',
                       'Strict JSON and complete-outer-fence classification are distinct metrics.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh summary file')
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for task, data in result['tasks'].items():
        print(task, json.dumps(data['adapter_aggregate']))
