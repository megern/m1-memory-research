"""Reproduce a complete local series sequentially into a fresh output directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from provenance import identify_base, verify_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--series', choices=['confirmatory', 'size1p7', 'normalized'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error('Choose a fresh output directory; recorded results must be preserved')
    model = identify_base(args.model)
    expected = 'mlx-community/Qwen3-0.6B-4bit' if args.series != 'size1p7' else 'mlx-community/Qwen3-1.7B-4bit'
    if model['repository'] != expected:
        parser.error('Model does not match the specified research series')
    base_hashes = verify_base(args.model)
    data_root = root/'data'/('normalized-auth' if args.series == 'normalized' else 'confirmatory-v2')
    data_protocol = json.loads((data_root/'protocol.json').read_text())
    tasks = ['auth', 'workflow'] if args.series == 'confirmatory' else ['auth']
    for task in tasks:
        for split in ['train', 'valid', 'test']:
            path = data_root/task/(split+'.jsonl')
            if hashlib.sha256(path.read_bytes()).hexdigest() != data_protocol['datasets'][task][split]['sha256']:
                raise ValueError('Dataset differs from the recorded frozen split')
    output = args.output.resolve()
    (output/'logs').mkdir(parents=True)
    state = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'local_compute_only': True,
             'series': args.series, 'base': model['repository'], 'base_revision': model['revision'],
             'verified_base_hashes': base_hashes, 'completed': [], 'phase': 'starting'}
    def execute(name, arguments):
        state['phase'] = name
        (output/'status.json').write_text(json.dumps(state, indent=2)+'\n')
        print(name, flush=True)
        with (output/'logs'/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable, *map(str, arguments)], stdout=log, stderr=subprocess.STDOUT, check=True)
        state['completed'].append(name)
    try:
        for task in tasks:
            for seed in [17, 23, 41]:
                execute(f'train-{task}-seed{seed}', [root/'tools/train_specialist.py', '--model', args.model.resolve(),
                        '--data', data_root/task, '--output', output/'artifacts'/f'{task}-seed{seed}',
                        '--iters', '400', '--layers', '16', '--seed', str(seed)])
            execute(f'eval-{task}-base', [root/'tools/evaluate_specialist.py', '--model', args.model.resolve(),
                    '--data', data_root/task/'test.jsonl', '--output', output/'results'/f'{task}-base.json'])
            for seed in [17, 23, 41]:
                execute(f'eval-{task}-seed{seed}', [root/'tools/evaluate_specialist.py', '--model', args.model.resolve(),
                        '--data', data_root/task/'test.jsonl',
                        '--adapter', output/'artifacts'/f'{task}-seed{seed}',
                        '--output', output/'results'/f'{task}-seed{seed}.json'])
        state['phase'] = 'complete'
    except Exception as error:
        state.update(phase='failed', error=str(error))
        raise
    finally:
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        (output/'status.json').write_text(json.dumps(state, indent=2)+'\n')


if __name__ == '__main__':
    main()
