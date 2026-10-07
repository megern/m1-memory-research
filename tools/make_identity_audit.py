"""Post-confirmatory counterfactual audit; change only noncausal identity fields."""
import argparse
import hashlib
import json
from pathlib import Path


def build(root, output):
    if output.exists():
        raise ValueError('Choose a fresh counterfactual audit directory')
    manifest = {'status': 'Exploratory audit designed after confirmatory evaluation and a failed auth demo. No retraining.',
                'reference_seed': 17,
                'intervention': 'Replace account/source or job identifiers with constant anonymous values; retain all rule evidence, labels and instruction text.',
                'not_independent_test': True, 'tasks': {}}
    output.mkdir(parents=True)
    for task in ['auth', 'workflow']:
        original = root / 'data/confirmatory-v2' / task / 'test.jsonl'
        rows = [json.loads(line) for line in original.read_text().splitlines()]
        for row in rows:
            text = row['messages'][1]['content']
            prefix, raw = text.split('{', 1)
            evidence = json.loads('{'+raw)
            if task == 'auth':
                evidence.update(account='anonymous-account', source='192.0.2.1')
            else:
                evidence['job'] = 'anonymous-job'
            row['messages'][1]['content'] = prefix+json.dumps(evidence, ensure_ascii=False)
        path = output / (task+'.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
        manifest['tasks'][task] = {'cases': len(rows), 'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                                    'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output/'protocol.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
