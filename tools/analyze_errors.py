"""Post-evaluation descriptive audit of language composition and auth boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path


def evidence(row):
    text = row['messages'][1]['content']
    return json.loads(text[text.index('{'):])


def analyze(root):
    output = {'status': 'Exploratory analysis after confirmatory evaluation; no new training or checkpoint selection.',
              'tasks': {}}
    for task in ['auth', 'workflow']:
        directory = root / 'data/confirmatory-v2' / task
        splits = {s: [json.loads(line) for line in (directory/(s+'.jsonl')).read_text().splitlines()]
                  for s in ['train', 'valid', 'test']}
        counts = {s: {lang: dict(Counter(r['label'] for r in rows if r['language'] == lang))
                      for lang in ['ar', 'en']} for s, rows in splits.items()}
        # Learn this shortcut from TRAIN only. It deliberately ignores all evidence.
        shortcut = {lang: max(counts['train'][lang], key=lambda label: (counts['train'][lang][label], label))
                    for lang in ['ar', 'en']}
        results = {'language_label_counts': counts,
                   'train_language_majority_labels': shortcut,
                   'test_language_only_correct': sum(shortcut[r['language']] == r['label'] for r in splits['test']),
                   'test_cases': len(splits['test'])}
        if task == 'auth':
            results['by_window_seconds'] = {}
            for name in ['base', 'seed17', 'seed23', 'seed41']:
                report = json.loads((root/'results'/f'auth-v2-{name}.json').read_text())
                windows = sorted({evidence(r)['window_seconds'] for r in report['rows']})
                results['by_window_seconds'][name] = {
                    str(window): {'cases': len(rows := [r for r in report['rows'] if evidence(r)['window_seconds'] == window]),
                                  'correct': sum(r['prediction_allowing_outer_fence'] == r['expected'] for r in rows)}
                    for window in windows}
        output['tasks'][task] = results
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh analysis file')
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
