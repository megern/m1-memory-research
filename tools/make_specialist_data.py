"""Original synthetic policy tasks; held-out surface templates and identities."""
import hashlib
import json
from pathlib import Path
import random

SYSTEM = 'Return only a JSON object with exactly one key, "label". Classify the supplied evidence using the stated rule. Do not execute actions. / أعد JSON فقط بمفتاح label واحد، وصنّف الدليل بحسب القاعدة.'

def row(task, split, index):
    rng = random.Random(f'{task}:{split}:{index}:17')
    language = 'ar' if index % 2 else 'en'
    if task == 'auth':
        failures = rng.randrange(0, 10)
        success = bool(rng.randrange(2))
        seconds = rng.choice([45, 150, 290, 310, 420])
        expected = ('high' if success else 'medium') if failures >= 5 and seconds <= 300 else 'low'
        evidence = {'failed_logins': failures, 'subsequent_success': success, 'window_seconds': seconds,
                    'account': f'{split}-user-{index}', 'source': f'192.0.2.{index % 200 + 1}'}
        rule_en = 'Rule: 5 or more failed logins within at most 300 seconds => medium; if followed by success => high. Otherwise low.'
        rule_ar = 'القاعدة: خمس محاولات دخول فاشلة أو أكثر خلال 300 ثانية أو أقل تعني medium، وإذا تبعها نجاح تعني high، وإلا low.'
    else:
        permission = rng.choice(['write-all', 'read-all', 'contents: read'])
        privileged = bool(rng.randrange(2))
        untrusted = bool(rng.randrange(2))
        expected = 'high' if permission == 'write-all' or (privileged and untrusted) else 'low'
        evidence = {'permission': permission, 'pull_request_target': privileged,
                    'checkout_untrusted_head': untrusted, 'job': f'{split}-job-{index}'}
        rule_en = 'Rule: write-all permissions OR pull_request_target combined with checkout_untrusted_head => high; otherwise low.'
        rule_ar = 'القاعدة: صلاحية write-all أو اجتماع pull_request_target مع checkout_untrusted_head تعني high، وإلا low.'
    rule = rule_ar if language == 'ar' else rule_en
    surface = {
        'train': 'Apply this rule to the record / طبّق القاعدة على السجل:\n',
        'valid': 'Choose a label for this evidence / اختر تصنيف هذا الدليل:\n',
        'test': 'Audit the following case and return its classification / راجع الحالة التالية وأعد تصنيفها:\n',
    }[split]
    prompt = rule + '\n' + surface + json.dumps(evidence, ensure_ascii=False)
    completion = json.dumps({'label': expected}, separators=(',', ':'))
    return {'id': f'{task}-{split}-{index:04}', 'language': language, 'label': expected,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': completion}]}

def build(root):
    root = Path(root)
    metadata = {'seed': 17, 'source': 'Original deterministic synthetic cases, no private logs.',
                'scope': 'Stated-rule classification; not general SOC competence.', 'datasets': {}}
    for task in ['auth', 'workflow']:
        directory = root / task
        directory.mkdir(parents=True, exist_ok=True)
        metadata['datasets'][task] = {}
        for split, count in [('train', 384), ('valid', 48), ('test', 48)]:
            rows = [row(task, split, i) for i in range(count)]
            path = directory / (split + '.jsonl')
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            metadata['datasets'][task][split] = {'rows': count,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'labels': {label: sum(r['label'] == label for r in rows) for label in ['low', 'medium', 'high']}}
    (root / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')

if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1] / 'data')
