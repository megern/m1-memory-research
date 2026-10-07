"""Same held-out prompts and greedy decoding for base and local adapters."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import time

def score(rows):
    labels = sorted({r['expected'] for r in rows})
    per_label = {}
    for label in labels:
        tp = sum(r['expected'] == label and r['prediction'] == label for r in rows)
        fp = sum(r['expected'] != label and r['prediction'] == label for r in rows)
        fn = sum(r['expected'] == label and r['prediction'] != label for r in rows)
        per_label[label] = {'tp': tp, 'fp': fp, 'fn': fn,
                            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    return {'cases': len(rows), 'correct': sum(r['prediction'] == r['expected'] for r in rows),
            'accuracy': sum(r['prediction'] == r['expected'] for r in rows)/len(rows),
            'valid_json': sum(r['valid_json'] for r in rows),
            'output_limit_cases': sum(r['finish_reason'] == 'length' for r in rows),
            'macro_f1': sum(r['f1'] for r in per_label.values())/len(per_label),
            'per_label': per_label}

def parse_label(text, allow_fence=False):
    candidate = text.strip()
    if allow_fence:
        fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', candidate, flags=re.DOTALL)
        if fence:
            candidate = fence.group(1)
    try:
        value = json.loads(candidate)
        if isinstance(value, dict) and set(value) == {'label'} and value['label'] in {'low','medium','high'}:
            return value['label']
    except (ValueError, TypeError):
        pass
    return None

def augment_scoring(report):
    for row in report['rows']:
        row['prediction_allowing_outer_fence'] = parse_label(row['raw_output'], allow_fence=True)
    semantic = [{**r, 'prediction':r['prediction_allowing_outer_fence'],
                 'valid_json':r['prediction_allowing_outer_fence'] is not None} for r in report['rows']]
    report['classification_allowing_outer_fence'] = score(semantic)
    report['classification_allowing_outer_fence_by_language'] = {
        lang:score([r for r in semantic if r['language']==lang]) for lang in ['ar','en']}
    report['scoring_note'] = ('Strict schema compliance and classification after removing only a complete outer Markdown JSON fence are reported separately. '
                              'Dual scoring was introduced after inspecting baseline pilot outputs; no new generation or hyperparameter selection used test scores. '
                              'No arbitrary prose extraction, label repair, or case conversion is permitted.')
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.model.is_dir() or not a.data.is_file():
        p.error('Local model directory and local test JSONL required')
    if a.output.exists() or a.output.resolve() == a.data.resolve():
        p.error('Choose a fresh report path')
    if a.adapter and not (a.adapter / 'adapters.safetensors').is_file():
        p.error('Existing local adapter required')
    from provenance import identify_base, verify_base
    base_manifest=identify_base(a.model)
    base_hashes=verify_base(a.model)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(4 * 1024**3)
    mx.random.seed(17)
    model, tokenizer = load(str(a.model.resolve()),
                            adapter_path=str(a.adapter.resolve()) if a.adapter else None,
                            tokenizer_config={'local_files_only': True, 'trust_remote_code': False},
                            trust_remote_code=False)
    cases = [json.loads(line) for line in a.data.read_text().splitlines() if line.strip()]
    rows = []
    for case in cases:
        messages = case['messages'][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(prompt) > 384:
            raise ValueError('Prompt exceeds protocol limit; refusing truncation')
        started = time.monotonic()
        text, last = '', None
        for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=48, sampler=make_sampler(temp=0)):
            text += chunk.text
            last = chunk
        prediction, valid = None, False
        try:
            value = json.loads(text.strip())
            valid = isinstance(value, dict) and set(value) == {'label'} and value['label'] in {'low', 'medium', 'high'}
            if valid:
                prediction = value['label']
        except (ValueError, TypeError):
            pass
        rows.append({'id': case['id'], 'language': case['language'], 'messages': messages,
                     'expected': case['label'], 'prediction': prediction, 'valid_json': valid,
                     'raw_output': text, 'elapsed_seconds': time.monotonic()-started,
                     'finish_reason': last.finish_reason if last else 'no_output',
                     'generated_tokens': last.generation_tokens if last else 0})
        if len(rows) % 12 == 0:
            print(f'Completed {len(rows)}/{len(cases)}', flush=True)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'base': base_manifest['repository'],
              'base_revision': base_manifest['revision'],
              'verified_base_hashes':base_hashes,
              'adapter_sha256': hashlib.sha256((a.adapter/'adapters.safetensors').read_bytes()).hexdigest() if a.adapter else None,
              'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
              'protocol': {'seed':17, 'temperature':0, 'max_output_tokens':48, 'enable_thinking':False,
                           'compute':'local Apple M1 Metal only', 'postprocessing':'strict JSON parse, no repair or label extraction'},
              'summary': score(rows),
              'by_language': {lang: score([r for r in rows if r['language']==lang]) for lang in ['ar','en']},
              'mlx_peak_bytes': mx.get_peak_memory(),
              'rss_high_water_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations': ['Original synthetic stated-rule tasks, not independent real-world cybersecurity competence.',
                              'Held-out surface prompts and account/job identities share the same underlying rules.',
                              'One training seed; no selection using test scores. Fixed iteration budget.',
                              'A base model plus different LoRA adapters, not independent foundation models.'],
              'rows': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    augment_scoring(report)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2), flush=True)

if __name__ == '__main__':
    main()
