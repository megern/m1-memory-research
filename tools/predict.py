"""Run a bounded experimental rule specialist entirely from local files."""
import argparse
import json
import os
from pathlib import Path
from evaluate_specialist import parse_label
from make_specialist_data import row
from provenance import verify_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--task', choices=['auth', 'workflow'], required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--language', choices=['ar', 'en'], default='en')
    parser.add_argument('--normalize', action='store_true', help='Auth only: perform numeric comparisons in Python before LLM classification')
    args = parser.parse_args()
    if args.normalize and args.task != 'auth':
        parser.error('--normalize is available only for auth')
    verify_base(args.model)
    if not (args.adapter/'adapters.safetensors').is_file():
        parser.error('Existing local adapter required')
    record = json.loads(args.record.read_text())
    if not isinstance(record, dict):
        parser.error('Evidence must be a JSON object')
    if args.task == 'auth':
        if any(type(record.get(key)) is not int or record[key] < 0
               for key in ['failed_logins', 'window_seconds']) or type(record.get('subsequent_success')) is not bool:
            parser.error('Auth requires nonnegative integer failed_logins/window_seconds and boolean subsequent_success')
    else:
        if record.get('permission') not in ['write-all', 'read-all', 'contents: read'] or any(
                type(record.get(key)) is not bool for key in ['pull_request_target', 'checkout_untrusted_head']):
            parser.error('Workflow requires supported permission and boolean trigger/checkout flags')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(4*1024**3)
    mx.random.seed(17)
    template = row(args.task, 'test', 1 if args.language == 'ar' else 0)
    messages = template['messages'][:-1]
    prefix = messages[1]['content'].split('{', 1)[0]
    messages[1]['content'] = prefix+json.dumps(record, ensure_ascii=False)
    if args.normalize:
        from normalized_auth import messages as projected_messages
        messages = projected_messages(record, args.language)
    model, tokenizer = load(str(args.model.resolve()), adapter_path=str(args.adapter.resolve()),
                            tokenizer_config={'local_files_only': True, 'trust_remote_code': False}, trust_remote_code=False)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if len(prompt) > 384:
        parser.error('Record exceeds the 384-token protocol; shorten the evidence')
    text = ''.join(chunk.text for chunk in stream_generate(model, tokenizer, prompt=prompt,
                          max_tokens=48, sampler=make_sampler(temp=0)))
    label = parse_label(text)
    result = {'label': label, 'valid_schema': label is not None, 'raw_output': text}
    if args.normalize:
        from normalized_auth import features
        result['projected_features'] = features(record)
        result['numeric_comparisons'] = 'Python code, not LLM reasoning'
    print(json.dumps(result, ensure_ascii=False))
    if label is None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
