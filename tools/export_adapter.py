"""Export a verified reference adapter and portable provenance, never base weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def export(root, task, destination, series='v2'):
    if destination.exists():
        raise ValueError('Choose a fresh export directory')
    if series != 'v2' and task != 'auth':
        raise ValueError('Only the original v2 series includes workflow')
    seed = 17
    source = root / 'artifacts' / f'{task}-{series}-seed{seed}'
    report = json.loads((root / 'results' / f'{task}-{series}-seed{seed}.json').read_text())
    protocol = json.loads((root / 'data' / ('normalized-auth' if series == 'normalized' else 'confirmatory-v2') / 'protocol.json').read_text())
    weights = source / 'adapters.safetensors'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != report['adapter_sha256']:
        raise ValueError('Export weights do not match evaluated adapter')
    if report['data_sha256'] != protocol['datasets'][task]['test']['sha256']:
        raise ValueError('Report does not match frozen test data')
    raw_config = json.loads((source / 'adapter_config.json').read_text())
    config = {key: raw_config[key] for key in ['fine_tune_type', 'num_layers', 'lora_parameters']}
    destination.mkdir(parents=True)
    shutil.copyfile(weights, destination / 'adapters.safetensors')
    shutil.copyfile(root / 'models/ADAPTER-LICENSE-APACHE-2.0', destination / 'LICENSE')
    (destination / 'adapter_config.json').write_text(json.dumps(config, indent=2)+'\n')
    training = json.loads((source / 'training-summary.json').read_text())
    provenance = {'base': report['base'], 'base_revision': report['base_revision'],
                  'verified_base_hashes': report['verified_base_hashes'],
                  'adapter_sha256': digest, 'adapter_bytes': weights.stat().st_size,
                  'training': training, 'test_sha256': report['data_sha256'],
                  'strict': report['summary'],
                  'outer_fence_allowed': report['classification_allowing_outer_fence'],
                  'selection': 'Reference seed 17; not selected using test accuracy.'}
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    score = report['summary']
    baseline=json.loads((root/'results'/f'{task}-{series}-base.json').read_text())
    baseline_accuracy=baseline['classification_allowing_outer_fence']['accuracy']
    card = f'''---
license: apache-2.0
base_model: {report['base']}
library_name: mlx
language:
- ar
- en
tags:
- lora
- mlx
- synthetic-data
---

# M1 {task} rule specialist — reference seed 17

Author: Megern Qaisse ([megern](https://github.com/megern)). Source, reproducible protocols and all seeds: [M1 Memory Research](https://github.com/megern/m1-memory-research).

An original local LoRA adapter over `{report['base']}`, trained on an Apple M1 with 16 GiB unified memory. This contains adapter weights only. Base revision: `{report['base_revision']}`.

The task is {'authentication-event priority classification from explicit counts, time windows and a subsequent-success flag' if task == 'auth' else 'workflow-policy priority classification from explicit permissions, trigger and untrusted-checkout flags'}. Prompts state the rule. No real incident logs, hosted training or teacher API was used. These synthetic tasks can also be solved directly by rule-based software.

## Recorded evaluation

Final seed-17 checkpoint on the frozen 96-case synthetic test: strict accuracy {score['correct']}/{score['cases']} ({score['accuracy']:.2%}), macro-F1 {score['macro_f1']:.4f}, valid JSON {score['valid_json']}/{score['cases']}. All three training seeds and a common base baseline must be read together in the source study; this adapter was not selected for its test score.

The same task's unadapted base classification accuracy is {baseline_accuracy:.2%} after removing only a complete outer Markdown JSON fence. Base strict schema accuracy is reported separately. This keeps formatting failures distinct from wrong labels. Authentication data has a language–label association and retained threshold failures; the study's exploratory composition audit must accompany per-language interpretations.

## Local use

Install the pinned study dependencies and acquire the pinned base checkpoint locally. Then use the study evaluator with `--model /absolute/local/base --adapter /absolute/local/this-adapter --data /absolute/local/test.jsonl --output /absolute/local/fresh-report.json`.

Generation uses the base chat template with `enable_thinking=False`, greedy decoding and a 48-token output cap. The adapter expects the study's system instruction, a stated classification rule, and JSON evidence. It is not a general conversational model.

## Training and limitations

384 training and 48 validation examples, 400 iterations, batch one, last 16 layers, query/value LoRA rank eight, learning rate 0.0002, prompt masking and gradient checkpointing. No test-based checkpoint selection. See `provenance.json` for hashes, measured allocation and timing. MLX allocation is different from total process RAM.

This is an experimental synthetic classifier. Scores do not demonstrate general Arabic proficiency, production SOC accuracy, adversarial robustness or security guarantees. No peer review or novel training algorithm is claimed. Base weights remain under their Apache-2.0 terms; this adapter is released under Apache-2.0. The original study tools and synthetic data use MIT.
'''
    if series == 'normalized':
        card = card.replace('from explicit counts, time windows and a subsequent-success flag', 'from three Boolean policy features computed by Python')
        start = card.index('Authentication data has a language')
        end = card.index('\n\n## Local use', start)
        card = card[:start] + 'This exploratory prototype uses different sampling, language balance, prompts and feature representation. Python performs the count and time comparisons; the LLM does not. There are only eight possible Boolean states. A direct rule baseline solves the entire closed task; model scores do not demonstrate numerical reasoning or general security competence.' + card[end:]
        card += '\nThe same-budget series contains seed accuracies 100%, 100% and 66.67% (mean 88.89%). Seed 41 fails every medium case; reference seed 17 was specified before training, not selected for its test score. Direct rule code scores 100%.\n'
        card += '\nUse `tools/predict.py --normalize --task auth` with the study base, this adapter and a raw evidence record. Omitting `--normalize` changes the evaluated interface.\n'
    (destination / 'README.md').write_text(card)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--task', choices=['auth', 'workflow'], required=True)
    parser.add_argument('--series', choices=['v2', 'size1p7', 'normalized'], default='v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    info = export(args.root, args.task, args.output, args.series)
    print(json.dumps({'adapter_sha256': info['adapter_sha256'], 'adapter_bytes': info['adapter_bytes']}))
