"""Separate optimizer residency, MLX allocation and RSS in paired training reports.

This audits recorded local-training reports, not the machine itself. It refuses
unmatched model/data/runtime identities and never infers scientific novelty.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError('Training reports must be objects')
    fields = ('peak_optimizer_resident_bytes', 'mlx_peak_bytes_this_process_segment',
              'process_rss_high_water_bytes', 'end_to_end_seconds_this_process_segment',
              'examples_consumed', 'tokens', 'heldout_nll_before', 'heldout_nll_after')
    for report in (left, right):
        if not isinstance(report.get('identity'), dict) or not isinstance(report['identity'].get('config'), dict):
            raise ValueError('Missing input/configuration identity')
        for field in fields:
            value = report.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (
                    value == 0 and not field.startswith('heldout_nll')):
                raise ValueError('Invalid finite measurement: ' + field)
        coverage = report.get('layer_update_coverage')
        if not isinstance(coverage, list) or not coverage or any(type(x) is not int or x < 0 for x in coverage):
            raise ValueError('Invalid layer update coverage')
    for field in ('train_sha256', 'valid_sha256', 'model_files', 'versions'):
        if field not in left['identity'] or not left['identity'][field] or left['identity'][field] != right['identity'].get(field):
            raise ValueError('Unmatched input identity: ' + field)
    for field in ('seed', 'rank', 'lora_scale', 'learning_rate', 'microbatch_size', 'accumulation_steps', 'max_seq_length'):
        if field not in left['identity']['config'] or left['identity']['config'][field] != right['identity']['config'].get(field):
            raise ValueError('Unmatched optimization setting: ' + field)
    if len(left['layer_update_coverage']) != len(right['layer_update_coverage']):
        raise ValueError('Different layer counts')
    if not math.isclose(left['heldout_nll_before'], right['heldout_nll_before'], abs_tol=1e-5, rel_tol=1e-5):
        raise ValueError('Initial validation NLL differs')
    metrics = {}
    for field in fields[:4]:
        metrics[field] = {'left': left[field], 'right': right[field],
                          'right_fraction_of_left': right[field] / left[field],
                          'reduction_percent': 100 * (1 - right[field] / left[field])}
    return {'status': 'paired recorded-resource comparison', 'measurements': metrics,
            'fairness': {'same_example_exposure': left['examples_consumed'] == right['examples_consumed'],
                         'same_supervised_target_count': left['tokens'] == right['tokens'],
                         'same_layer_update_coverage': left['layer_update_coverage'] == right['layer_update_coverage']},
            'validation_nll_difference': right['heldout_nll_after'] - left['heldout_nll_after'],
            'limitations': ['Resident optimizer bytes are not total RAM savings.',
                           'MLX allocation is not a system-wide physical-memory measurement.',
                           'Process RSS is a high-water mark and excludes other applications.',
                           'Elapsed times begin after model/data loading; shared-machine interference is uncontrolled.',
                           'Unequal update coverage or exposure prevents attributing quality changes to memory management alone.',
                           'Reported identities are compared, not independently authenticated; no novelty or model-fit claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in (args.left.resolve(), args.right.resolve()):
            raise ValueError('Output must not overwrite input reports')
        raw = [path.read_bytes() for path in (args.left, args.right)]
        result = compare(*[json.loads(value) for value in raw])
        result['input_sha256'] = [hashlib.sha256(value).hexdigest() for value in raw]
        text = json.dumps(result, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end='')
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(2, 'Resource audit failed: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
