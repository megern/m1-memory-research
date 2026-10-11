"""Read-only consistency audit of the exported confirmation study; no ML runtime."""
import argparse
import hashlib
import json
from pathlib import Path
from public_sms_study import metrics


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'export-manifest.json').read_text())
    for name, expected in manifest.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError('Export hash mismatch or unsafe path: ' + name)
    if not {'protocol.json', 'journal.json'} <= set(manifest):
        raise ValueError('Required export entries missing from manifest')
    protocol = json.loads((directory / 'protocol.json').read_text())
    journal = json.loads((directory / 'journal.json').read_text())
    for name, expected in protocol['frozen_sources'].items():
        path = (directory / 'source' / name).resolve()
        if not path.is_relative_to(directory / 'source') or 'source/' + name not in manifest or digest(path) != expected:
            raise ValueError('Frozen source identity mismatch')
    if digest(directory / 'source/original-plan.json') != protocol['plan_sha256']:
        raise ValueError('Original plan identity mismatch')
    cells = protocol['cells']
    ids = [cell['id'] for cell in cells]
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError('Expected sixteen distinct planned cells')
    expected_design = {(seed, iterations, rate, accumulation) for seed in (23, 41)
                       for iterations in (120, 480) for rate in (0.0001, 0.0002) for accumulation in (1, 4)}
    actual_design = {(cell['seed'], cell['iterations'], cell['learning_rate'], cell['gradient_accumulation_steps']) for cell in cells}
    if actual_design != expected_design:
        raise ValueError('Incomplete factorial design')
    paired = None
    reports = []
    for name in ids:
        initial = json.loads((directory / (name + '-valid.json')).read_text())
        retry = directory / (name + '-valid-retry.json')
        if retry.exists() and initial['complete']:
            raise ValueError('A completed evaluation cannot be replaced by a retry')
        selected = retry if retry.exists() else directory / (name + '-valid.json')
        if selected.name not in manifest:
            raise ValueError('Selected evaluation missing from manifest')
        report = json.loads(selected.read_text())
        run_key = selected.stem
        if not report['complete'] or journal[run_key]['exit_code'] != 0:
            raise ValueError('Incomplete evaluation: ' + name)
        if report['split'] != 'valid' or report['data_sha256'] != protocol['splits_sha256']['valid']:
            raise ValueError('Wrong evaluation split or dataset')
        if report['verified_base'] != protocol['verified_base']:
            raise ValueError('Base model identity mismatch')
        if report['adapter_sha256'] != journal[name + '-train']['adapter_sha256']:
            raise ValueError('Adapter identity mismatch')
        control = protocol['historical_controls'].get(name)
        if control and report['adapter_sha256'] != control['sha256']:
            raise ValueError('Historical control identity mismatch')
        rows = report['rows']
        identity = [(row['id'], row['label']) for row in rows]
        if len(rows) != 128 or len({row['id'] for row in rows}) != 128:
            raise ValueError('Expected 128 distinct validation cases')
        if any(row['label'] not in ('ham', 'spam') or row['prediction'] not in ('ham', 'spam') for row in rows):
            raise ValueError('Invalid class label')
        if paired is not None and identity != paired:
            raise ValueError('Validation cases are not paired in the same order')
        paired = identity
        if metrics(rows) != report['metrics']:
            raise ValueError('Metrics disagree with prediction vectors')
        reports.append({'cell': name, 'report': selected.name, 'sha256': digest(selected)})
    return {'status': 'passed', 'manifest_entries_verified': len(manifest), 'cells_verified': len(reports),
            'paired_validation_cases': len(paired), 'reports': reports,
            'scope': 'Internal consistency of supplied unkeyed hashes and prediction vectors, not independent authenticity, fresh held-out performance, or a novelty claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.directory), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Audit failed: {error}\n')

if __name__ == '__main__':
    main()
