"""Post-run leave-one-case-out sensitivity; no new training or held-out scoring."""
import argparse
import json
from pathlib import Path
from audit_sms_confirmation import audit, digest
from public_sms_study import metrics


def influence(directory):
    directory = Path(directory)
    checked = audit(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    reports = {item['cell']: json.loads((directory/item['report']).read_text()) for item in checked['reports']}
    baseline = {name: report['metrics']['macro_f1'] for name, report in reports.items()}
    factors = {'iterations': (120,480), 'learning_rate': (.0001,.0002), 'gradient_accumulation_steps': (1,4)}
    def effects(values):
        return {factor: sum((1 if cell[factor] == levels[1] else -1)*values[cell['id']]/8
                           for cell in protocol['cells']) for factor,levels in factors.items()}
    point = effects(baseline)
    case_count = checked['paired_validation_cases']
    removed = []
    for position in range(case_count):
        values = {name: metrics(report['rows'][:position]+report['rows'][position+1:])['macro_f1'] for name,report in reports.items()}
        case = next(iter(reports.values()))['rows'][position]
        removed.append({'removed_case_id': case['id'], 'label': case['label'], 'effects': effects(values)})
    return {'audit_status': checked['status'], 'protocol_sha256': digest(directory/'protocol.json'),
            'analysis_source_sha256': digest(Path(__file__)), 'input_reports': checked['reports'],
            'baseline_two_seed_mean_effects': point,
            'leave_one_out': {factor: {'minimum': min(row['effects'][factor] for row in removed),
                                     'maximum': max(row['effects'][factor] for row in removed),
                                     'direction_changes': sum(row['effects'][factor]*point[factor]<0 for row in removed)}
                             for factor in factors}, 'removed_cases': removed,
            'scope': 'Post-run diagnostic on the same reused 128-case validation set and two fixed seeds. Leave-one-out ranges are not confidence intervals. No new data, training, population-seed inference, best-model selection or significance claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    try: print(json.dumps(influence(args.directory),indent=2))
    except (ValueError,OSError,KeyError,TypeError) as error: parser.exit(2,f'Influence analysis rejected: {error}\n')

if __name__=='__main__':main()
