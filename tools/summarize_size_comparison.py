"""Retain the exploratory 1.7B baseline and every specified training seed."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root):
    protocol=json.loads((root/'data/size-comparison-protocol.json').read_text())
    data_protocol=json.loads((root/'data/confirmatory-v2/protocol.json').read_text())
    entries=[]
    for seed in [None,*protocol['training_seeds']]:
        name='base' if seed is None else f'seed{seed}'
        report=json.loads((root/'results'/f'auth-size1p7-{name}.json').read_text())
        if report['base'] != protocol['base'] or report['base_revision'] != protocol['base_revision']:
            raise ValueError('Wrong base identity in size comparison')
        if report['data_sha256'] != data_protocol['datasets']['auth']['test']['sha256']:
            raise ValueError('Wrong test split in size comparison')
        entry={'variant':name,'strict':report['summary'],
               'outer_fence_allowed':report['classification_allowing_outer_fence'],
               'by_language':report['classification_allowing_outer_fence_by_language'],
               'evaluation_mlx_peak_bytes':report['mlx_peak_bytes'],
               'evaluation_rss_high_water_bytes':report['rss_high_water_bytes']}
        if seed is not None:
            source=root/'artifacts'/f'auth-size1p7-seed{seed}'
            digest=hashlib.sha256((source/'adapters.safetensors').read_bytes()).hexdigest()
            if digest != report['adapter_sha256']:
                raise ValueError('Wrong adapter weights in size comparison')
            entry.update(adapter_sha256=digest,training=json.loads((source/'training-summary.json').read_text()))
        entries.append(entry)
    aggregate={metric:{'mean':statistics.mean(values := [e['outer_fence_allowed'][metric] for e in entries[1:]]),
                       'min':min(values),'max':max(values)} for metric in ['accuracy','macro_f1']}
    identity=json.loads((root/'results/auth-size1p7-identity-seed17.json').read_text())
    return {'protocol':protocol,'entries':entries,'adapter_aggregate':aggregate,
            'reference_identity_audit':identity['summary'],
            'release_seed':17,'scope':'Exploratory reuse of the previously scored synthetic test; all seeds retained.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh summary path')
    result=summarize(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['adapter_aggregate'],indent=2))
