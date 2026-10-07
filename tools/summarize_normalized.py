"""Verify the feature-projection series and its deterministic rule comparator."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from normalized_auth import features, rule_label


def summarize(root):
    protocol=json.loads((root/'data/normalized-auth/protocol.json').read_text())
    test=root/'data/normalized-auth/auth/test.jsonl'
    digest=hashlib.sha256(test.read_bytes()).hexdigest()
    if digest != protocol['datasets']['auth']['test']['sha256']:
        raise ValueError('Changed frozen test data')
    cases=[json.loads(line) for line in test.read_text().splitlines()]
    direct_correct=sum(rule_label(case['raw_evidence'])==case['label'] for case in cases)
    test_states={tuple(features(case['raw_evidence']).values()) for case in cases}
    entries=[]
    for seed in [None,*protocol['training_seeds']]:
        name='base' if seed is None else f'seed{seed}'
        report=json.loads((root/'results'/f'auth-normalized-{name}.json').read_text())
        if report['data_sha256']!=digest or report['base']!=protocol['base']:
            raise ValueError('Wrong data or base attribution')
        entry={'variant':name,'strict':report['summary'],
               'outer_fence_allowed':report['classification_allowing_outer_fence'],
               'by_language':report['by_language']}
        if seed is not None:
            source=root/'artifacts'/f'auth-normalized-seed{seed}'
            weight_digest=hashlib.sha256((source/'adapters.safetensors').read_bytes()).hexdigest()
            if weight_digest!=report['adapter_sha256']:
                raise ValueError('Wrong adapter weights')
            entry.update(adapter_sha256=weight_digest,training=json.loads((source/'training-summary.json').read_text()))
        entries.append(entry)
    aggregate={metric:{'mean':statistics.mean(values := [e['strict'][metric] for e in entries[1:]]),
                       'min':min(values),'max':max(values)} for metric in ['accuracy','macro_f1']}
    return {'protocol':protocol,'entries':entries,'adapter_aggregate':aggregate,
            'distinct_test_boolean_states':len(test_states),
            'direct_rule_baseline':{'cases':len(cases),'correct':direct_correct,'accuracy':direct_correct/len(cases)},
            'release_seed':17,'scope':'Combined code projection and LLM classification on eight Boolean states; not LLM arithmetic or general incident reasoning.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output')
    result=summarize(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'adapter_aggregate':result['adapter_aggregate'],'direct_rule_baseline':result['direct_rule_baseline']},indent=2))
