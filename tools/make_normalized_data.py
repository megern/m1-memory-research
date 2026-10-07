"""Exploratory decomposed-policy prototype, frozen before training; fresh output only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
from normalized_auth import messages,rule_label
from make_specialist_data import row


def build(output):
    if output.exists():
        raise ValueError('Choose a fresh dataset directory')
    directory=output/'auth'
    directory.mkdir(parents=True)
    protocol={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'status':'Exploratory prototype specified after both raw-numeric model-size comparisons; not a preregistered independent confirmation.',
              'base':'mlx-community/Qwen3-0.6B-4bit','training_seeds':[17,23,41],
              'iterations':400,'layers':16,'rank':8,'learning_rate':0.0002,
              'reference_seed':17,'selection':'Final checkpoint, all seeds retained.',
              'projection':'Python computes count>=5 and seconds<=300; the LLM receives Boolean facts and does not perform those numeric comparisons.',
              'limits':['Only eight possible Boolean feature states; no operational security competence claim.',
                        'Direct rule code solves this closed policy without an LLM.',
                        'New sampling, balanced language strata, wording and feature representation all differ from the raw-numeric series; no single-factor causal attribution.'],
              'datasets':{'auth':{}}}
    for split,count in [('train',384),('valid',48),('test',96)]:
        rows=[]
        for index in range(count):
            label=['low','medium','high'][index%3]
            language='ar' if index%6<3 else 'en'
            for candidate in range(10000):
                original=row('auth',split,200000000+index*10000+candidate)
                text=original['messages'][1]['content']
                record=json.loads(text[text.index('{'):])
                record['window_seconds']=random.Random(f'normalized:{split}:{index}:{candidate}').choice(
                    {'train':[45,150,290,310,420], 'valid':[90,240,330,450], 'test':[1,60,299,300,301,600]}[split])
                if rule_label(record)==label:
                    prompt=messages(record,language,split)
                    prompt.append({'role':'assistant','content':json.dumps({'label':label},separators=(',',':'))})
                    rows.append({'id':f'normalized-auth-{split}-{index:04}', 'language':language,
                                 'label':label,'raw_evidence':record,'messages':prompt})
                    break
            else:
                raise ValueError('Could not satisfy the fixed class stratum')
        random.Random('normalized:'+split).shuffle(rows)
        path=directory/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        protocol['datasets']['auth'][split]={'rows':count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'labels':{label:sum(r['label']==label for r in rows) for label in ['low','medium','high']},
            'language_label_counts':{lang:{label:sum(r['language']==lang and r['label']==label for r in rows)
                                     for label in ['low','medium','high']} for lang in ['ar','en']}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(protocol['datasets'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.output.resolve())
