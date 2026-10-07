import json
from datetime import datetime, timezone
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
import make_specialist_data as original
import hashlib
import random
import argparse

parser=argparse.ArgumentParser(description='Regenerate deterministic data into a fresh directory; preserve the original frozen protocol.')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
directory=args.output.resolve()
if directory.exists():
 parser.error('Choose a fresh directory; never overwrite the recorded frozen protocol')
manifest={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'rationale':'Pilot auth adapter collapsed to the majority class. This confirmatory protocol balances labels before training and uses fresh test surface templates and time values.',
 'training_seeds':[17,23,41], 'iterations':400, 'layers':16, 'rank':8, 'learning_rate':0.0002,
 'test_selection':'Fixed before training; evaluate final adapters only, no checkpoint selection using test results.',
 'datasets':{}}
for task in ['auth','workflow']:
 labels=['low','medium','high'] if task=='auth' else ['low','high']
 target=directory/task
 target.mkdir(parents=True,exist_ok=True)
 manifest['datasets'][task]={}
 for split,count in [('train',384),('valid',48),('test',96)]:
  rows=[]
  for i in range(count):
   wanted=labels[i%len(labels)]
   for j in range(10000):
    r=original.row(task,split,10000+i*10000+j)
    prompt=r['messages'][1]['content']
    if task=='auth':
     record=json.loads(prompt[prompt.index('{'):])
     secs=random.Random(f'v2:{split}:{i}:{j}').choice({'train':[45,150,290,310,420], 'valid':[90,240,330,450], 'test':[1,60,299,300,301,600]}[split])
     record['window_seconds']=secs
     predicted=('high' if record['subsequent_success'] else 'medium') if record['failed_logins']>=5 and secs<=300 else 'low'
     prompt=prompt[:prompt.index('{')]+json.dumps(record,ensure_ascii=False)
    else:
     predicted=r['label']
    if predicted==wanted:
     r.update(id=f'{task}-v2-{split}-{i:04}',label=wanted)
     r['messages'][1]['content']=prompt
     r['messages'][2]['content']=json.dumps({'label':wanted},separators=(',',':'))
     rows.append(r)
     break
   else:
    raise ValueError('Failed to generate class-balanced data')
  random.Random(f'{task}:{split}:v2').shuffle(rows)
  path=target/(split+'.jsonl')
  path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  manifest['datasets'][task][split]={'rows':len(rows),'labels':{label:sum(r['label']==label for r in rows) for label in labels},'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(directory/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest['datasets'],indent=2))
