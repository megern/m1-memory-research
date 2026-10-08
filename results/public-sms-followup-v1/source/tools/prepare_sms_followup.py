"""Follow-up split excludes every source group used in the exposed first study."""
import argparse,hashlib,json,random,shutil
from collections import Counter
from pathlib import Path
from prepare_public_sms import groups,messages

def prepare(source,previous,output):
    if output.exists():raise ValueError('Fresh output required')
    prior=json.loads((previous/'protocol.json').read_text());raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=prior['source']['member_sha256']:raise ValueError('Source mismatch')
    used={i for meta in prior['splits'].values() for members in meta['source_group_ids'].values() for i in members}
    rows=[]
    for index,line in enumerate(raw.decode().splitlines()):
        label,text=line.split('\t',1);rows.append({'id':f'uci-{index:05d}','label':label,'text':text})
    clusters,_=groups(rows);pool={'ham':[],'spam':[]};membership={}
    for group in clusters:
        if used&{r['id'] for r in group} or len({r['label'] for r in group})!=1:continue
        row=min(group,key=lambda x:x['id']);pool[row['label']].append(row);membership[row['id']]=[r['id'] for r in group]
    rng=random.Random(20261009)
    for values in pool.values():values.sort(key=lambda x:x['id']);rng.shuffle(values)
    selected=pool['spam'][:26]+pool['ham'][:230]
    if len(selected)!=256:raise ValueError('Insufficient fresh groups')
    rng.shuffle(selected);output.mkdir(parents=True)
    for name in ('train','valid'):
        path=previous/(name+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=prior['splits'][name]['sha256']:raise ValueError('Previous split changed')
        shutil.copy2(path,output/path.name)
    payload=''.join(json.dumps({'id':r['id'],'label':r['label'],'language':'en','messages':messages(r['text'])+[{'role':'assistant','content':json.dumps({'label':r['label']},separators=(',',':'))}]},ensure_ascii=False)+'\n' for r in selected).encode()
    (output/'test.jsonl').write_bytes(payload)
    prior['splits']['test']={'rows':256,'labels':dict(Counter(r['label'] for r in selected)),'sha256':hashlib.sha256(payload).hexdigest(),'source_group_ids':{r['id']:membership[r['id']] for r in selected}}
    prior.update(followup={'reason':'First 120-step experiment exposed seed sensitivity on its test. This is an exploratory follow-up with a new independent test, not a preregistered replication. No old test group enters the new test.','previous_protocol_sha256':hashlib.sha256((previous/'protocol.json').read_bytes()).hexdigest(),'fresh_test_seed':20261009,'fixed_test_composition':'26 spam/230 ham, matching prior composition; not an estimate of deployment prevalence.','validation_use':'Same diagnostic validation set. Hyperparameters fixed before new test; no best-seed/checkpoint/threshold selection.'})
    prior['training'].update(iterations=480,learning_rate=.0001,gradient_accumulation_steps=4)
    (output/'protocol.json').write_text(json.dumps(prior,indent=2)+'\n');print(json.dumps(prior['followup'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','previous','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.previous,a.output)
