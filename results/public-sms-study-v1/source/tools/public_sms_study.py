"""Prospective, local-only public SMS adaptation experiment; no test selection."""
import argparse,collections,hashlib,json,math,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path

def read_rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def metrics(rows):
    tp=sum(r['label']=='spam' and r['prediction']=='spam' for r in rows)
    tn=sum(r['label']=='ham' and r['prediction']=='ham' for r in rows)
    fp=sum(r['label']=='ham' and r['prediction']=='spam' for r in rows);fn=len(rows)-tp-tn-fp
    ratio=lambda a,b:a/b if b else 0.
    return {'macro_f1':(ratio(2*tp,2*tp+fp+fn)+ratio(2*tn,2*tn+fp+fn))/2,'spam_precision':ratio(tp,tp+fp),'spam_recall':ratio(tp,tp+fn),'false_positive_rate':ratio(fp,fp+tn),'accuracy':ratio(tp+tn,len(rows)),'tp':tp,'tn':tn,'fp':fp,'fn':fn}
def lexical(train,test):
    words=lambda r:re.findall(r'\w+',json.loads(r['messages'][1]['content'])['sms'].casefold())
    counts={k:collections.Counter(w for r in train if r['label']==k for w in words(r)) for k in ('ham','spam')}
    vocab=set(counts['ham'])|set(counts['spam']);totals={k:sum(v.values())+len(vocab) for k,v in counts.items()}
    priors=collections.Counter(r['label'] for r in train)
    result=[]
    for row in test:
        score={k:math.log(priors[k]/len(train))+sum(math.log((counts[k][w]+1)/totals[k]) for w in words(row) if w in vocab) for k in counts}
        result.append({'id':row['id'],'label':row['label'],'prediction':'spam' if score['spam']>score['ham'] else 'ham','margin':score['spam']-score['ham']})
    return result

def evaluate(model,data,output,adapter=None):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    from provenance import verify_base
    verified=verify_base(model)
    import mlx.core as mx
    from mlx_lm import load
    mx.set_memory_limit(4*1024**3)
    lm,tok=load(str(model),adapter_path=str(adapter) if adapter else None)
    labels={'ham':5604,'spam':75545}
    for label,token in labels.items():
        if tok.encode(label,add_special_tokens=False)!=[token]:raise ValueError('Candidate token mismatch')
    rows=read_rows(data/'test.jsonl');results=[];started=time.monotonic()
    for row in rows:
        prefix=tok.apply_chat_template(row['messages'][:2],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'{"label":"'
        ids=tok.encode(prefix,add_special_tokens=False)
        for label,token in labels.items():
            if tok.encode(prefix+label,add_special_tokens=False)!=ids+[token]:raise ValueError('Candidate boundary mismatch')
        logits=lm(mx.array(ids)[None,:])[:,-1,:];mx.eval(logits)
        margin=float((logits[0,labels['spam']].astype(mx.float32)-logits[0,labels['ham']].astype(mx.float32)).item())
        if not math.isfinite(margin):raise ValueError('Nonfinite classification margin')
        results.append({'id':row['id'],'label':row['label'],'prediction':'spam' if margin>0 else 'ham','margin':margin})
        if len(results)%32==0:print('Evaluated',len(results),flush=True)
        output.write_text(json.dumps({'complete':len(results)==len(rows),'verified_base':verified,'rows':results,'metrics':metrics(results),'elapsed_seconds':time.monotonic()-started,'mlx_peak_bytes':mx.get_peak_memory()},indent=2)+'\n')

def guarded(cmd,log,timeout=900):
    import psutil
    baseline=psutil.swap_memory().used;started=time.monotonic();low_since=None;peak=0;reason=None
    with log.open('w') as stream:
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                proc=psutil.Process(child.pid);rss=sum(p.memory_info().rss for p in [proc]+proc.children(recursive=True));peak=max(peak,rss)
            except psutil.Error:rss=0
            now=time.monotonic();available=psutil.virtual_memory().available;swap=psutil.swap_memory().used-baseline
            low_since=(low_since or now) if available<512*1024**2 else None
            if rss>6*1024**3:reason='process RSS above 6 GiB'
            elif swap>512*1024**2:reason='whole-system swap growth above 512 MiB (not causal attribution)'
            elif low_since and now-low_since>=5:reason='available memory below 512 MiB for 5 seconds'
            elif now-started>timeout:reason='900 second timeout'
            if reason:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(.5)
    return {'exit_code':child.returncode,'guard_stop':reason,'peak_process_tree_rss_bytes':peak,'elapsed_seconds':time.monotonic()-started,'system_swap_growth_bytes':psutil.swap_memory().used-baseline,'log':log.name}

def analyze(output):
    import numpy as np
    names=['lexical','base']+[f'seed-{s}' for s in (17,23,41)]
    reports={n:json.loads((output/(n+'.json')).read_text()) for n in names if (output/(n+'.json')).is_file()}
    complete={n:r for n,r in reports.items() if r['complete']}
    summary={'completed':list(complete),'metrics':{n:r['metrics'] for n,r in complete.items()},'intervals':{},'bootstrap':'2000 label-stratified paired case resamples; percentile 95%; conditional on this fixed split; no multiple-comparison significance claim.'}
    if 'base' in complete:
        base=complete['base']['rows'];rng=np.random.default_rng(8062);groups=[[i for i,r in enumerate(base) if r['label']==k] for k in ('ham','spam')];scores={n:[] for n in complete};deltas={n:[] for n in complete if n!='base'}
        for r in complete.values():
            if [(x['id'],x['label']) for x in r['rows']]!=[(x['id'],x['label']) for x in base]:raise ValueError('Paired case mismatch')
        for _ in range(2000):
            indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            sample={n:metrics([r['rows'][int(i)] for i in indices])['macro_f1'] for n,r in complete.items()}
            for n in scores:scores[n].append(sample[n])
            for n in deltas:deltas[n].append(sample[n]-sample['base'])
        for n,values in scores.items():summary['intervals'][n]={'macro_f1_95':np.quantile(values,[.025,.975]).tolist()}
        for n,values in deltas.items():summary['intervals'][n]['paired_delta_vs_base_95']=np.quantile(values,[.025,.975]).tolist()
    (output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

def study(model,data,output):
    if output.exists():raise ValueError('Fresh study output required')
    output.mkdir(parents=True);protocol=json.loads((data/'protocol.json').read_text());all_ids=set()
    for split,meta in protocol['splits'].items():
        path=data/(split+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Split hash mismatch')
        members={i for group in meta['source_group_ids'].values() for i in group}
        if all_ids&members:raise ValueError('Source group leaks across splits')
        all_ids|=members
    source=output/'source';(source/'tools').mkdir(parents=True);(source/'models').mkdir()
    root=Path(__file__).resolve().parents[1]
    for name in ['public_sms_study.py','prepare_public_sms.py','train_specialist.py','provenance.py']:shutil.copy2(root/'tools'/name,source/'tools'/name)
    shutil.copy2(root/'models/qwen3-0.6b-original.json',source/'models/qwen3-0.6b-original.json')
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    protocol.update(frozen_source_sha256=hashes,guards={'rss_gib':6,'system_swap_growth_mib':512,'available_mib':512,'available_grace_seconds':5,'timeout_seconds_per_child':900})
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=lexical(read_rows(data/'train.jsonl'),read_rows(data/'test.jsonl'))
    (output/'lexical.json').write_text(json.dumps({'complete':True,'rows':rows,'metrics':metrics(rows),'method':'Word multinomial NB, Laplace alpha 1, train-only vocabulary and priors, ignore unseen words; no tuning.'},indent=2)+'\n')
    journal={};runner=source/'tools/public_sms_study.py'
    def save(): (output/'journal.json').write_text(json.dumps(journal,indent=2)+'\n')
    cmd=[sys.executable,str(runner),'--mode','evaluate','--model',str(model),'--data',str(data)]
    journal['base']=guarded(cmd+['--output',str(output/'base.json')],output/'base.log');save()
    for seed in protocol['training']['seeds']:
        adapter=output/f'adapter-{seed}';name=f'seed-{seed}'
        journal[name+'-train']=guarded([sys.executable,str(source/'tools/train_specialist.py'),'--model',str(model),'--data',str(data),'--output',str(adapter),'--seed',str(seed),'--layers','16','--iters','120'],output/(name+'-train.log'));save()
        if journal[name+'-train']['exit_code']==0:
            journal[name+'-evaluate']=guarded(cmd+['--adapter',str(adapter),'--output',str(output/(name+'.json'))],output/(name+'-evaluate.log'));save()
        if journal[name+'-train']['guard_stop']:break
    analyze(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['study','evaluate','analyze'],default='study')
    p.add_argument('--model',type=Path);p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter',type=Path)
    a=p.parse_args()
    if a.mode=='analyze':analyze(a.output)
    elif a.mode=='evaluate':evaluate(a.model,a.data,a.output,a.adapter)
    else:study(a.model,a.data,a.output)
