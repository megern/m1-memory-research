"""Compare fixed old/new training regimes on paired fresh cases, not different tests."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analyze_public_sms import scores

def analyze(directory):
    protocol=json.loads((directory/'paired-training-control-protocol.json').read_text());pairs={}
    for seed in (17,23,41):
        old=json.loads((directory/f'control-seed-{seed}.json').read_text());new=json.loads((directory/f'seed-{seed}.json').read_text())
        if not old['complete'] or not new['complete']:raise ValueError('Incomplete control pair')
        if old.get('adapter_sha256')!=protocol['control_adapter_sha256'][str(seed)]:raise ValueError('Control adapter mismatch')
        if old.get('data_sha256')!=protocol['test_sha256'] or new.get('data_sha256')!=protocol['test_sha256']:raise ValueError('Test identity mismatch')
        if [(r['id'],r['label']) for r in old['rows']]!=[(r['id'],r['label']) for r in new['rows']]:raise ValueError('Case pairing mismatch')
        pairs[str(seed)]=(old,new)
    ids=[(r['id'],r['label']) for r in pairs['17'][0]['rows']]
    for pair in pairs.values():
        if [(r['id'],r['label']) for r in pair[0]['rows']]!=ids:raise ValueError('Seed pairing mismatch')
    y=np.array([label=='spam' for _,label in ids]);groups=[np.flatnonzero(y),np.flatnonzero(~y)];rng=np.random.default_rng(8062)
    pred={seed:tuple(np.array([r['prediction']=='spam' for r in report['rows']]) for report in pair) for seed,pair in pairs.items()}
    point={seed:tuple(scores(y,p)['macro_f1'] for p in pair) for seed,pair in pred.items()};samples={seed:[] for seed in pred};means=[]
    for _ in range(2000):
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);values=[]
        for seed,(old,new) in pred.items():
            delta=scores(y[indices],new[indices])['macro_f1']-scores(y[indices],old[indices])['macro_f1'];samples[seed].append(delta);values.append(delta)
        means.append(float(np.mean(values)))
    summary={'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_sha256':protocol['test_sha256'],'same_test_cases':len(y),'per_seed':{seed:{'old_macro_f1':old,'new_macro_f1':new,'paired_delta':new-old,'paired_case_bootstrap_ci95':np.quantile(samples[seed],[.025,.975]).tolist()} for seed,(old,new) in point.items()},'old_seed_mean_macro_f1':float(np.mean([p[0] for p in point.values()])),'new_seed_mean_macro_f1':float(np.mean([p[1] for p in point.values()])),'mean_paired_delta_case_ci95':np.quantile(means,[.025,.975]).tolist(),'old_seed_sd':float(np.std([p[0] for p in point.values()],ddof=1)),'new_seed_sd':float(np.std([p[1] for p in point.values()],ddof=1)),'scope':'Exploratory two-regime comparison with three fixed seeds on one fresh held-out corpus split. Longer exposure, learning rate and accumulation changed together; cannot isolate the effect of accumulation or establish algorithmic novelty. Bootstrap conditional on trained seeds, not a population seed-variance interval.'}
    (directory/'paired-training-control-analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);analyze(p.parse_args().results)
