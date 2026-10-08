"""Paired stratified case bootstrap; keep seeds separate and no best-seed selection."""
import argparse,json
from pathlib import Path
import numpy as np

def scores(y,p):
    tp=np.sum(y&p);tn=np.sum(~y&~p);fp=np.sum(~y&p);fn=np.sum(y&~p)
    def ratio(a,b):return float(a/b) if b else 0.
    return {'macro_f1':(ratio(2*tp,2*tp+fp+fn)+ratio(2*tn,2*tn+fp+fn))/2,'spam_recall':ratio(tp,tp+fn),'spam_precision':ratio(tp,tp+fp),'false_positive_rate':ratio(fp,fp+tn),'accuracy':ratio(tp+tn,len(y))}
def analyze(directory):
    names=['lexical','base','seed-17','seed-23','seed-41'];reports={};paths={}
    for name in names:
        path=directory/(name+'.json')
        if name=='base' and (directory/'base-retry.json').is_file():path=directory/'base-retry.json'
        if path.is_file():
            report=json.loads(path.read_text())
            if report['complete']:reports[name]=report;paths[name]=path.name
    if 'base' not in reports:raise ValueError('A complete base reference is required')
    reference=reports['base']['rows'];ids=[(r['id'],r['label']) for r in reference]
    for report in reports.values():
        if [(r['id'],r['label']) for r in report['rows']]!=ids:raise ValueError('Pair mismatch')
    y=np.array([label=='spam' for _,label in ids]);pred={n:np.array([r['prediction']=='spam' for r in report['rows']]) for n,report in reports.items()}
    groups=[np.flatnonzero(y),np.flatnonzero(~y)];rng=np.random.default_rng(8062)
    point={n:scores(y,p) for n,p in pred.items()};samples={n:{k:[] for k in point[n]} for n in pred};delta={n:[] for n in pred if n!='base'};mean_delta=[];seeds=[n for n in pred if n.startswith('seed-')]
    for _ in range(2000):
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);sample={n:scores(y[indices],p[indices]) for n,p in pred.items()}
        for n in pred:
            for metric,value in sample[n].items():samples[n][metric].append(value)
        for n in delta:delta[n].append(sample[n]['macro_f1']-sample['base']['macro_f1'])
        if seeds:mean_delta.append(float(np.mean([sample[n]['macro_f1'] for n in seeds]))-sample['base']['macro_f1'])
    interval=lambda v:np.quantile(v,[.025,.975]).tolist()
    summary={'reports':paths,'test_cases':len(y),'spam_test_cases':int(y.sum()),'metrics':point,'intervals':{n:{k:interval(v) for k,v in metric.items()} for n,metric in samples.items()},'paired_macro_f1_delta_vs_base':{n:{'point':point[n]['macro_f1']-point['base']['macro_f1'],'ci95':interval(v)} for n,v in delta.items()},'method':'2000 paired label-stratified case resamples, percentile 95%, seed 8062; fixed split; no multiplicity-corrected significance claim.'}
    if seeds:summary['all_seed_summary']={'completed_seeds':len(seeds),'macro_f1_mean':float(np.mean([point[n]['macro_f1'] for n in seeds])),'macro_f1_seed_sd':float(np.std([point[n]['macro_f1'] for n in seeds],ddof=1)) if len(seeds)>1 else None,'mean_delta_vs_base':float(np.mean([point[n]['macro_f1'] for n in seeds]))-point['base']['macro_f1'],'mean_delta_case_bootstrap_ci95':interval(mean_delta),'scope':'Same resampled cases across seeds; interval conditional on these trained seeds. Does not estimate population training-seed uncertainty.'}
    (directory/'quality-analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);analyze(p.parse_args().results)
