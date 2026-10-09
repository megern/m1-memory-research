"""Paired case-bootstrap factorial contrasts, conditional on eight fixed policies/one seed."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from analyze_public_sms import scores

def analyze(output):
    protocol=json.loads((output/'protocol.json').read_text());cells=protocol['cells'];reports={};paths={}
    if len(cells)!=8:raise ValueError('Complete eight-cell factorial required')
    for cell in cells:
        name=cell['id'];path=output/(name+'-valid.json');report=json.loads(path.read_text())
        if not report['complete']:
            path=output/(name+'-valid-retry.json');report=json.loads(path.read_text())
        if not report['complete']:raise ValueError('Incomplete cell; no complete-case factor estimates')
        if report['data_sha256']!=protocol['splits_sha256']['valid'] or report['split']!='valid':raise ValueError('Validation identity mismatch')
        reports[name]=report;paths[name]=path.name
    ids=[(r['id'],r['label']) for r in reports[cells[0]['id']]['rows']]
    if any([(r['id'],r['label']) for r in report['rows']]!=ids for report in reports.values()):raise ValueError('Case order mismatch')
    y=np.array([label=='spam' for _,label in ids]);pred={name:np.array([r['prediction']=='spam' for r in report['rows']]) for name,report in reports.items()}
    factors={'iterations':(120,480),'learning_rate':(.0001,.0002),'gradient_accumulation_steps':(1,4)};weights={}
    for factor,levels in factors.items():weights[factor]={r['id']:(.25 if r[factor]==levels[1] else -.25) for r in cells}
    for left,right in itertools.combinations(factors,2):
        weights[left+' x '+right]={r['id']:(.5 if (r[left]==factors[left][1])==(r[right]==factors[right][1]) else -.5) for r in cells}
    point={name:scores(y,p)['macro_f1'] for name,p in pred.items()};contrasts={name:sum(point[cell]*weight for cell,weight in w.items()) for name,w in weights.items()};samples={name:[] for name in weights}
    rng=np.random.default_rng(8062);groups=[np.flatnonzero(y),np.flatnonzero(~y)]
    for _ in range(2000):
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);score={name:scores(y[indices],p[indices])['macro_f1'] for name,p in pred.items()}
        for name,w in weights.items():samples[name].append(sum(score[cell]*weight for cell,weight in w.items()))
    result={'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'validation_reports':paths,'cases':len(y),'spam_cases':int(y.sum()),'completed_cells':8,'seed':17,'contrasts':{name:{'point':contrasts[name],'paired_case_ci95':np.quantile(samples[name],[.025,.975]).tolist(),'weights':weights[name]} for name in weights},'definitions':'Main: second level minus first averaged over other policies. Pair interaction: difference of one factor effect between levels of the other, averaged over the third. Units are Macro-F1, multiply by 100 for percentage points.','scope':'Descriptive post-run uncertainty analysis (source hash recorded), not preregistered significance testing; 2000 shared stratified case resamples seed 8062, percentile 95%, no multiplicity correction. Conditional on these policies and this one seed; no population seed uncertainty or unseen-test claim. Eight score vectors share 128 cases and are not independent samples. No best-model/checkpoint selection.'}
    (output/'factor-effects-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({name:{k:v for k,v in row.items() if k!='weights'} for name,row in result['contrasts'].items()},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);analyze(p.parse_args().results)
