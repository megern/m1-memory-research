"""Recompute confirmation contrasts from complete paired case vectors; no best selection."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from analyze_public_sms import scores

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def analyze(output):
 protocol=json.loads((output/'protocol.json').read_text());cells=protocol['cells'];pred={};paths={};identity=None
 for cell in cells:
  path=output/(cell['id']+'-valid.json')
  if not path.is_file():raise ValueError('All 16 validation cells required')
  report=json.loads(path.read_text())
  if not report['complete']:
   path=output/(cell['id']+'-valid-retry.json')
   if not path.is_file():raise ValueError('Incomplete validation cell')
   report=json.loads(path.read_text())
  if not report['complete'] or report['split']!='valid' or report['data_sha256']!=protocol['splits_sha256']['valid']:raise ValueError('Incomplete or unbound validation report')
  ids=[(r['id'],r['label']) for r in report['rows']]
  if identity is None:identity=ids
  if identity!=ids:raise ValueError('Unpaired validation cases')
  if any(r['prediction'] not in ('ham','spam') or r['label'] not in ('ham','spam') for r in report['rows']):raise ValueError('Unknown labels')
  prediction=np.array([r['prediction']=='spam' for r in report['rows']]);y=np.array([r['label']=='spam' for r in report['rows']])
  computed=scores(y,prediction)
  if abs(computed['macro_f1']-report['metrics']['macro_f1'])>1e-12:raise ValueError('Stored metric differs from prediction vector')
  pred[cell['id']]=prediction;paths[cell['id']]={'path':path.name,'sha256':sha(path)}
 seeds=(23,41);factors={'iterations':(120,480),'learning_rate':(.0001,.0002),'gradient_accumulation_steps':(1,4)};weights={}
 for seed in seeds:
  seed_cells=[c for c in cells if c['seed']==seed]
  if len(seed_cells)!=8 or len({(c['iterations'],c['learning_rate'],c['gradient_accumulation_steps']) for c in seed_cells})!=8:raise ValueError('Incomplete factorial design')
  for name,levels in factors.items():weights[f'seed{seed}:{name}']={c['id']:(.25 if c[name]==levels[1] else -.25) for c in seed_cells}
  for left,right in itertools.combinations(factors,2):weights[f'seed{seed}:{left} x {right}']={c['id']:(.5 if (c[left]==factors[left][1])==(c[right]==factors[right][1]) else -.5) for c in seed_cells}
 contrast_names=list(factors)+[left+' x '+right for left,right in itertools.combinations(factors,2)]
 for name in contrast_names:weights['two_seed_mean:'+name]={cell:weight/2 for seed in seeds for cell,weight in weights[f'seed{seed}:{name}'].items()}
 point_scores={name:scores(y,p)['macro_f1'] for name,p in pred.items()}
 point={name:sum(point_scores[c]*w for c,w in ws.items()) for name,ws in weights.items()}
 samples={name:[] for name in weights};rng=np.random.default_rng(8062);groups=[np.flatnonzero(y),np.flatnonzero(~y)]
 for _ in range(2000):
  indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);values={name:scores(y[indices],p[indices])['macro_f1'] for name,p in pred.items()}
  for name,ws in weights.items():samples[name].append(sum(values[c]*w for c,w in ws.items()))
 result={'completed_cells':16,'seeds':list(seeds),'case_count':len(y),'spam_case_count':int(y.sum()),'report_identities':paths,'analysis_source_sha256':sha(Path(__file__)),'protocol_sha256':sha(output/'protocol.json'),'contrasts':{name:{'point':point[name],'conditional_paired_case_ci95':np.quantile(samples[name],[.025,.975]).tolist()} for name in weights},'scope':'Confirmation on two additional fixed seeds of the same reused validation corpus. Post-run descriptive shared stratified case bootstrap, 2000 resamples seed 8062. Conditional on fixed models; intervals exclude population seed uncertainty. No multiplicity correction, significance claim, unseen test scoring or best-model selection. Two-seed mean is not a guarantee for another seed/domain. Historical models are quality controls, not controlled timing comparisons.'}
 (output/'confirmation-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['contrasts'],indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);analyze(p.parse_args().results)
