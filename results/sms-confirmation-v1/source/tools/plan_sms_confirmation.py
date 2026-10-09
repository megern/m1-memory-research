"""Freeze a validation-only multi-seed confirmation; never select on test scores."""
import argparse, hashlib, itertools, json
from pathlib import Path

def plan():
 cells=[]
 for seed, exposure, rate, accumulation in itertools.product((23,41),(120,480),(1e-4,2e-4),(1,4)):
  cells.append({'id':f'seed{seed}-n{exposure}-lr{rate}-a{accumulation}','seed':seed,'iterations':exposure,'learning_rate':rate,'gradient_accumulation_steps':accumulation,'optimizer_updates':exposure//accumulation})
 return {'status':'planned_not_executed','cells':cells,'split':'valid','test_evaluation':False,'primary':'Macro-F1; average factorial contrasts separately per seed, then report both seed contrasts and their mean','selection':'All 16 cells retained; no best model selection; no replacing failures with more favorable runs','limitations':'Validation has been reused. This confirms seed sensitivity on this corpus, not unseen-domain generalization. Two additional seeds are not adequate for broad seed-population inference. Accumulation changes optimizer-update count at fixed exposure.','guards':{'process_rss_bytes':6*1024**3,'mlx_allocation_bytes':4*1024**3,'system_swap_growth_bytes':1536*1024**2,'minimum_available_bytes':512*1024**2,'timeout_seconds':900},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(plan(),f,indent=2);f.write('\n')
