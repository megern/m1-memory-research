"""Freeze matched-control identities before any new adapter test result exists."""
import argparse,hashlib,json,time
from pathlib import Path

def freeze(results,controls):
    destination=results/'paired-training-control-protocol.json'
    if destination.exists():raise ValueError('A control plan already exists')
    if any((results/f'seed-{seed}.json').exists() for seed in (17,23,41)):raise ValueError('Control plan must precede new adapter test results')
    new=json.loads((results/'protocol.json').read_text());old=json.loads((controls/'protocol.json').read_text())
    if new['splits']['train']['sha256']!=old['splits']['train']['sha256']:raise ValueError('Matched regimes must share training data')
    hashes={}
    for seed in (17,23,41):
        path=controls/f'adapter-{seed}'
        summary=json.loads((path/'training-summary.json').read_text())
        if summary['iterations']!=old['training']['iterations']:raise ValueError('Incomplete or mismatched old training')
        hashes[str(seed)]=hashlib.sha256((path/'adapters.safetensors').read_bytes()).hexdigest()
    script=results/'source/tools/public_sms_study.py'
    plan={'planned_at_unix':time.time(),'purpose':'Paired old/new training-regime evaluation on the same fresh test, all fixed seeds, no best-seed selection.','test_sha256':new['splits']['test']['sha256'],'frozen_evaluator_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'control_adapter_sha256':hashes,'old_training':old['training'],'new_training':new['training'],'primary':'Paired macro-F1 delta per seed and mean; 2000 stratified paired case bootstrap resamples, seed 8062.','guard_swap_mib':new['guards']['system_swap_growth_mib']}
    destination.write_text(json.dumps(plan,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--controls',type=Path,required=True);a=p.parse_args();freeze(a.results,a.controls)
