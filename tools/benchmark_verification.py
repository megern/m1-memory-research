"""Bounded paired file-hashing study; does not load or execute any model."""
import argparse,hashlib,json,random,time
from pathlib import Path
from run_direct_original import verify_guarded


def run(model,manifest_path,output,rounds=2,seed=8057):
    if output.exists():raise ValueError('Fresh output required')
    manifest=json.loads(manifest_path.read_text());output.mkdir(parents=True)
    source=output/'source';source.mkdir()
    hashes={}
    for name in ['benchmark_verification.py','run_direct_original.py','original_tensor_store.py']:
        raw=Path(__file__).with_name(name).read_bytes();(source/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    (output/'manifest.json').write_bytes(manifest_path.read_bytes())
    rng=random.Random(seed);jobs=[]
    for r in range(rounds):
        workers=[1,8];rng.shuffle(workers);jobs.extend({'round':r,'workers':w} for w in workers)
    summary={'planned_order':jobs,'source_sha256':hashes,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'trials':[],'completed_series':False,'scope':'Setup-only comparison, no inference; OS caches and other apps uncontrolled, F_NOCACHE is a descriptor hint, not cold-cache proof.'}
    for job in jobs:
        started=time.monotonic();identities,phase=verify_guarded(model,manifest,900,True,job['workers'])
        item={**job,'completed':identities is not None,'verification':{k:v for k,v in phase.items() if k!='started_monotonic'},'elapsed_seconds':time.monotonic()-started,'verified_files':len(identities) if identities else 0,'identities':identities}
        summary['trials'].append(item)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in item.items() if k not in {'verification','identities'}},indent=2),flush=True)
    summary['completed_series']=True
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model.resolve(),a.manifest.resolve(),a.output.resolve())
