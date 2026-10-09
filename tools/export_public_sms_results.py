"""Export audit reports and frozen source only, never raw SMS or model weights."""
import argparse,hashlib,json,shutil
from pathlib import Path

def export(source,target):
    if target.exists():raise ValueError('Fresh export target required')
    target.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and file.suffix in {'.json','.log'}:shutil.copy2(file,target/file.name)
    if (source/'source').is_dir():
        shutil.copytree(source/'source',target/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    adapters={}
    for directory in sorted(set(source.glob('adapter-*')) | set(source.glob('*-adapter'))):
        if not directory.is_dir():continue
        summary=directory/'training-summary.json'
        if summary.is_file():
            destination=target/(directory.name+'-training-summary.json');shutil.copy2(summary,destination)
        weights=directory/'adapters.safetensors'
        if weights.is_file():adapters[directory.name]={'sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'bytes':weights.stat().st_size,'completed_final_training':summary.is_file(),'hash_recorded_at_export':True}
    (target/'local-adapter-inventory.json').write_text(json.dumps({'weights_uploaded':False,'scope':'Post-run inventory; earlier evaluator hashes where present provide the evaluation-time identity. A stopped run final-weight filename does not imply completed final training.','adapters':adapters},indent=2)+'\n')
    (target/'export-manifest.json').write_text(json.dumps({str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob('*')) if p.is_file()},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);a=p.parse_args();export(a.source,a.target)
