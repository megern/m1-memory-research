"""Descriptive setup-only timing with intact file identities and failure counts."""
import argparse,hashlib,json,statistics
from pathlib import Path


def analyze(root):
    s=json.loads((root/'summary.json').read_text())
    if not s['completed_series']:raise ValueError('All planned attempts must finish')
    if [(t['round'],t['workers']) for t in s['trials']]!=[(t['round'],t['workers']) for t in s['planned_order']]:raise ValueError('Outcomes missing or reordered')
    for name,digest in s['source_sha256'].items():
        if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed')
    if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()!=s['manifest_sha256']:raise ValueError('Manifest changed')
    base=next((t['identities'] for t in s['trials'] if t['completed']),None)
    stats={}
    for workers in [1,8]:
        trials=[t for t in s['trials'] if t['workers']==workers];done=[t for t in trials if t['completed']]
        stats[str(workers)]={'planned':len(trials),'completed':len(done),'stopped':len(trials)-len(done),'median_verification_seconds':statistics.median(t['verification']['seconds'] for t in done) if done else None,'mean_process_cpu_cores_during_verification':[],'max_recorded_process_threads':[],'max_system_swap_growth_bytes':[]}
        for t in done:
            if t['identities']!=base:raise ValueError('Verified file identities changed between trials')
            samples=t['verification']['samples'];a,b=samples[0],samples[-1]
            stats[str(workers)]['mean_process_cpu_cores_during_verification'].append((b['process_cpu_seconds']-a['process_cpu_seconds'])/(b['seconds']-a['seconds']))
            stats[str(workers)]['max_recorded_process_threads'].append(max(v['process_threads'] for v in samples))
            stats[str(workers)]['max_system_swap_growth_bytes'].append(max(0,max(v['system_swap_growth_bytes'] for v in samples)))
    result={'source_integrity_verified':True,'verified_file_identities_equal_all_completed':base is not None,'statistics':stats,'limitations':['Two exploratory rounds, same-process setup-only work, OS caches and other apps uncontrolled.','Both orders happened to be 8 workers then 1; no balanced crossover or cold-cache claim.','Setup time improvement is not an inference or token-generation speedup.']}
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    print(json.dumps(analyze(p.parse_args().directory),indent=2))
