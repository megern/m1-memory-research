"""Completion-aware descriptive summary, preserving failures and paired rounds."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def analyze(directory):
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['completed_series']:
        raise ValueError('All frozen jobs must finish before descriptive analysis')
    for name,digest in summary['source_sha256'].items():
        if hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source snapshot hash differs')
    if hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest()!=summary['protocol_sha256']:
        raise ValueError('Protocol snapshot hash differs')
    modes={}
    for mode in json.loads((directory/'protocol.json').read_text())['modes']:
        trials=[t for t in summary['trials'] if t['mode']==mode]
        complete=[t for t in trials if t['completed']]
        modes[mode]={'planned':len(trials),'completed':len(complete),'stopped':len(trials)-len(complete),
            'completed_only_median_forward_seconds':statistics.median(t['forward_seconds'] for t in complete) if complete else None,
            'completed_only_median_mlx_bytes':statistics.median(t['peak_mlx_bytes'] for t in complete) if complete else None,
            'completed_only_median_rss_bytes':statistics.median(t['sampled_rss_peak_bytes'] for t in complete) if complete else None,
            'worker_swap_growth_bytes_all_attempts':[t['max_system_swap_growth_bytes'] for t in trials],
            'verification_swap_delta_bytes_all_attempts':[t['verification_swap_delta_bytes'] for t in trials]}
    paired=[]
    for trial in summary['trials']:
        if trial['mode']=='native':continue
        reference=next(t for t in summary['trials'] if t['round']==trial['round'] and t['mode']=='native')
        comparable=trial['completed'] and reference['completed']
        paired.append({'round':trial['round'],'mode':trial['mode'],'both_completed':comparable,
            'greedy_token_sequences_equal':([c['token_ids'] for c in trial['cases']]==[c['token_ids'] for c in reference['cases']]) if comparable else None,
            'forward_seconds_difference_from_native':trial['forward_seconds']-reference['forward_seconds'] if comparable else None})
    result={'source_integrity_verified':True,'all_planned_outcomes_retained':len(summary['trials'])==len(summary['planned_order']),
        'mode_statistics':modes,'same_round_comparisons':paired,
        'limitations':['Completed-only medians are conditional; failure rates must remain beside them.',
            'Stopped durations are not speed measurements of a completed workload.',
            'Two rounds, uncontrolled OS caches and other apps do not establish causal or general speed effects.',
            'Greedy token equality is not full numerical equivalence or general quality.',
            'Worker maximum swap growth and verification net delta have different baselines.',
            'Nocache policy acceptance does not measure physical SSD reads or prove absence of cached pages.']}
    (directory/'descriptive-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();print(json.dumps(analyze(a.results),indent=2))
