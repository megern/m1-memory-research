"""Failure-aware paired description of frozen prompt-schedule trials."""
import argparse,hashlib,json,statistics
from pathlib import Path


def analyze(directory):
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['completed_series']:raise ValueError('All planned jobs must end first')
    protocol_bytes=(directory/'protocol.json').read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest()!=summary['protocol_sha256']:raise ValueError('Protocol changed')
    protocol=json.loads(protocol_bytes)
    field = 'variant' if 'variants' in protocol else 'chunk_size'
    choices = protocol['variants'] if field == 'variant' else protocol['chunk_sizes']
    reference = 'unchunked' if field == 'variant' else 128
    for name,digest in summary['source_sha256'].items():
        if hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed')
    if [(t['round'],t[field]) for t in summary['trials']]!=[(t['round'],t[field]) for t in summary['planned_order']]:raise ValueError('Planned outcomes missing or reordered')
    stats={}
    for size in choices:
        trials=[t for t in summary['trials'] if t[field]==size]
        done=[t for t in trials if t['completed']]
        stats[str(size)]={'planned':len(trials),'completed':len(done),'stopped':len(trials)-len(done)}
        for key in ['forward_seconds','peak_mlx_bytes','sampled_rss_peak_bytes','selected_tensor_bytes','row_bytes_read','layer_weight_loads','layer_calls']:
            values=[t[key] for t in done if t.get(key) is not None]
            stats[str(size)]['completed_only_median_'+key]=statistics.median(values) if values else None
        stats[str(size)]['max_setup_and_worker_swap_bytes_all_attempts']=[t.get('max_setup_and_worker_swap_growth_bytes') for t in trials]
    pairs=[]
    for trial in summary['trials']:
        if trial[field]==reference:continue
        baseline=next(t for t in summary['trials'] if t['round']==trial['round'] and t[field]==reference)
        comparable=trial['completed'] and baseline['completed']
        pairs.append({'round':trial['round'],field:trial[field],'both_completed':comparable,
          'greedy_token_sequences_equal':([c['token_ids'] for c in trial['cases']]==[c['token_ids'] for c in baseline['cases']]) if comparable else None,
          'forward_seconds_difference_from_unchunked':trial['forward_seconds']-baseline['forward_seconds'] if comparable else None,
          'peak_mlx_bytes_difference_from_unchunked':trial['peak_mlx_bytes']-baseline['peak_mlx_bytes'] if comparable else None,
          'sampled_rss_peak_bytes_difference_from_unchunked':trial['sampled_rss_peak_bytes']-baseline['sampled_rss_peak_bytes'] if comparable else None})
    transport_pairs=[]
    if field=='variant':
        for control,candidate,label in [('raw-serial','raw-prefetch1','raw_serial'),('prefix-serial64','prefix-prefetch64','prefix_serial')]:
            if not {control,candidate} <= set(choices):continue
            for trial in summary['trials']:
                if trial[field]!=candidate:continue
                baseline=next(t for t in summary['trials'] if t['round']==trial['round'] and t[field]==control)
                comparable=trial['completed'] and baseline['completed']
                item={'round':trial['round'],'control':control,'candidate':candidate,'both_completed':comparable,
                      'greedy_token_sequences_equal':([c['token_ids'] for c in trial['cases']]==[c['token_ids'] for c in baseline['cases']]) if comparable else None}
                for key in ['forward_seconds','peak_mlx_bytes','sampled_rss_peak_bytes']:
                    item[key+'_difference_from_'+label]=trial[key]-baseline[key] if comparable else None
                transport_pairs.append(item)
    result={'source_integrity_verified':True,'all_planned_outcomes_retained':True,('variant_statistics' if field=='variant' else 'chunk_statistics'):stats,'same_round_comparisons':pairs,'same_transport_read_ahead_comparisons':transport_pairs,'limitations':['Conditional medians exclude stopped attempts; completion counts retained.','Small exploratory workloads, uncontrolled OS caches and other apps; no causal or broad quality claims.','Known chunked prefill changes BF16 kernel shapes; identical short output is not logits equivalence.','Tensor byte counters are logical requested payloads, not physical SSD reads.','No training, quantization, novel algorithm or original 70B claim.']}
    if transport_pairs:
        result['limitations'].append('Prefetched host byte buffers are additional RAM outside MLX allocation counters; RSS includes conversion temporaries. This is known read-ahead, not algorithmic novelty.')
    (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    print(json.dumps(analyze(p.parse_args().directory),indent=2))
