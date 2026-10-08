"""Describe a single consecutive fresh-process pair without inferring cache residency."""
import argparse,hashlib,json
from pathlib import Path

def analyze(directory):
    summary=json.loads((directory/'summary.json').read_text())
    for name,digest in summary['source_sha256'].items():
        if hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed')
    for filename,key in [('protocol.json','protocol_sha256'),('manifest.json','manifest_sha256')]:
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=summary[key]:raise ValueError('Pinned study changed')
    trials=summary['trials'];complete=summary['completed_series'] and len(trials)==2 and all(t['smoke_runtime_completed'] for t in trials)
    result={'source_and_protocol_integrity_verified':True,'both_processes_completed':complete,'planned':2,'recorded':len(trials),'trials':trials,'limitations':['One sequential pair, not a cold/warm randomized comparison.','A full-file checksum precedes the first process; no checksum between them.','Fresh processes exclude application KV and session caches; OS caches and other apps remain uncontrolled.','First nonwhite stdout includes loading and prefill; not an exact token callback.','A time difference alone cannot measure page-cache residency or establish why it changed.','Page-in/fault maxima are sampled before exit, not physical SSD byte totals.','This IQ2_XXS 70B model is quantized; original BF16 studies are separate.']}
    if complete:
        first,second=trials
        a,b=first['time_to_first_nonwhitespace_stdout_seconds'],second['time_to_first_nonwhitespace_stdout_seconds']
        result.update(first_stdout_seconds_difference=b-a,observed_first_stdout_reduction_percent=(1-b/a)*100,output_sequences_equal=first['output']==second['output'],no_rehash_between_processes=True)
    (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    print(json.dumps(analyze(parser.parse_args().directory),indent=2))
