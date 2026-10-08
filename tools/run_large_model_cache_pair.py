"""Two adjacent fresh local 70B processes; one guarded checksum, no intervening rehash."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import psutil

def identity(path):
    stat=path.stat()
    return [stat.st_size,stat.st_mtime_ns,stat.st_ino]

def verify(model,manifest,baseline,observe=None):
    original=identity(model);digest=hashlib.sha256();started=time.monotonic()
    if original[0]!=manifest['bytes']:raise ValueError('Wrong model size')
    with model.open('rb') as source:
        while block:=source.read(8*1024**2):
            digest.update(block)
            sample={'seconds':time.monotonic()-started,'available_bytes':psutil.virtual_memory().available,'system_swap_growth_bytes':psutil.swap_memory().used-baseline}
            if observe:observe(sample)
            if sample['system_swap_growth_bytes']>512*1024**2:raise MemoryError('Verification swap growth over 512 MiB')
            if sample['available_bytes']<512*1024**2:raise MemoryError('Verification available memory below 512 MiB')
            if sample['seconds']>180:raise TimeoutError('Verification timed out')
    if digest.hexdigest()!=manifest['sha256'] or identity(model)!=original:raise ValueError('Hash or original file identity changed')
    return {'sha256':digest.hexdigest(),'identity':original,'seconds':time.monotonic()-started}

def run(runtime,model,manifest_path,output):
    if output.exists():raise ValueError('Fresh output required')
    model,runtime=model.resolve(),runtime.resolve()
    manifest=json.loads(manifest_path.read_text())
    output.mkdir(parents=True);source=output/'source';source.mkdir()
    hashes={}
    for name in ['run_large_model.py','run_large_model_cache_pair.py']:
        raw=Path(__file__).with_name(name).read_bytes();(source/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    (output/'manifest.json').write_bytes(manifest_path.read_bytes())
    protocol={'rounds':1,'fresh_processes':2,'prompt':'Reply with exactly the single English word naming the capital of France.','tokens':8,'timeout_per_process':900,'threads':4,'sequence':'One full-file checksum, process 1 exits, immediately start process 2 without rehash or sleep. No prompt/KV/session cache persists between processes.','limits':['Not cold versus warm: initial checksum already warms OS caches.','A latency change alone cannot quantify cache survival or prove paging caused it.','Same previous IQ2_XXS checkpoint, not original BF16 70B.','One pair, uncontrolled other apps and thermal state; all stops retained.','No LinkedIn replies or publication.']}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summary={'source_sha256':hashes,'protocol_sha256':hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'runtime_binary_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),'trials':[],'completed_series':False}
    def save():(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save();baseline=psutil.swap_memory().used
    samples=[]
    def observe(sample):
        if not samples or sample['seconds']-samples[-1]['seconds']>=.5:samples.append(sample)
    try:verified=verify(model,manifest,baseline,observe)
    except Exception as error:
        summary.update(stage='verification_failed',verification_error=f'{type(error).__name__}: {error}',verification_samples=samples);save();return summary
    summary.update(verification=verified,verification_samples=samples,stage='running');save()
    # Execute the exact copied module, not a subsequently edited working file.
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_large_model',source/'run_large_model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    previous_end=None
    for index in range(2):
        if identity(model)!=verified['identity']:
            summary['stage']='identity_changed_before_process';save();return summary
        beginning=time.monotonic()
        report=module.run(runtime,model,manifest,output/f'trial-{index:02d}',900,8,verified,baseline)
        summary['trials'].append({'index':index,'seconds_since_previous_return':beginning-previous_end if previous_end is not None else None,**{k:report[k] for k in ['smoke_runtime_completed','stopped_by_guard','elapsed_seconds','time_to_first_nonwhitespace_stdout_seconds','observed_process_tree_rss_peak_bytes','output','parsed_timing_lines','model_identity_unchanged_after_run','root_process_pageins_sampled_max','root_process_faults_sampled_max']}})
        previous_end=time.monotonic();save()
        if not report['smoke_runtime_completed']:
            summary['stage']='stopped_no_automatic_retry';save();return summary
    summary.update(completed_series=True,stage='all_planned_processes_ended');save();return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['runtime','model','manifest','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=run(args.runtime,args.model,args.manifest,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in {'verification_samples','trials'}},indent=2))
    if not result['completed_series']:sys.exit(1)
