"""Frozen sequential randomized original-weight prefill schedule comparison, all outcomes kept."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


def order(protocol):
    field = 'variant' if 'variants' in protocol else 'chunk_size'
    choices = list(protocol['variants']) if field == 'variant' else protocol['chunk_sizes']
    rng = random.Random(protocol['seed'])
    jobs = []
    for round_index in range(protocol['rounds']):
        modes = choices.copy()
        rng.shuffle(modes)
        jobs.extend({'round':round_index,field:mode} for mode in modes)
    return jobs


def summarize(report):
    result = next((e for e in report['events'] if e['event']=='direct_original_result'),None)
    steps = [e for e in report['events'] if e['event']=='step']
    return {'completed':report['completed'],'guard':report['stopped_by_guard'],
            'worker_seconds':report['worker_elapsed_seconds'],
            'verification_seconds':report['verification_seconds'],
            'verification_swap_delta_bytes':report.get('verification_system_swap_delta_bytes'),
            'forward_seconds':sum(e['elapsed_seconds'] for e in steps),
            'recorded_steps':len(steps),
            'peak_mlx_bytes':result['mlx_peak_allocated_bytes'] if result else
                max((e.get('mlx_peak_bytes_so_far',0) for e in steps),default=None),
            'peak_mlx_scope':'full' if result else 'observed completed steps only',
            'sampled_rss_peak_bytes':report['sampled_process_tree_rss_peak_bytes'],
            'max_system_swap_growth_bytes':max((v['system_swap_growth_bytes'] for v in report['measurements']),default=0),
            'token_events':[{'case':e['case'],'step':e['step'],'token_id':e['token_id']} for e in steps],
            'cases':result['cases'] if result else None,
            'nocache_descriptors':result.get('f_nocache_descriptors') if result else None,
            'layer_calls':result.get('layer_calls') if result else None,
            'layer_weight_loads':result.get('layer_weight_loads') if result else None,
            'prefill_chunks':result.get('prefill_chunks') if result else None,
            'head_passes':result.get('head_passes') if result else None,
            'row_bytes_read':result.get('row_bytes_read_by_pread') if result else None,
            'selected_tensor_bytes':result.get('selected_native_tensor_bytes_requested') if result else None,
            'max_setup_and_worker_swap_growth_bytes':max([0]+[v['system_swap_growth_bytes'] for v in report.get('verification_measurements',[])]+[v['system_swap_growth_bytes'] for v in report['measurements']])}


def run(model, manifest, protocol_path, output):
    if output.exists():
        raise ValueError('Fresh trial directory required')
    protocol = json.loads(protocol_path.read_text())
    pinned = json.loads(manifest.read_text())
    if (protocol['repository'],protocol['revision']) != (pinned['repository'],pinned['revision']):
        raise ValueError('Protocol and original model identities differ')
    jobs = order(protocol)
    output.mkdir(parents=True)
    snapshot = output/'source'
    snapshot.mkdir()
    hashes = {}
    for name in ['run_direct_original.py','direct_original_engine.py','original_tensor_store.py','run_prefill_trials.py']:
        raw = Path(__file__).with_name(name).read_bytes()
        (snapshot/name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    (output/'manifest.json').write_bytes(manifest.read_bytes())
    (output/'protocol.json').write_bytes(protocol_path.read_bytes())
    summary = {'repository':pinned['repository'],'revision':pinned['revision'],
               'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
               'source_sha256':hashes,'planned_order':jobs,'trials':[],
               'completed_series':False,'stage':'running'}
    def save():
        temporary = output/'summary.tmp'
        temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(output/'summary.json')
    save()
    for index, job in enumerate(jobs):
        for name,digest in hashes.items():
            if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen source changed')
        label = job.get('variant',job.get('chunk_size'))
        directory = output/f'trial-{index:02d}-{label}'
        config = protocol['variants'][job['variant']] if 'variant' in job else {'chunk_size':job['chunk_size'],'schedule':'chunk-major'}
        command = [sys.executable,str(snapshot/'run_direct_original.py'),
                   '--model',str(model),'--manifest',str(output/'manifest.json'),
                   '--output',str(directory),'--io-mode','native',
                   '--verification-nocache','--prefill-chunk-size',str(config['chunk_size']),
                   '--prefill-schedule',config['schedule'],
                   '--tokens',str(protocol['tokens_per_case']),
                   '--budget-mib',str(protocol['budget_mib']),
                   '--head-rows',str(protocol['head_rows']),
                   '--timeout',str(protocol['timeout_seconds'])]
        for prompt in protocol['prompts']:
            command.extend(['--prompt',prompt])
        with (output/f'trial-{index:02d}.runner-stdout.txt').open('w') as stdout, (output/f'trial-{index:02d}.runner-stderr.txt').open('w') as stderr:
            exit_code = subprocess.call(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL)
        report = directory/'report.json'
        item = {'index':index,**job,'runner_exit_code':exit_code}
        if report.exists():
            item.update(summarize(json.loads(report.read_text())))
        else:
            item.update(completed=False,guard='no_report')
        summary['trials'].append(item)
        save()
        print(json.dumps({k:v for k,v in item.items() if k not in {'cases','token_events'}},ensure_ascii=False),flush=True)
    summary.update(completed_series=True,stage='finished_all_planned_jobs')
    save()
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.model.resolve(),a.manifest.resolve(),a.protocol.resolve(),a.output.resolve())
