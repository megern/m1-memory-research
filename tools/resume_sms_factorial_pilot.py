"""One disclosed continuation of a pressure-stopped pilot, with unchanged frozen cells."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def resume(model,data,output):
    protocol=json.loads((output/'protocol.json').read_text());destination=output/'continuation-protocol.json'
    if destination.exists():raise ValueError('One continuation only; choose no automatic additional retries')
    for name,digest in protocol['frozen_sources'].items():
        if sha(output/'source'/name)!=digest:raise ValueError('Frozen sources changed')
    for name,digest in protocol['splits_sha256'].items():
        if sha(data/(name+'.jsonl'))!=digest:raise ValueError('Dataset changed')
    original=json.loads((output/'journal.json').read_text())
    if not any(v.get('guard_stop') for v in original.values()):raise ValueError('Continuation requires a recorded pressure stop')
    script=output/'source/tools/public_sms_study.py';training=output/'source/tools/train_specialist.py'
    spec=importlib.util.spec_from_file_location('frozen_sms',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    summaries={}
    for directory in output.glob('*-adapter'):
        if (directory/'training-summary.json').is_file():summaries[directory.name]={'summary_sha256':sha(directory/'training-summary.json'),'weights_sha256':sha(directory/'adapters.safetensors')}
    plan={'planned_at_unix':time.time(),'reason':'One identical continuation after the initial validation pressure stop. No source, hyperparameter, seed, split, threshold or guard changes; preserve partial report.','prior_protocol_sha256':sha(output/'protocol.json'),'prior_journal_sha256':sha(output/'journal.json'),'completed_local_training_identities':summaries,'continuation_source_sha256':sha(Path(__file__)),'test_evaluated':False,'maximum_continuations':1}
    destination.write_text(json.dumps(plan,indent=2)+'\n');journal={}
    for cell in protocol['cells']:
        name=cell['id'];old_report=output/(name+'-valid.json')
        if old_report.is_file() and json.loads(old_report.read_text())['complete']:continue
        match=next((h for h in protocol['historical_controls'].values() if tuple(h['configuration'])==(cell['iterations'],cell['learning_rate'],cell['gradient_accumulation_steps'])),None)
        adapter=Path(match['path']) if match else output/(name+'-adapter')
        summary_path=adapter/'training-summary.json'
        if summary_path.is_file():
            summary=json.loads(summary_path.read_text())
            if (summary['iterations'],summary['learning_rate'],summary.get('gradient_accumulation_steps',1),summary['seed'],summary['adapted_layers'])!=(cell['iterations'],cell['learning_rate'],cell['gradient_accumulation_steps'],17,16):raise ValueError('Adapter configuration mismatch')
            expected=match['weight_sha256'] if match else summaries[adapter.name]['weights_sha256']
            if sha(adapter/'adapters.safetensors')!=expected:raise ValueError('Completed weights changed')
            journal[name+'-train']={'reused_completed_training':True,'adapter_sha256':expected}
        else:
            if adapter.exists():raise ValueError('Incomplete adapter cannot be resumed silently')
            command=[sys.executable,str(training),'--model',str(model),'--data',str(data),'--output',str(adapter),'--iters',str(cell['iterations']),'--seed','17','--layers','16','--learning-rate',str(cell['learning_rate']),'--gradient-accumulation',str(cell['gradient_accumulation_steps'])]
            result=module.guarded(command,output/(name+'-train.log'));journal[name+'-train']=result
            (output/'continuation-journal.json').write_text(json.dumps(journal,indent=2)+'\n')
            if result['guard_stop']:break
            if result['exit_code']!=0:continue
        report=output/(name+('-valid-retry.json' if old_report.exists() else '-valid.json'))
        if report.exists():raise ValueError('Never overwrite an evaluation report')
        command=[sys.executable,str(script),'--mode','evaluate','--model',str(model),'--data',str(data),'--adapter',str(adapter),'--split','valid','--output',str(report)]
        result=module.guarded(command,report.with_suffix('.log'));journal[name+'-valid']=result
        (output/'continuation-journal.json').write_text(json.dumps(journal,indent=2)+'\n');print(name,result['exit_code'],flush=True)
        if result['guard_stop']:break
    from run_sms_factorial_pilot import summarize
    summarize(output)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();resume(a.model.resolve(),a.data.resolve(),a.output.resolve())
