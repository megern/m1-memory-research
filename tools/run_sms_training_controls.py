"""Run prospectively planned old-adapter controls after the new regime completes."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path

def run(model,data,results,controls):
    journal=json.loads((results/'journal.json').read_text());plan=json.loads((results/'paired-training-control-protocol.json').read_text())
    if not all(journal.get(f'seed-{s}-evaluate',{}).get('exit_code')==0 for s in (17,23,41)):raise ValueError('Finish all new-regime GPU jobs first')
    script=results/'source/tools/public_sms_study.py'
    if hashlib.sha256(script.read_bytes()).hexdigest()!=plan['frozen_evaluator_sha256']:raise ValueError('Frozen evaluator changed')
    if hashlib.sha256((data/'test.jsonl').read_bytes()).hexdigest()!=plan['test_sha256']:raise ValueError('Test changed')
    spec=importlib.util.spec_from_file_location('frozen_sms',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);outcomes={}
    for seed in (17,23,41):
        adapter=controls/f'adapter-{seed}';output=results/f'control-seed-{seed}.json'
        if output.exists():raise ValueError('Never overwrite a previous control outcome')
        if hashlib.sha256((adapter/'adapters.safetensors').read_bytes()).hexdigest()!=plan['control_adapter_sha256'][str(seed)]:raise ValueError('Control weights changed')
        cmd=[sys.executable,str(script),'--mode','evaluate','--model',str(model.resolve()),'--data',str(data.resolve()),'--adapter',str(adapter.resolve()),'--output',str(output.resolve())]
        outcomes[str(seed)]=module.guarded(cmd,results/f'control-seed-{seed}.log')
        (results/'paired-training-control-outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n');print(json.dumps({seed:outcomes[str(seed)]}),flush=True)
        if outcomes[str(seed)]['guard_stop']:break
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','results','controls'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.data,a.results,a.controls)
