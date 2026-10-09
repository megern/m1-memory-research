"""Validation-only 2x2x2 local training pilot; no test scoring or best-model selection."""
import argparse,hashlib,itertools,json,random,shutil,sys,time
from pathlib import Path
from public_sms_study import guarded,metrics,verify_original_base

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def cells():
    result=[{'iterations':steps,'learning_rate':lr,'gradient_accumulation_steps':acc,'seed':17} for steps,lr,acc in itertools.product((120,480),(.0001,.0002),(1,4))]
    for row in result:row.update(id=f"steps{row['iterations']}-lr{row['learning_rate']:.4f}-acc{row['gradient_accumulation_steps']}",optimizer_updates=row['iterations']//row['gradient_accumulation_steps'])
    random.Random(20261009).shuffle(result)
    return result

def summarize(output):
    protocol=json.loads((output/'protocol.json').read_text());journal=json.loads((output/'journal.json').read_text());values={}
    for cell in protocol['cells']:
        path=output/(cell['id']+'-valid.json')
        if path.is_file():
            report=json.loads(path.read_text())
            if report['complete']:values[cell['id']]={'configuration':cell,'metrics':report['metrics']}
    effects={}
    if len(values)==8:
        for factor,levels in [('iterations',(120,480)),('learning_rate',(.0001,.0002)),('gradient_accumulation_steps',(1,4))]:
            means={str(level):sum(v['metrics']['macro_f1'] for v in values.values() if v['configuration'][factor]==level)/4 for level in levels}
            effects[factor]={'level_means_macro_f1':means,'second_minus_first':means[str(levels[1])]-means[str(levels[0])]}
    summary={'completed_validation_cells':len(values),'planned_cells':8,'seed':17,'validation_cases':128,'validation_spam_cases':13,'cells':values,'descriptive_factor_effects':effects,'test_evaluated':False,'best_model_selected':False,'scope':'Exploratory single-seed validation pilot, correlated cases. No population seed inference, no novel algorithm, no unseen test/generalization improvement claim. Changing accumulation changes optimizer-update count at fixed microbatches. Two prior adapters are reused as historical quality controls; no timing/resource comparison across cells.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

def run(model,data,output,old,new,execute):
    if output.exists():raise ValueError('Fresh output required')
    original=verify_original_base(model);dataset=json.loads((data/'protocol.json').read_text())
    for split,meta in dataset['splits'].items():
        if sha(data/(split+'.jsonl'))!=meta['sha256']:raise ValueError('Dataset changed')
    historical={}
    for label,directory,expected in [('old',old,(120,.0002,1)),('new',new,(480,.0001,4))]:
        parent=json.loads((directory/'protocol.json').read_text());adapter=directory/'adapter-17';summary=json.loads((adapter/'training-summary.json').read_text())
        if any(parent['splits'][s]['sha256']!=dataset['splits'][s]['sha256'] for s in ('train','valid')):raise ValueError('Historical training/validation mismatch')
        actual=(summary['iterations'],summary['learning_rate'],summary.get('gradient_accumulation_steps',1))
        if actual!=expected or summary['seed']!=17 or summary['adapted_layers']!=16:raise ValueError('Historical configuration mismatch')
        historical[label]={'configuration':expected,'path':str(adapter.resolve()),'weight_sha256':sha(adapter/'adapters.safetensors')}
    output.mkdir(parents=True);source=output/'source';(source/'tools').mkdir(parents=True);(source/'models').mkdir()
    root=Path(__file__).resolve().parents[1]
    for name in ('run_sms_factorial_pilot.py','public_sms_study.py','train_specialist.py','provenance.py'):shutil.copy2(root/'tools'/name,source/'tools'/name)
    shutil.copy2(root/'models/qwen3-0.6b-original.json',source/'models/qwen3-0.6b-original.json')
    protocol={'planned_at_unix':time.time(),'client_date':'2026-10-09','question':'Which exposure, learning-rate and accumulation factors explain validation behavior? Pilot feasibility before multi-seed confirmation and independent held-out corpus.','design':'Balanced complete 2x2x2 factorial, one fixed seed (smallest prior seed), randomized job order; validation only.','cells':cells(),'historical_controls':historical,'data_protocol_sha256':sha(data/'protocol.json'),'splits_sha256':{k:v['sha256'] for k,v in dataset['splits'].items()},'verified_base_and_tokenizer':original,'frozen_sources':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},'guards':{'process_tree_rss_gib':6,'whole_system_swap_growth_mib':1536,'available_memory_mib':512,'low_available_grace_seconds':5,'child_timeout_seconds':900,'mlx_allocation_gib':4},'local_compute_only':True,'test_evaluated':False,'decision_rule':'No checkpoint/threshold/seed selection or scoring the exposed test. All complete and stopped cells retained. Validation conclusions are exploratory; future confirmation needs multiple seeds and a separate source/domain holdout.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if not execute:print('Protocol frozen; no training launched.');return
    journal={};evaluation=source/'tools/public_sms_study.py';training=source/'tools/train_specialist.py'
    for cell in protocol['cells']:
        name=cell['id'];match=next((h for h in historical.values() if tuple(h['configuration'])==(cell['iterations'],cell['learning_rate'],cell['gradient_accumulation_steps'])),None)
        if match:
            adapter=Path(match['path'])
            if sha(adapter/'adapters.safetensors')!=match['weight_sha256']:raise ValueError('Historical weights changed')
            journal[name+'-train']={'reused_historical_control':True,'adapter_sha256':match['weight_sha256']}
        else:
            adapter=output/(name+'-adapter')
            command=[sys.executable,str(training),'--model',str(model),'--data',str(data),'--output',str(adapter),'--iters',str(cell['iterations']),'--seed','17','--layers','16','--learning-rate',str(cell['learning_rate']),'--gradient-accumulation',str(cell['gradient_accumulation_steps'])]
            result=guarded(command,output/(name+'-train.log'));journal[name+'-train']=result
            (output/'journal.json').write_text(json.dumps(journal,indent=2)+'\n')
            if result['guard_stop']:break
            if result['exit_code']!=0:continue
        command=[sys.executable,str(evaluation),'--mode','evaluate','--model',str(model),'--data',str(data),'--adapter',str(adapter),'--split','valid','--output',str(output/(name+'-valid.json'))]
        result=guarded(command,output/(name+'-valid.log'));journal[name+'-valid']=result
        (output/'journal.json').write_text(json.dumps(journal,indent=2)+'\n');print(name,result['exit_code'],flush=True)
        if result['guard_stop']:break
    summarize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','output','old','new'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args();run(*(getattr(a,k).resolve() for k in ('model','data','output','old','new')),a.execute)
