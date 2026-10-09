"""Execute the frozen 16-cell confirmation locally, with auditable historical controls."""
import argparse,hashlib,importlib.util,json,random,shutil,sys,time
from pathlib import Path
from public_sms_study import verify_original_base

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(output):
 protocol=json.loads((output/'protocol.json').read_text());reports={};effects={}
 for cell in protocol['cells']:
  path=output/(cell['id']+'-valid.json')
  if path.is_file():
   report=json.loads(path.read_text())
   retry=output/(cell['id']+'-valid-retry.json')
   if not report['complete'] and retry.is_file():report=json.loads(retry.read_text())
   if report['complete']:reports[cell['id']]={'configuration':cell,'metrics':report['metrics']}
 for seed in (23,41):
  rows=[r for r in reports.values() if r['configuration']['seed']==seed]
  if len(rows)==8:
   effects[str(seed)]={}
   for factor,levels in [('iterations',(120,480)),('learning_rate',(.0001,.0002)),('gradient_accumulation_steps',(1,4))]:
    means=[sum(r['metrics']['macro_f1'] for r in rows if r['configuration'][factor]==level)/4 for level in levels]
    effects[str(seed)][factor]={'level_means':means,'second_minus_first':means[1]-means[0]}
 result={'planned_cells':16,'completed_validation_cells':len(reports),'cells':reports,'per_seed_factor_effects':effects,'test_evaluated':False,'best_model_selected':False,'scope':protocol['limitations'],'source_sha256':sha(Path(__file__))}
 (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':len(reports),'planned':16,'effects':effects}),flush=True)

def run(model,data,output,plan_path,old,new,resume=False):
 root=Path(__file__).resolve().parents[1]
 if resume:
  protocol=json.loads((output/'protocol.json').read_text());journal=json.loads((output/'journal.json').read_text())
  if (output/'continuation-protocol.json').exists():raise ValueError('One explicit continuation only')
  if not any(r.get('guard_stop') for r in journal.values()):raise ValueError('Continuation requires recorded pressure stop')
  if str(model)!=protocol['model'] or str(data)!=protocol['data']:raise ValueError('Identity changed')
  for relative,expected in protocol['frozen_sources'].items():
   if sha(output/'source'/relative)!=expected:raise ValueError('Frozen source changed')
  verify_original_base(model)
  for split,expected in protocol['splits_sha256'].items():
   if sha(data/(split+'.jsonl'))!=expected:raise ValueError('Dataset changed')
  (output/'continuation-protocol.json').write_text(json.dumps({'created_at_unix':time.time(),'rule':'One unchanged continuation after guard stop; preserve partial reports, no favorable replacement of completed outcomes','initial_journal_sha256':sha(output/'journal.json')},indent=2)+'\n')
 else:
  if output.exists():raise ValueError('Fresh output required')
  plan=json.loads(plan_path.read_text())
  if plan['status']!='planned_not_executed' or len(plan['cells'])!=16 or plan['split']!='valid' or plan['test_evaluation']:raise ValueError('Wrong frozen confirmation design')
  verified=verify_original_base(model);dataset=json.loads((data/'protocol.json').read_text())
  for split,meta in dataset['splits'].items():
   if sha(data/(split+'.jsonl'))!=meta['sha256']:raise ValueError('Dataset changed')
  controls={}
  for seed in (23,41):
   for directory,settings in ((old,(120,.0002,1)),(new,(480,.0001,4))):
    adapter=directory/f'adapter-{seed}';summary=json.loads((adapter/'training-summary.json').read_text());parent=json.loads((directory/'protocol.json').read_text())
    if (summary['iterations'],summary['learning_rate'],summary.get('gradient_accumulation_steps',1))!=settings or summary['seed']!=seed or summary['adapted_layers']!=16:raise ValueError('Historical settings mismatch')
    if any(parent['splits'][split]['sha256']!=dataset['splits'][split]['sha256'] for split in ('train','valid')):raise ValueError('Historical data mismatch')
    match=next(c for c in plan['cells'] if c['seed']==seed and (c['iterations'],c['learning_rate'],c['gradient_accumulation_steps'])==settings)
    controls[match['id']]={'path':str(adapter),'sha256':sha(adapter/'adapters.safetensors')}
  output.mkdir(parents=True);source=output/'source';(source/'tools').mkdir(parents=True);(source/'models').mkdir()
  for name in ('run_sms_confirmation.py','plan_sms_confirmation.py','public_sms_study.py','train_specialist.py','provenance.py'):shutil.copy2(root/'tools'/name,source/'tools'/name)
  shutil.copy2(root/'models/qwen3-0.6b-original.json',source/'models/qwen3-0.6b-original.json');shutil.copy2(plan_path,source/'original-plan.json')
  cells=list(plan['cells']);random.Random(20261010).shuffle(cells)
  protocol={**plan,'status':'execution_protocol_frozen','planned_at_unix':time.time(),'model':str(model),'data':str(data),'cells':cells,'random_order_seed':20261010,'historical_controls':controls,'verified_base':verified,'splits_sha256':{k:v['sha256'] for k,v in dataset['splits'].items()},'plan_sha256':sha(plan_path),'frozen_sources':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},'historical_reuse':'Four prior quality controls reused; their timings are not compared to newly trained adapters'}
  (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');journal={}
 spec=importlib.util.spec_from_file_location('frozen_confirmation_guard',output/'source/tools/public_sms_study.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
 for cell in protocol['cells']:
  name=cell['id'];valid=output/(name+'-valid.json');train_key=name+'-train';valid_key=name+'-valid'
  if valid.exists() and json.loads(valid.read_text())['complete']:continue
  if name in protocol['historical_controls']:
   control=protocol['historical_controls'][name];adapter=Path(control['path'])
   if sha(adapter/'adapters.safetensors')!=control['sha256']:raise ValueError('Historical weights changed')
   journal[train_key]={'reused_historical_control':True,'adapter_sha256':control['sha256']}
  else:
   adapter=output/(name+'-adapter')
   if train_key in journal and journal[train_key]['exit_code']==0:
    if sha(adapter/'adapters.safetensors')!=journal[train_key]['adapter_sha256']:raise ValueError('Completed weights changed')
   else:
    if adapter.exists():adapter=output/(name+'-retry-adapter')
    if adapter.exists():raise ValueError('Never overwrite training attempts')
    command=[sys.executable,str(output/'source/tools/train_specialist.py'),'--model',str(model),'--data',str(data),'--output',str(adapter),'--iters',str(cell['iterations']),'--seed',str(cell['seed']),'--layers','16','--learning-rate',str(cell['learning_rate']),'--gradient-accumulation',str(cell['gradient_accumulation_steps'])]
    attempt=train_key+('-retry' if train_key in journal else '')
    outcome=guard.guarded(command,output/(attempt+'.log'));journal[attempt]=outcome
    if outcome['exit_code']==0:
     outcome['adapter_sha256']=sha(adapter/'adapters.safetensors');outcome['adapter_path']=str(adapter)
    (output/'journal.json').write_text(json.dumps(journal,indent=2)+'\n');print(name,'train',outcome['exit_code'],flush=True)
    if outcome['guard_stop'] or outcome['exit_code']!=0:break
   # A completed retry can have a separate immutable adapter directory.
   if train_key+'-retry' in journal and journal[train_key+'-retry']['exit_code']==0:adapter=Path(journal[train_key+'-retry']['adapter_path'])
  if valid.exists():valid=output/(name+'-valid-retry.json');valid_key+='-retry'
  if valid.exists():raise ValueError('Never overwrite evaluation attempts')
  command=[sys.executable,str(output/'source/tools/public_sms_study.py'),'--mode','evaluate','--model',str(model),'--data',str(data),'--adapter',str(adapter),'--split','valid','--output',str(valid)]
  journal[valid_key]=guard.guarded(command,output/(valid_key+'.log'))
  (output/'journal.json').write_text(json.dumps(journal,indent=2)+'\n');print(name,'valid',journal[valid_key]['exit_code'],flush=True)
  summarize(output)
  if journal[valid_key]['guard_stop'] or journal[valid_key]['exit_code']!=0:break
 summarize(output)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('model','data','output','plan','old','new'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--resume',action='store_true');a=p.parse_args();run(*(getattr(a,k).resolve() for k in ('model','data','output','plan','old','new')),a.resume)
