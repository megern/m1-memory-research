"""Original small-model BF16 resident references for bounded longer-prompt audits."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import psutil
from run_direct_original import verify_guarded


def run(model_path,manifest_path,protocol_path,output,chunk_size,baseline=None):
    if output.exists():raise ValueError('Fresh output required')
    manifest=json.loads(manifest_path.read_text());protocol=json.loads(protocol_path.read_text())
    if manifest['repository']!='Qwen/Qwen3-0.6B':raise ValueError('Small resident control only')
    if (protocol['repository'],protocol['revision'])!=(manifest['repository'],manifest['revision']):raise ValueError('Protocol identity differs')
    if not 1<=chunk_size<=1024:raise ValueError('Chunk size 1..1024')
    teacher=json.loads((baseline/'reference.json').read_text()) if baseline else None
    if teacher and teacher['protocol_sha256']!=hashlib.sha256(protocol_path.read_bytes()).hexdigest():raise ValueError('Baseline protocol differs')
    identities,verification=verify_guarded(model_path,manifest,900,True)
    if identities is None:raise RuntimeError(verification['stopped_by_guard'])
    output.mkdir(parents=True)
    (output/'protocol.json').write_bytes(protocol_path.read_bytes())
    samples=[]
    def guard():
        growth=psutil.swap_memory().used-verification['baseline_system_swap_bytes']
        rss=psutil.Process().memory_info().rss;available=psutil.virtual_memory().available
        samples.append({'system_swap_growth_bytes':growth,'rss_bytes':rss,'available_bytes':available})
        if growth>512*1024**2 or rss>6*1024**3 or available<512*1024**2 or time.monotonic()-verification['started_monotonic']>900:
            (output/'stopped.json').write_text(json.dumps({'completed':False,'samples':samples},indent=2)+'\n')
            raise RuntimeError('Reference pressure/time guard stopped run')
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_memory_limit(4*1024**3);mx.random.seed(17);guard()
    model,tokenizer=load(str(model_path),lazy=True)
    mx.eval(model.parameters());guard()
    expected=None
    if teacher:
        if teacher['protocol_sha256']!=hashlib.sha256(protocol_path.read_bytes()).hexdigest():raise ValueError('Baseline protocol differs')
        if hashlib.sha256((baseline/'logits.npz').read_bytes()).hexdigest()!=teacher['logits_sha256']:raise ValueError('Baseline logits changed')
        expected=np.load(baseline/'logits.npz',allow_pickle=False)
    arrays,cases,comparisons={},[],[]
    for index,prompt in enumerate(protocol['prompts']):
        ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if len(ids)>protocol['max_input_tokens']:raise ValueError('Prompt exceeds frozen limit')
        cache=make_prompt_cache(model);tokens=[]
        count=len(teacher['cases'][index]['tokens']) if teacher else protocol['reference_steps']
        for step in range(count):
            if step==0:
                for start in range(0,len(ids),chunk_size):
                    hidden=model.model(mx.array([ids[start:start+chunk_size]]),cache=cache)
                    mx.eval(hidden,[c.state for c in cache]);guard()
                scores=model.model.embed_tokens.as_linear(hidden[:,-1:,:])[:,-1,:]
            else:scores=model(mx.array([[tokens[-1]]]),cache=cache)[:,-1,:]
            mx.eval(scores);guard()
            actual=np.asarray(scores.astype(mx.float32)).copy();key=f'case{index}_step{step}';arrays[key]=actual
            greedy=int(mx.argmax(scores,axis=-1).item())
            if teacher:
                target=expected[key]
                comparisons.append({'case':index,'step':step,'greedy_token_equal':greedy==int(np.argmax(target,axis=-1).item()),'allclose_atol_0p01_rtol_0p01':bool(np.allclose(actual,target,atol=.01,rtol=.01)),'maximum_absolute_logit_difference':float(np.max(np.abs(actual-target)))})
            token=teacher['cases'][index]['tokens'][step] if teacher else greedy
            tokens.append(token)
            if not teacher and protocol['reference_stop_at_eos'] and token==tokenizer.eos_token_id:break
        cases.append({'prompt':prompt,'prompt_tokens':len(ids),'tokens':tokens,'text':tokenizer.decode(tokens,skip_special_tokens=True)})
        del cache,hidden,scores;mx.clear_cache()
    np.savez(output/'logits.npz',**arrays)
    ref={'reference_scope':'extended-context-control','reference_kind':'upstream resident BF16 bounded longer prompt, final-position projection','prefill_chunk_size':chunk_size,'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'logits_sha256':hashlib.sha256((output/'logits.npz').read_bytes()).hexdigest(),'cases':cases,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'reference.json').write_text(json.dumps(ref,indent=2)+'\n')
    report={'completed':True,'chunk_size':chunk_size,'prompt_tokens':[c['prompt_tokens'] for c in cases],'reference_steps':len(arrays),'comparison_to_unchunked':comparisons,'samples':samples,'scope':'Small resident reference; timings not used as larger-model performance evidence.'}
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'samples','comparison_to_unchunked'}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','manifest','protocol','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=1024);p.add_argument('--baseline',type=Path)
    a=p.parse_args();run(a.model.resolve(),a.manifest.resolve(),a.protocol.resolve(),a.output.resolve(),a.chunk_size,a.baseline.resolve() if a.baseline else None)
