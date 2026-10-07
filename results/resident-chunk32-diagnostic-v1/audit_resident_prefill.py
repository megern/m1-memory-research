"""Characterize upstream resident BF16 chunking against a fixed teacher-forced reference.
Creates a separately labelled reference; never overwrites the original baseline.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from run_direct_original import verify_guarded


def run(model_path,manifest_path,reference_path,output,chunk_size):
    if output.exists():raise ValueError('Fresh output required')
    if not 1<=chunk_size<=128:raise ValueError('Chunk size 1..128')
    manifest=json.loads(manifest_path.read_text())
    if manifest['repository']!='Qwen/Qwen3-0.6B':raise ValueError('Resident diagnostic restricted to original 0.6B control')
    identities,verification=verify_guarded(model_path,manifest,900,True)
    if identities is None:raise RuntimeError(verification['stopped_by_guard'])
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_memory_limit(4*1024**3)
    model,tokenizer=load(str(model_path),lazy=False)
    original=json.loads((reference_path/'reference.json').read_text())
    raw=(reference_path/'logits.npz').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=original['logits_sha256']:raise ValueError('Reference changed')
    expected=np.load(reference_path/'logits.npz',allow_pickle=False)
    arrays,comparisons={},[]
    started=time.monotonic()
    for case_index,case in enumerate(original['cases']):
        ids=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        cache=make_prompt_cache(model)
        for step,token in enumerate(case['tokens']):
            if step==0:
                for start in range(0,len(ids),chunk_size):
                    end=min(start+chunk_size,len(ids))
                    hidden=model.model(mx.array([ids[start:end]]),cache=cache)
                    mx.eval(hidden,[c.state for c in cache])
                scores=model.model.embed_tokens.as_linear(hidden[:,-1:,:])[:, -1, :]
            else:scores=model(mx.array([[case['tokens'][step-1]]]),cache=cache)[:,-1,:]
            mx.eval(scores)
            actual=np.asarray(scores.astype(mx.float32)).copy()
            key=f'case{case_index}_step{step}';arrays[key]=actual
            comparisons.append({'case':case_index,'step':step,'greedy_token_equal':int(mx.argmax(scores,axis=-1).item())==token,'maximum_absolute_logit_difference':float(np.max(np.abs(actual-expected[key]))),'allclose_atol_0p01_rtol_0p01':bool(np.allclose(actual,expected[key],atol=.01,rtol=.01))})
        del cache,hidden,scores
    output.mkdir(parents=True)
    np.savez(output/'logits.npz',**arrays)
    derived={**original,'logits_sha256':hashlib.sha256((output/'logits.npz').read_bytes()).hexdigest(),'reference_kind':'upstream resident BF16 chunked prefill, final-position projection','prefill_chunk_size':chunk_size}
    (output/'reference.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2)+'\n')
    report={'reference_kind':derived['reference_kind'],'repository':manifest['repository'],'revision':manifest['revision'],'prefill_chunk_size':chunk_size,'comparison_count':len(comparisons),'greedy_equal_count':sum(c['greedy_token_equal'] for c in comparisons),'allclose_count':sum(c['allclose_atol_0p01_rtol_0p01'] for c in comparisons),'maximum_absolute_logit_difference':max(c['maximum_absolute_logit_difference'] for c in comparisons),'comparisons':comparisons,'seconds':time.monotonic()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_reference_npz_sha256':original['logits_sha256'],'derived_reference_npz_sha256':derived['logits_sha256'],'scope':'Characterization of known BF16 chunk shape differences; not performance measurement or acceptance of changed tolerance.'}
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','manifest','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=8)
    a=p.parse_args();run(a.model.resolve(),a.manifest.resolve(),a.reference.resolve(),a.output.resolve(),a.chunk_size)
