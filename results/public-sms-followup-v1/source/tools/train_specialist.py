"""Bounded local MLX-LM LoRA SFT; local files, prompt masking, no telemetry."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
import types

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iters', type=int, default=160)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--gradient-accumulation', type=int, default=1)
    a = p.parse_args()
    model, data, output = a.model.resolve(), a.data.resolve(), a.output.resolve()
    if not model.is_dir() or not (model / 'config.json').is_file():
        p.error('Existing local MLX model required')
    if not 1 <= a.iters <= 500:
        p.error('Iterations must be 1..500')
    if not 1 <= a.layers <= 28:
        p.error('Adapted layers must be 1..28')
    if not 0 < a.learning_rate <= .001 or not 1 <= a.gradient_accumulation <= 16:
        p.error('Invalid learning rate or gradient accumulation')
    if output.exists():
        p.error('Choose a fresh adapter output directory')
    for split in ['train', 'valid', 'test']:
        if not (data / (split + '.jsonl')).is_file():
            p.error('All local dataset splits required')
    from provenance import identify_base, verify_base
    base_manifest=identify_base(model)
    base_hashes=verify_base(model)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
    import mlx.core as mx
    import psutil
    from mlx_lm.lora import CONFIG_DEFAULTS, run
    mx.set_memory_limit(4 * 1024**3)
    config = dict(CONFIG_DEFAULTS)
    config.update(model=str(model), data=str(data), adapter_path=str(output),
                  train=True, seed=a.seed, num_layers=a.layers, batch_size=1,
                  iters=a.iters, val_batches=12, learning_rate=a.learning_rate,
                  grad_accumulation_steps=a.gradient_accumulation,
                  steps_per_report=40, steps_per_eval=80, save_every=80,
                  max_seq_length=384, mask_prompt=True, grad_checkpoint=True,
                  report_to=None, trust_remote_code=False,
                  lora_parameters={'rank': 8, 'dropout': 0.0, 'scale': 20.0,
                                   'keys': ['self_attn.q_proj', 'self_attn.v_proj']})
    started = time.monotonic()
    swap_before=psutil.swap_memory().used
    run(types.SimpleNamespace(**config))
    weight_file=output/'adapters.safetensors'
    with weight_file.open('rb') as weights:
        header_length=struct.unpack('<Q',weights.read(8))[0]
        header=json.loads(weights.read(header_length))
    adapter_parameters=sum(math.prod(tensor['shape']) for name,tensor in header.items() if name!='__metadata__')
    summary = {'runtime': 'mlx-lm 0.32.0', 'local_compute_only': True,
               'base':base_manifest['repository'],'base_revision':base_manifest['revision'],
               'verified_base_hashes':base_hashes,
               'seed': a.seed, 'iterations': a.iters, 'batch_size': 1,
               'adapted_layers': a.layers, 'rank': 8, 'learning_rate': a.learning_rate,
               'gradient_accumulation_steps': a.gradient_accumulation,
               'prompt_masked': True, 'gradient_checkpointing': True,
               'mlx_allocation_guard_bytes': 4 * 1024**3,
               'mlx_peak_bytes': mx.get_peak_memory(),
               'rss_high_water_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'system_swap_growth_bytes':psutil.swap_memory().used-swap_before,
               'adapter_parameter_count':adapter_parameters,
               'adapter_weight_bytes':weight_file.stat().st_size,
               'elapsed_seconds': time.monotonic()-started,
               'notes': 'Allocation guard is not a hard RSS cap. Dataset claims require the associated study protocol.'}
    (output / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
