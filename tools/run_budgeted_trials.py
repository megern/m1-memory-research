"""Guarded reference and trials for the exploratory bounded-cache engine."""
import argparse
import os
from pathlib import Path

from run_original_precision import run

PROTOCOL = Path(__file__).resolve().parents[1]/'models/budgeted-engine-protocol.json'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--reference', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--experiment', default='budgeted-reference')
    a = p.parse_args()
    model = a.model.resolve()
    if a.worker:
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        mode = a.experiment.removeprefix('budgeted-')
        if mode not in ('reference', 'resident', 'file', 'reuse', 'cache'):
            p.error('Unknown bounded engine mode')
        from benchmark_original_scheduler import worker
        worker(model, mode, protocol_path=PROTOCOL, reference_suffix='budgeted-reference', engine_budget=640)
    elif not a.output:
        p.error('--output is required')
    elif a.reference:
        run(model, a.output.resolve(), 'budgeted-reference', worker_script=Path(__file__).resolve())
    else:
        from run_scheduler_trials import main
        main(model, a.output.resolve(), protocol_path=PROTOCOL,
            worker_script=Path(__file__).resolve(), experiment_prefix='budgeted-',
            reference_suffix='budgeted-reference',
            extra_sources=[Path(__file__), Path(__file__).with_name('budgeted_original_engine.py')])
