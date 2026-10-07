"""One local continuation: wait for verified acquisition, then run a frozen probe.

No scheduling service, remote inference, retry selection, or automatic publishing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil


def continue_probe(model, manifest, protocol_path, output, acquisition_pid=None):
    protocol = json.loads(protocol_path.read_text())
    pinned = json.loads(manifest.read_text())
    if (protocol['repository'],protocol['revision']) != (pinned['repository'],pinned['revision']):
        raise ValueError('Protocol and model identities differ')
    if output.exists():
        raise ValueError('Fresh continuation directory required')
    output.mkdir(parents=True)
    sources = ['original_tensor_store.py', 'direct_original_engine.py',
               'run_direct_original.py', 'wait_original_probe.py']
    source_dir = output/'source'
    source_dir.mkdir()
    hashes = {}
    for name in sources:
        raw = Path(__file__).with_name(name).read_bytes()
        (source_dir/name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    (output/'protocol.json').write_bytes(protocol_path.read_bytes())
    (output/'manifest.json').write_bytes(manifest.read_bytes())
    state = {'stage': 'waiting_for_verified_weights', 'source_sha256': hashes,
             'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
             'model_executed': False, 'completed': False}
    def save():
        state['updated_unix_seconds'] = time.time()
        temporary = output/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(output/'status.json')
    save()
    acquisition = psutil.Process(acquisition_pid) if acquisition_pid else None
    try:
        while True:
            p = model/'acquisition-state.json'
            if p.exists():
                current = json.loads(p.read_text())
                if (current['repository'],current['revision']) != (pinned['repository'],pinned['revision']):
                    raise ValueError('Acquisition identity differs')
                if current['status'] == 'verified_complete':
                    break
                if current['status'] == 'failed_resumable':
                    raise RuntimeError('Acquisition failed; partial weights retained')
            if acquisition and (not acquisition.is_running() or acquisition.status() == psutil.STATUS_ZOMBIE):
                raise RuntimeError('Acquisition process ended before verification')
            time.sleep(15)
        # Execute the frozen snapshot, even if development sources change while
        # this process is waiting. The runner independently rehashes all weights.
        for name, expected in hashes.items():
            if hashlib.sha256((source_dir/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source changed')
        state['stage'] = 'verifying_and_running_local_probe'
        state['model_executed'] = None
        save()
        import subprocess
        command = [sys.executable,str(source_dir/'run_direct_original.py'),
                   '--model',str(model),'--manifest',str(output/'manifest.json'),
                   '--output',str(output/'probe'),'--tokens',str(protocol['tokens_per_case']),
                   '--budget-mib',str(protocol['budget_mib']),
                   '--head-rows',str(protocol['head_rows']),
                   '--timeout',str(protocol['timeout_seconds'])]
        for prompt in protocol['prompts']:
            command.extend(['--prompt',prompt])
        with (output/'runner.stdout.txt').open('w') as stdout,(output/'runner.stderr.txt').open('w') as stderr:
            exit_code = subprocess.call(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL)
        report_path = output/'probe/report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        state.update(stage='completed' if exit_code == 0 else 'stopped_or_failed',
                     completed=bool(report.get('completed')), exit_code=exit_code,
                     model_executed=True if report.get('events') else None,
                     stopped_by_guard=report.get('stopped_by_guard'))
        save()
        return exit_code
    except Exception as exc:
        state.update(stage='failed',error_type=type(exc).__name__)
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acquisition-pid',type=int)
    a = p.parse_args()
    raise SystemExit(continue_probe(a.model.resolve(),a.manifest.resolve(),
                     a.protocol.resolve(),a.output.resolve(),a.acquisition_pid))
