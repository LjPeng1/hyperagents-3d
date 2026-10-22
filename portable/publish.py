"""Publish audit sources and execution requests without raw logs or account paths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from portable.experiment import ROOT, read, write


def publish(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    source.relative_to(ROOT / 'workspace')
    destination.relative_to(ROOT / 'examples')
    if destination.exists():
        raise RuntimeError('Publication directory exists; refusing overwrite')
    report = read(source / 'report.json')
    destination.mkdir(parents=True)

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            for prefix, replacement in ((str(source), '.'), (str(ROOT.parent / 'dream-rsi-3d'), '<DREAM>'),
                                        (str(ROOT), '<HYPER>'), (str(Path.home()), '<HOME>')):
                value = value.replace(prefix, replacement)
                value = value.replace(prefix.replace('\\', '/'), replacement)
            return value
        return value

    # Preserve complete generated solver bytes and the selected version sources.
    shutil.copytree(source / 'candidates', destination / 'candidates')
    for path in (destination / 'candidates').glob('*/candidate.json'):
        write(path, clean(read(path)))
    for arm in ('hyper_adaptive', 'hyper_fixed'):
        for cycle in (1, 2):
            call = source / arm / 'calls' / f'update{cycle}'
            data = read(call / 'result.json')
            target = destination / 'agents' / arm / f'g{cycle}'
            target.mkdir(parents=True)
            for role, code in data['bundle'].items():
                (target / f'{role}_agent.py').write_text(code, encoding='utf-8')
            write(target / 'lineage.json', {k: v for k, v in data.items() if k != 'bundle'})
            write(target / 'executed_request.json', clean(read(call / 'executed_request.json')))
    for cycle in (0, 1):
        source_job = source / 'dream' / 'learning' / f'iter{cycle:03d}'
        for path in source_job.rglob('method.py'):
            if 'trace_pool' not in path.parts and 'trace' not in path.parts:
                target = destination / 'dream_policies' / f'update{cycle+1}' / path.relative_to(source_job)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    # Public per-episode accounting omits command lines, stdout and thread ids.
    report['usage']['runs'] = [{k: row.get(k) for k in ('arm', 'returncode', 'timed_out', 'error', 'duration_sec', 'usage')}
                                for row in report['usage']['runs']]
    amendments = source / 'amendments.json'
    if amendments.exists():
        report['engineering_amendments'] = clean(read(amendments))
        write(destination / 'amendments.json', clean(read(amendments)))
        diagnostic = source / 'diagnostics' / 'zero-call-contract'
        for name in ('original_protocol.json', 'journal.json', 'worker_verification.json'):
            write(destination / 'diagnostics' / name, clean(read(diagnostic / name)))
    report['task_invocations'] = {}
    for arm in ('hyper_adaptive', 'hyper_fixed'):
        completed = [read(p) for p in (source / arm / 'calls').glob('online*/completed.json')]
        report['task_invocations'][arm] = {'completed': len(completed),
            'zero_model_calls': sum(c.get('model_episodes') == 0 for c in completed),
            'with_model_call': sum(c.get('model_episodes') == 1 for c in completed)}
    report['render_verification'] = {}
    for name in ('e0', 'dream'):
        render = source / ('render-' + name)
        if render.exists():
            target = destination / 'renders' / name
            target.mkdir(parents=True)
            for filename in ('layout.png', 'layout.glb', 'reimport_check.json'):
                shutil.copyfile(render / filename, target / filename)
            report['render_verification'][name] = read(render / 'reimport_check.json')
            provenance = render / 'source_provenance.json'
            if provenance.exists():
                write(target / provenance.name, clean(read(provenance)))
                report['render_verification'][name]['source'] = clean(read(provenance))
    write(destination / 'report.json', clean(report))
    write(destination / 'protocol.json', clean(read(source / 'protocol.json')))
    write(destination / 'initial_bundle.json', read(source / 'initial_bundle.json'))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.source, args.destination))
