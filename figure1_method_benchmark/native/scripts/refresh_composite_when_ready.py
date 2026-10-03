"""Refresh both real Figure 1 cohorts after the already-running launcher finishes each.

The launcher may hold an earlier renderer in memory. This watcher only revalidates and
renders completed, native cohort outputs; it never substitutes benchmark results.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

ROOT = Path('/home/nikki/SP_2026')
FIGURE = ROOT / 'figure1_method_benchmark'
SCRIPT = ROOT / 'scripts/fig1_method_benchmark.py'
PYTHON = '/home/nikki/miniconda3/envs/benchmark_env/bin/python'
STEMS = {
    'full': 'full_cortex_without_comet',
    'subsample': 'subsampled_cortex_including_comet',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--launcher-pid', required=True, type=int)
    args = parser.parse_args()
    batch = args.batch.resolve()
    if not batch.is_relative_to((FIGURE / 'native/runs').resolve()):
        raise ValueError('Expected Figure 1 native batch')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/nikki/miniconda3/envs/benchmark_env/lib' + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLCONFIGDIR'] = str(FIGURE / 'native/.cache/matplotlib')
    remaining = set(STEMS)
    while remaining:
        for name in tuple(remaining):
            manifest_path = batch / name / 'run_manifest.json'
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest['status'] == 'failed':
                print(f'{name}: native analysis failed; refresh requires repair', flush=True)
                remaining.remove(name)
                continue
            if manifest['status'] != 'complete':
                continue
            stem = 'Figure1_' + STEMS[name]
            provenance_path = FIGURE / 'figures' / STEMS[name] / (stem + '.provenance.json')
            provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            if provenance.get('layout', '').startswith('Panel A:'):
                print(f'{name}: composite already current', flush=True)
                remaining.remove(name)
                continue
            # Wait until the launcher writes its old figure; if it has stopped,
            # render directly from the validated complete cohort.
            if not provenance and Path(f'/proc/{args.launcher_pid}').exists():
                continue
            print(f'{name}: revalidating and rendering methods-plus-six composite', flush=True)
            subprocess.run([PYTHON, '-B', str(SCRIPT), '--finalize-batch', str(batch),
                            '--analysis', name], cwd=ROOT, env=env, check=True)
            new_provenance = json.loads(provenance_path.read_text())
            if not new_provenance['layout'].startswith('Panel A:'):
                raise RuntimeError(f'{name}: composite layout was not saved')
            remaining.remove(name)
        if remaining:
            time.sleep(20)
    print('Composite refresh complete for all completed cohorts', flush=True)


if __name__ == '__main__':
    main()
