#!/usr/bin/env python3
"""Rerun native CellSieve for Figures 1/2 at one alpha; preserve comparator fits.

Example: bash scripts/rerun_figures12.sh --alpha 0.9
Resume validated cohorts: bash scripts/rerun_figures12.sh --alpha 0.9 --batch <printed-batch-path>
Rendering alone remains available through make figure1 / make figure2-panel40.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from workspace_paths import ROOT, CELLSIEVE, local_path
from fig1_method_benchmark import sha, write_json, validate_table

NATIVE = ROOT / 'results/figure1/native'
RESULTS = ROOT / 'results/figure1/results'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--batch', type=Path, help='Resume this existing audited batch')
    parser.add_argument('--dry-run', action='store_true', help='Show planned inputs without fitting or writing')
    args = parser.parse_args(argv)
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    previous = json.loads((RESULTS / 'latest_cellsieve_batch.json').read_text())
    if args.batch:
        batch = args.batch.resolve()
        if not batch.is_relative_to((NATIVE / 'runs').resolve()):
            parser.error('--batch must be inside the Figure 1 native/runs directory')
        meta = json.loads((batch / 'rerun_manifest.json').read_text())
        if meta['alpha'] != args.alpha or meta['cellsieve_sha256'] != sha(CELLSIEVE):
            raise ValueError('Resume alpha or native package differs from batch')
    else:
        old = json.loads((local_path(previous['batch']) / 'rerun_manifest.json').read_text())
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        batch = NATIVE / 'runs' / f'figure12_cellsieve_alpha{args.alpha:.2f}_{stamp}'
        meta = dict(status='running', alpha=args.alpha, cellsieve_source=str(CELLSIEVE),
                    cellsieve_sha256=sha(CELLSIEVE), panel_sizes=[10,20,40,80,160],
                    previous_batch=previous, started_utc=stamp)
        for name in ('full', 'subsample'):
            for suffix in ('_input','_input_sha256','_time_limit_seconds'):
                meta[name + suffix] = old[name + suffix]
            meta[name + '_run_directory'] = name + '/cell_sieve'
    from importlib.metadata import version
    reference_provenance = json.loads((local_path(previous['full_run']) / 'execution_provenance.json').read_text())
    runtime_packages = {name: version(name) for name in reference_provenance['packages']}
    if runtime_packages != reference_provenance['packages']:
        raise ValueError('Use the original benchmark environment/package versions for this rerun')
    meta['python'] = sys.executable
    meta['packages'] = runtime_packages
    env = os.environ.copy()
    env.update(BENCHMARK_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMBA_CACHE_DIR=str(NATIVE / '.cache/numba'),
               MPLCONFIGDIR=str(NATIVE / '.cache/matplotlib'))
    commands = {}
    for name in ('full', 'subsample'):
        source = local_path(meta[name + '_input'])
        if sha(source) != meta[name + '_input_sha256']:
            raise ValueError('Preserved cohort changed: ' + str(source))
        commands[name] = ['bash', str(ROOT / 'scripts/native/scripts/run.sh'), '--adata-path', str(source),
            '--out-dir', str(batch / meta[name + '_run_directory']), '--methods', 'cell_sieve',
            '--panel-sizes', '10,20,40,80,160', '--max-genes', '0', '--cv-splits', '3',
            '--random-seed', '42', '--n-workers', '4', '--cell-sieve-alpha', str(args.alpha),
            '--cell-sieve-time-limit', str(meta[name + '_time_limit_seconds']),
            '--cell-sieve-mip-gap', '0.10']
    print(f'Figure 1/2 alpha={args.alpha:g}; batch: {batch}', flush=True)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        print('Then rebuild panel40, select overlap panel50, refit MERFISH, and render Figure 2.')
        return
    # Fail before expensive fits if this execution context cannot use the license.
    import gurobipy as gp
    with gp.Env(params={'OutputFlag': 0}) as license_env:
        with gp.Model(env=license_env):
            pass
    batch.mkdir(parents=True, exist_ok=True)
    meta['commands'] = commands
    write_json(batch / 'rerun_manifest.json', meta)
    status_path = RESULTS / 'figures12_rerun_status.json'
    write_json(status_path, dict(batch=str(batch), alpha=args.alpha, status='running'))

    def fit(name):
        # Recover a completed cohort even if its parent was interrupted before
        # writing the batch status. Never refit a validated saved cohort.
        import pandas as pd
        run = batch / meta[name + '_run_directory']
        completed_summary = run / 'benchmark_summary_all.csv'
        completed_provenance = run / 'execution_provenance.json'
        if completed_summary.exists() and completed_provenance.exists():
            candidate = pd.read_csv(completed_summary, dtype={'requested_panel_size': str})
            recorded = json.loads(completed_provenance.read_text())
            recorded_argv = recorded['argv']
            same_alpha = ('--cell-sieve-alpha' in recorded_argv and
                          float(recorded_argv[recorded_argv.index('--cell-sieve-alpha') + 1]) == args.alpha)
            if same_alpha and len(candidate) == 5 and candidate.status.eq('ok').all():
                comparator = local_path(json.loads((RESULTS / 'native_comparator_batch.json').read_text())['batch'])
                state = json.loads((comparator / name / 'run_manifest.json').read_text())
                validate_table(candidate, run, {**state, 'dataset': meta[name + '_input'], 'methods': ['cell_sieve']})
                return name, sha(completed_summary)
        if meta.get(name + '_status') != 'validated_5_of_5':
            # The native CLI requires a fresh cohort directory. Preserve an
            # interrupted attempt, then retry that cohort; validated cohorts skip fitting.
            target = batch / meta[name + '_run_directory']
            if target.exists():
                archived = batch / 'interrupted_attempts' / (name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
                archived.parent.mkdir(exist_ok=True)
                target.rename(archived)
            print(f'Starting {name} CellSieve; log: {batch / (name + ".log")}', flush=True)
            with (batch / (name + '.log')).open('a') as log:
                subprocess.run(commands[name], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        comparator = local_path(json.loads((RESULTS / 'native_comparator_batch.json').read_text())['batch'])
        state = json.loads((comparator / name / 'run_manifest.json').read_text())
        import pandas as pd
        run = batch / meta[name + '_run_directory']
        table = pd.read_csv(run / 'benchmark_summary_all.csv', dtype={'requested_panel_size': str})
        validate_table(table, run, {**state, 'dataset': meta[name + '_input'], 'methods': ['cell_sieve']})
        return name, sha(run / 'benchmark_summary_all.csv')

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(fit, name) for name in ('full', 'subsample')]
            for future in as_completed(futures):
                name, digest = future.result()
                meta[name + '_status'] = 'validated_5_of_5'
                meta[name + '_summary_sha256'] = digest
                write_json(batch / 'rerun_manifest.json', meta)
                print(f'Validated {name}: 5/5 panels', flush=True)
        meta['status'] = 'complete'
        write_json(batch / 'rerun_manifest.json', meta)
        write_json(RESULTS / 'latest_cellsieve_batch.json', dict(batch=str(batch),status='complete',alpha=args.alpha,
            full_run=str(batch / meta['full_run_directory']),subsample_run=str(batch / meta['subsample_run_directory'])))
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/fig1_method_benchmark.py')],cwd=ROOT,env=env,check=True)
        active = json.loads((RESULTS/'native_active_batch.json').read_text())
        active.update(cellsieve_batch=str(batch),cellsieve_alpha=args.alpha,
                      cellsieve_source=str(CELLSIEVE),cellsieve_sha256=sha(CELLSIEVE))
        write_json(RESULTS/'native_active_batch.json',active)
        # Preserve the previous Figure 2 before replacing derived tables/figures.
        backup = batch/'previous_figure2'
        if not backup.exists():
            backup.mkdir()
            for name in ('biology_panel40_fig1','panel50','merfish_panel50_coefficients'):
                shutil.copytree(ROOT/'results/figure2/results'/name,backup/name)
            shutil.copytree(ROOT/'results/figure2/exports',backup/'figures')
        for script in ('build_fig2_panel40.py','build_fig2_violins.py',
                       'reselect_fig2_panel50.py','refit_fig2_merfish.py','fig2_cortex_merfish.py',
                       'finalize_figures12.py'):
            print(f'Running {script}',flush=True)
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,check=True)
        write_json(status_path,dict(batch=str(batch),alpha=args.alpha,status='complete'))
        print('Figures 1 and 2 completed and validated.',flush=True)
    except Exception as exc:
        write_json(status_path,dict(batch=str(batch),alpha=args.alpha,status='failed',error=str(exc)))
        raise


if __name__ == '__main__':
    main()
