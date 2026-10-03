"""Run the real cortex native benchmark and render one Figure 1 plus its supplement.

Default: render the two validated native cohorts selected by their saved run pointers.
--run-native: execute the real dataset, validate per-method outputs, then render.
Two analyses: full cortex without COMET, and a shared stratified subset including COMET.
No synthetic input, unverified proxy, or partial-run plotting is supported.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess
import sys
import pandas as pd
from figure_style import ROOT, plt, set_style, save, METHODS as METHOD_STYLE
from workspace_paths import CELLSIEVE, local_path

NATIVE = ROOT / 'figure1_method_benchmark/native'
FIGURE = ROOT / 'figure1_method_benchmark'
METHODS_PDF = FIGURE / 'CellSieve_ Combinatorially compressed gene sets from scRNAseq data - Google Slides.pdf'
METHODS_IMAGE = FIGURE / 'figures/methods_layout_source.png'
DATA = ROOT / 'h5ads/B6J_CortexHippocampus_subset.h5ad'
METHODS = ('cell_sieve', 'nsforest', 'scgenefit',
           'logistic_regression_de', 'l1_regularization', 'comet')
PANELS = (10, 20, 40, 80, 160)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def validate_table(table, run, manifest):
    """Require complete real-data rows and matching saved panels/CV metrics."""
    methods = manifest["methods"]
    data = local_path(manifest["dataset"])
    if set(table.method) != set(methods):
        raise ValueError('Native run must account for every original method')
    if set(table.dataset) != {data.stem}:
        raise ValueError('Figure 1 accepts only the real cortex dataset')
    if any(local_path(p) != data.resolve() for p in table.adata_path):
        raise ValueError('Foreign or synthetic dataset rejected')
    if set(table.evaluation_mode) != {'full_train_full_eval_plus_3fold_cv'}:
        raise ValueError('The original evaluation design must be preserved')
    for method in methods:
        rows = table[table.method == method]
        expected = {str(k) for k in PANELS}
        if len(rows) != len(expected) or set(rows.requested_panel_size.astype(str)) != expected:
            raise ValueError('Incomplete or duplicated panel grid: ' + method)
        for row in rows.itertuples():
            output = Path(row.output_dir).resolve()
            if not output.is_relative_to(run.resolve()):
                raise ValueError('Cannot merge outputs from another run')
            if row.status != 'ok':
                raise ValueError(f'{method} failed; see {output}/error.txt')
            if not float(row.elapsed_seconds) > 0 or not float(row.peak_rss_mb) > 0:
                raise ValueError('Missing measured runtime or peak memory')
            report = json.loads((output / 'classification_report.json').read_text())
            if abs(float(row.accuracy) - report['accuracy']) > 1e-12 or abs(float(row.macro_f1) - report['macro avg']['f1-score']) > 1e-12:
                raise ValueError('Saved refit metrics do not match summary')
            genes = (output / 'selected_genes.txt').read_text().splitlines()
            if not genes or len(genes) != len(set(genes)) or len(genes) != int(row.actual_panel_size):
                raise ValueError('Invalid native panel: ' + str(output))
            if len(genes) != int(row.requested_panel_size):
                raise ValueError('Requested panel size was not produced')
            metrics = json.loads((output / 'cv_metrics.json').read_text())
            for key in ('cv_accuracy', 'cv_macro_f1'):
                value = float(getattr(row, key))
                if not 0 <= value <= 1 or abs(value - metrics[key]) > 1e-12:
                    raise ValueError('Saved CV metrics do not match ' + str(output))
            if method in ('nsforest', 'comet'):
                ranking = pd.read_csv(output.parent / 'native_rankings/native_global_ranking.csv')
                if genes != ranking.gene.astype(str).tolist()[:int(row.requested_panel_size)]:
                    raise ValueError('Panel does not match saved native output ranking')
    return table


def preflight():
    if NATIVE.is_symlink() or not NATIVE.is_dir():
        raise RuntimeError('Figure 1 requires its own physical native runtime directory')
    for relative in ['scripts/run.sh', 'scripts/benchmark_execution.py', 'scripts/panel_adapters.py',
                     'scripts/run_comet_native.py', 'vendor_manifest.json', '.venv_comet/bin/python']:
        if not (NATIVE / relative).exists(): raise FileNotFoundError(NATIVE / relative)
    for path in [DATA, CELLSIEVE, METHODS_PDF, METHODS_IMAGE]:
        if not path.is_file(): raise FileNotFoundError(path)
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLCONFIGDIR'] = str(NATIVE / '.cache/matplotlib')
    subprocess.run([str(NATIVE / '.venv_comet/bin/python'), '-B', '-c',
        'import xlmhg.mhg_cython, numpy, pandas, scipy; print("Compiled native COMET dependencies available")'], env=env, check=True)
    set_style()
    print('Figure 1 preflight passed: physical runtime, original inputs, compiled COMET, fonts', flush=True)


def execute_native(workers):
    preflight()
    import scanpy as sc
    from sklearn.model_selection import train_test_split
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    batch = NATIVE / 'runs' / ('figure1_two_analyses_' + stamp)
    batch.mkdir(parents=True, exist_ok=False)
    data = sc.read_h5ad(DATA)
    if data.shape != (86427, 5083):
        raise ValueError('Unexpected real cortex dataset')
    subset_indices, _ = train_test_split(list(range(data.n_obs)), train_size=2500,
        stratify=data.obs['celltype'].astype(str), random_state=42)
    subset_indices = sorted(subset_indices)
    subset = data[subset_indices].copy()
    if set(subset.obs.celltype.astype(str)) != set(data.obs.celltype.astype(str)):
        raise ValueError('Stratification omitted a cortex class')
    subset_path = batch / 'cortex_stratified_2500.h5ad'
    subset.write_h5ad(subset_path)
    pd.DataFrame({'source_row': subset_indices, 'cell_id': subset.obs_names,
                  'celltype': subset.obs.celltype.astype(str).to_numpy()}).to_csv(batch / 'subset_cells.csv', index=False)
    counts = pd.DataFrame({'full': data.obs.celltype.astype(str).value_counts(),
                           'subset': subset.obs.celltype.astype(str).value_counts()}).fillna(0).astype(int)
    counts.to_csv(batch / 'cohort_class_counts.csv')
    original_hash = sha(DATA)
    del subset, data
    specs = [('full', DATA, 86427, tuple(m for m in METHODS if m != 'comet')),
             ('subsample', subset_path, 2500, METHODS)]
    states = {}; runs = {}
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(workers),
               OPENBLAS_NUM_THREADS=str(workers), MKL_NUM_THREADS=str(workers),
               NUMBA_NUM_THREADS=str(workers), PYTHONUNBUFFERED='1')
    for name, source, cells, methods in specs:
        run = batch / name; run.mkdir(); runs[name] = run
        state = {'analysis': name, 'status': 'running', 'started_utc': stamp,
            'dataset': str(source), 'input_sha256': sha(source), 'input_shape': [cells, 5083],
            'original_dataset': str(DATA), 'original_sha256': original_hash,
            'subset_cells': str(batch / 'subset_cells.csv') if name == 'subsample' else None,
            'panel_sizes': list(PANELS), 'methods': list(methods),
            'evaluation': 'Full cohort selection and conditional three-fold classifier CV, separately within each analysis',
            'workers_per_method': workers, 'parallel_methods': 3,
            'cellsieve_package_source': str(CELLSIEVE),
            'cellsieve_package_sha256': sha(str(CELLSIEVE)),
            'completed_methods': [], 'failed_methods': [], 'commands': {},
            'source_hashes': {str(p): sha(p) for p in [NATIVE / 'scripts/benchmark_execution.py',
                NATIVE / 'scripts/panel_adapters.py', CELLSIEVE, Path(__file__).resolve()]}}
        for method in methods:
            state['commands'][method] = ['bash', str(NATIVE / 'scripts/run.sh'),
                '--adata-path', str(source), '--out-dir', str(run / method), '--methods', method,
                '--panel-sizes', ','.join(map(str, PANELS)), '--max-genes', '0', '--cv-splits', '3',
                '--random-seed', '42', '--n-workers', str(workers),
                '--cell-sieve-script', str(CELLSIEVE), '--cell-sieve-alpha', '0.9',
                '--cell-sieve-time-limit', '300', '--cell-sieve-mip-gap', '0.10']
        states[name] = state
        write_json(run / 'run_manifest.json', state)
    write_json(FIGURE / 'results/native_active_batch.json', {'batch': str(batch), 'status': 'running'})
    print('Two real cortex analyses: ' + str(batch), flush=True)

    def worker(name, method):
        with (runs[name] / (method + '.log')).open('w') as log:
            result = subprocess.run(states[name]['commands'][method], cwd=ROOT,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        return name, method, result.returncode

    def finish(name):
        run = runs[name]; state = states[name]
        try:
            if state['failed_methods']:
                raise RuntimeError('Native failures: ' + ', '.join(state['failed_methods']))
            frames = []
            for method in state['methods']:
                provenance = json.loads((run / method / 'execution_provenance.json').read_text())
                if not provenance.get('native_only') or not provenance.get('evaluation_unchanged'):
                    raise ValueError('Missing native execution provenance: ' + method)
                frames.append(pd.read_csv(run / method / 'benchmark_summary_all.csv', dtype={'requested_panel_size': str}))
            table = validate_table(pd.concat(frames, ignore_index=True), run, state)
            summary = run / 'benchmark_summary_all.csv'; table.to_csv(summary, index=False)
            state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                         summary_sha256=sha(summary), successful_rows=len(table))
            write_json(run / 'run_manifest.json', state)
            write_json(FIGURE / f'results/native_{name}_run.json', {
                'run_directory': str(run.relative_to(ROOT)), 'summary_sha256': sha(summary),
                'input_sha256': state['input_sha256']})
        except Exception as exc:
            state.update(status='failed', error=str(exc))
            write_json(run / 'run_manifest.json', state)
            print(f'{name} failed: {exc}', flush=True)

    # Long native jobs begin first; all jobs use real cells from their cohort.
    jobs = [('subsample', 'comet'), ('full', 'nsforest'), ('full', 'cell_sieve')]
    jobs += [('subsample', m) for m in METHODS if ('subsample', m) not in jobs]
    jobs += [('full', m) for m in states['full']['methods'] if ('full', m) not in jobs]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, name, method) for name, method in jobs]
        for job in as_completed(futures):
            name, method, code = job.result(); state = states[name]
            state['completed_methods'].append({'method': method, 'exit_code': code})
            if code != 0: state['failed_methods'].append(method)
            write_json(runs[name] / 'run_manifest.json', state)
            print(f'{name}/{method}: exit {code}; log: {runs[name] / (method + ".log")}', flush=True)
            if len(state['completed_methods']) == len(state['methods']): finish(name)
    success = all(s['status'] == 'complete' for s in states.values())
    write_json(FIGURE / 'results/native_active_batch.json', {'batch': str(batch), 'status': 'complete' if success else 'failed'})
    if not success: raise RuntimeError('At least one native analysis failed; see run manifests')
    render_publication_figures(load_native('full'), load_native('subsample'))


def load_native(name):
    pointer_path = FIGURE / f'results/native_{name}_run.json'
    if not pointer_path.exists():
        raise RuntimeError(f'No completed real {name} native cortex run. Use --run-native first.')
    pointer = json.loads(pointer_path.read_text())
    run = (ROOT / pointer['run_directory']).resolve()
    if not run.is_relative_to((NATIVE / 'runs').resolve()): raise ValueError('Unexpected native run location')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    summary = run / 'benchmark_summary_all.csv'
    expected_cells = 86427 if name == 'full' else 2500
    if manifest['status'] != 'complete' or manifest['input_shape'] != [expected_cells, 5083]:
        raise ValueError('Only completed real cortex runs can be plotted')
    if local_path(manifest['original_dataset']) != DATA.resolve(): raise ValueError('Foreign dataset rejected')
    if sha(summary) != pointer['summary_sha256'] or pointer['input_sha256'] != manifest['input_sha256']:
        raise ValueError('Native run provenance mismatch')
    return validate_table(pd.read_csv(summary, dtype={'requested_panel_size': str}), run, manifest), run, manifest


def load_latest_cellsieve(name, reference_state):
    """Load and validate five measured panels from the sole authorized package."""
    pointer = json.loads((FIGURE / 'results/latest_cellsieve_batch.json').read_text())
    if pointer.get('status') != 'complete':
        raise RuntimeError('Latest-package CellSieve runs are not yet complete')
    batch = Path(pointer['batch']).resolve()
    if not batch.is_relative_to((NATIVE / 'runs').resolve()):
        raise ValueError('Unexpected latest CellSieve batch location')
    source = CELLSIEVE.resolve()
    meta = json.loads((batch / 'rerun_manifest.json').read_text())
    if local_path(meta['cellsieve_source']) != source or meta['cellsieve_sha256'] != sha(source):
        raise ValueError('Current CellSieve package differs from the executed source')
    if (local_path(meta[name + '_input']) != local_path(reference_state['dataset']) or
            meta[name + '_input_sha256'] != reference_state['input_sha256']):
        raise ValueError('Latest CellSieve run used a different real cohort input')
    run = batch / meta.get(name + '_run_directory', name + '/cell_sieve')
    if not run.resolve().is_relative_to(batch):
        raise ValueError('Latest CellSieve run leaves its audited batch')
    provenance = json.loads((run / 'execution_provenance.json').read_text())
    if (local_path(provenance['cellsieve_script']) != source or
            provenance['cellsieve_script_sha256'] != sha(source) or
            not provenance.get('native_only') or not provenance.get('evaluation_unchanged')):
        raise ValueError('Latest CellSieve run lacks matching native package provenance')
    argv = provenance['argv']
    expected_limit = str(meta[name + '_time_limit_seconds'])
    if '--cell-sieve-time-limit' not in argv or float(argv[argv.index('--cell-sieve-time-limit') + 1]) != float(expected_limit):
        raise ValueError('CellSieve solver time limit does not match the requested cohort setting')
    if '--adata-path' not in argv or local_path(argv[argv.index('--adata-path')+1]) != local_path(reference_state['dataset']):
        raise ValueError('Latest CellSieve run used a different cortex cohort')
    table = pd.read_csv(run / 'benchmark_summary_all.csv', dtype={'requested_panel_size': str})
    validate_table(table, run, {**reference_state, 'methods': ['cell_sieve']})
    table['result_origin'] = 'latest_package_native_run'
    return table, run, meta


def compile_saved_full_l1(batch=None):
    """Use validated current full-cortex runs and the five saved real L1 rows."""
    if batch is None:
        batch = Path(json.loads((FIGURE / 'results/native_comparator_batch.json').read_text())['batch'])
    batch = Path(batch).resolve()
    if not batch.is_relative_to((NATIVE / 'runs').resolve()):
        raise ValueError('Expected Figure 1 native batch')
    source_run = batch / 'full'
    source_state = json.loads((source_run / 'run_manifest.json').read_text())
    current_methods = [m for m in source_state['methods'] if m not in ('l1_regularization', 'linear_regression_de')]
    new_cell, cell_run, cell_meta = load_latest_cellsieve('full', source_state)
    frames = []
    for method in current_methods:
        if method == 'cell_sieve':
            frames.append(new_cell)
            continue
        provenance = json.loads((source_run / method / 'execution_provenance.json').read_text())
        if not provenance.get('native_only') or not provenance.get('evaluation_unchanged'):
            raise ValueError('Missing current native provenance: ' + method)
        frames.append(pd.read_csv(source_run / method / 'benchmark_summary_all.csv',
                                  dtype={'requested_panel_size': str}))
    current = validate_table(pd.concat(frames, ignore_index=True), NATIVE / 'runs',
                             {**source_state, 'methods': current_methods})
    current.loc[current.method != 'cell_sieve', 'result_origin'] = 'current_native_batch'

    legacy_file = FIGURE / 'results/benchmark/benchmark_summary_all.csv'
    legacy = pd.read_csv(legacy_file, dtype={'requested_panel_size': str})
    legacy = legacy[legacy.method == 'l1_regularization'].copy()
    if len(legacy) != len(PANELS) or set(legacy.requested_panel_size) != {str(n) for n in PANELS}:
        raise ValueError('Saved L1 does not cover the five original panel sizes')
    legacy_root = FIGURE / 'results/benchmark/row_outputs'
    original_outputs = []
    for row in legacy.itertuples():
        if row.dataset != DATA.stem or Path(row.adata_path).name != DATA.name:
            raise ValueError('Saved L1 dataset name differs from full cortex')
        if row.status != 'ok' or row.evaluation_mode != 'full_train_full_eval_plus_3fold_cv':
            raise ValueError('Saved L1 evaluation was not successful or comparable')
        size = int(row.requested_panel_size)
        if int(row.actual_panel_size) != size or float(row.elapsed_seconds) <= 0 or float(row.peak_rss_mb) <= 0:
            raise ValueError('Saved L1 lacks exact panel size or measured resources')
        relative = Path(row.output_dir)
        if not relative.is_relative_to('Benchmark/results'):
            raise ValueError('Unexpected saved L1 output path')
        output = (legacy_root / relative.relative_to('Benchmark/results')).resolve()
        if not output.is_relative_to(legacy_root.resolve()):
            raise ValueError('Saved L1 output leaves the archive')
        genes = (output / 'selected_genes.txt').read_text().splitlines()
        if len(genes) != size or len(genes) != len(set(genes)):
            raise ValueError('Saved L1 panel genes are incomplete')
        report = json.loads((output / 'classification_report.json').read_text())
        cv = json.loads((output / 'cv_metrics.json').read_text())
        for saved, actual in ((row.accuracy, report['accuracy']),
                              (row.macro_f1, report['macro avg']['f1-score']),
                              (row.cv_accuracy, cv['cv_accuracy']),
                              (row.cv_macro_f1, cv['cv_macro_f1'])):
            if abs(float(saved) - float(actual)) > 1e-10:
                raise ValueError('Saved L1 score does not match original panel report')
        original_outputs.append(str(output))
    legacy['original_output_dir'] = legacy.output_dir
    legacy['output_dir'] = original_outputs
    legacy['result_origin'] = 'saved_historical_l1'
    table = pd.concat([current, legacy], ignore_index=True)
    if len(table) != 25 or set(table.method) != set(current_methods + ['l1_regularization']):
        raise ValueError('Compiled full cortex must contain five requested methods and 25 real rows')
    compiled_run = FIGURE / 'results/full_cortex_package_l1'
    compiled_run.mkdir(parents=True, exist_ok=True)
    summary = compiled_run / 'benchmark_summary_all.csv'
    table.to_csv(summary, index=False)
    manifest = {
        'analysis': 'full', 'status': 'complete_with_saved_l1',
        'provenance_mode': 'saved_full_l1', 'dataset': str(DATA),
        'input_shape': [86427, 5083], 'current_input_sha256': source_state['input_sha256'],
        'saved_l1_input_sha256': None,
        'saved_l1_input_limit': 'Historical file records the same dataset name but no input checksum',
        'panel_sizes': list(PANELS), 'methods': current_methods + ['l1_regularization'],
        'cellsieve_package_source': str(local_path(cell_meta['cellsieve_source'])),
        'cellsieve_package_sha256': cell_meta['cellsieve_sha256'],
        'cellsieve_time_limit_seconds': cell_meta['full_time_limit_seconds'],
        'result_sources': {
            'current_native_methods': {m: str((cell_run if m == 'cell_sieve' else source_run / m).relative_to(ROOT)) for m in current_methods},
            'saved_l1_table': str(legacy_file.relative_to(ROOT)),
            'saved_l1_table_sha256': sha(legacy_file),
            'saved_l1_outputs': str(legacy_root.relative_to(ROOT)),
        },
        'summary_sha256': sha(summary), 'successful_rows': len(table),
    }
    write_json(compiled_run / 'run_manifest.json', manifest)
    write_json(FIGURE / 'results/full_package_l1_run.json', {
        'run_directory': str(compiled_run.relative_to(ROOT)),
        'summary_sha256': sha(summary), 'saved_l1_table_sha256': sha(legacy_file),
        'current_input_sha256': source_state['input_sha256']})
    return table, compiled_run, manifest


def compile_latest_subsample():
    """Replace only CellSieve in the shared real subset with validated package runs."""
    source_run = Path(json.loads((FIGURE / 'results/native_subsample_run.json').read_text())['run_directory'])
    source_run = ROOT / source_run
    source_state = json.loads((source_run / 'run_manifest.json').read_text())
    new_cell, cell_run, cell_meta = load_latest_cellsieve('subsample', source_state)
    frames = [new_cell]
    for method in source_state['methods']:
        if method == 'cell_sieve':
            continue
        provenance = json.loads((source_run / method / 'execution_provenance.json').read_text())
        if not provenance.get('native_only') or not provenance.get('evaluation_unchanged'):
            raise ValueError('Missing native provenance: ' + method)
        rows = pd.read_csv(source_run / method / 'benchmark_summary_all.csv',
                           dtype={'requested_panel_size': str})
        rows['result_origin'] = 'current_native_batch'
        frames.append(rows)
    table = validate_table(pd.concat(frames, ignore_index=True), NATIVE / 'runs', source_state)
    compiled_run = FIGURE / 'results/subsample_package_cellsieve'
    compiled_run.mkdir(parents=True, exist_ok=True)
    summary = compiled_run / 'benchmark_summary_all.csv'
    table.to_csv(summary, index=False)
    manifest = {**source_state,
        'status': 'complete_with_package_cellsieve',
        'provenance_mode': 'mixed_native_package',
        'cellsieve_package_source': str(local_path(cell_meta['cellsieve_source'])),
        'cellsieve_package_sha256': cell_meta['cellsieve_sha256'],
        'cellsieve_time_limit_seconds': cell_meta['subsample_time_limit_seconds'],
        'result_sources': {m: str((cell_run if m == 'cell_sieve' else source_run / m).relative_to(ROOT))
                           for m in source_state['methods']},
        'summary_sha256': sha(summary), 'successful_rows': len(table)}
    write_json(compiled_run / 'run_manifest.json', manifest)
    write_json(FIGURE / 'results/subsample_package_cellsieve_run.json', {
        'run_directory': str(compiled_run.relative_to(ROOT)),
        'summary_sha256': sha(summary), 'input_sha256': source_state['input_sha256']})
    return table, compiled_run, manifest


def cellsieve_solver_rows(cohort, manifest, run):
    """Read the package's actual Gurobi completion record for each panel."""
    run = Path(run).resolve()
    dataset_stem = Path(manifest['dataset']).stem
    rows = []
    for panel in PANELS:
        log_path = run / dataset_stem / 'cell_sieve_package_runs' / 'Package_0.1.0' / f'panel_size_{panel}' / 'stdout.txt'
        output = log_path.read_text()
        if ('Gurobi Optimizer version' not in output or
                f'FINAL REPORT: {panel} GENES' not in output or
                'Solution count ' not in output):
            raise ValueError('Missing native Gurobi solution evidence: ' + str(log_path))
        status = ('time_limit_feasible' if 'Time limit reached' in output else
                  'gap_target_reached' if 'Optimal solution found' in output else None)
        match = re.search(r'Best objective ([^,]+), best bound ([^,]+), gap ([0-9.]+)%', output)
        count = re.search(r'Solution count (\d+)', output)
        duration = re.search(r'Explored [^\n]+ in ([0-9.]+) seconds', output)
        if status is None or match is None or count is None or duration is None:
            raise ValueError('Incomplete native Gurobi completion record: ' + str(log_path))
        rows.append({'cohort': cohort, 'panel_size': panel,
            'solver_status': status, 'gurobi_solutions': int(count.group(1)),
            'best_objective': float(match.group(1)), 'best_bound': float(match.group(2)),
            'optimality_gap_percent': float(match.group(3)),
            'gurobi_seconds': float(duration.group(1)),
            'configured_time_limit_seconds': manifest['cellsieve_time_limit_seconds'],
            'native_stdout': str(log_path.relative_to(ROOT)),
            'native_stdout_sha256': sha(log_path)})
    return rows


def prepare_figure_table(table, manifest):
    """Keep only the six requested methods and annotate observed resources."""
    shown = [m for m in manifest['methods'] if m != 'linear_regression_de']
    table = table[table.method.isin(shown)].copy()
    if set(table.method) != set(shown):
        raise ValueError('Incomplete requested method set')
    table['peak_rss_gib'] = table.peak_rss_mb / 1024
    table['ranking_reused'] = False
    ranked = ('nsforest', 'comet', 'logistic_regression_de')
    if manifest.get('provenance_mode') != 'saved_full_l1':
        ranked += ('l1_regularization',)
    for method in ranked:
        rows = table[table.method == method].sort_values('actual_panel_size')
        if len(rows):
            table.loc[rows.index[1:], 'ranking_reused'] = True
    if 'result_origin' not in table:
        table['result_origin'] = 'current_native_batch'
    return table


def plot_metric(axis, table, methods, metric, title, letter, subtitle_size=None):
    styles = dict(METHOD_STYLE)
    styles['cell_sieve'] = ('CellSieve 0.1.0', styles['cell_sieve'][1])
    styles['comet'] = ('COMET', styles['comet'][1])
    for method in methods:
        display, color = styles[method]
        rows = table[table.method == method].sort_values('actual_panel_size')
        if len(rows) != len(PANELS) or rows[metric].isna().any():
            raise ValueError('Missing real measurements for ' + method + '/' + metric)
        axis.plot(rows.actual_panel_size, rows[metric], '-o', color=color,
                  label=display, ms=6.2, lw=2.25)
    if subtitle_size is None:
        axis.set_title(title, loc='left', fontsize=20, fontweight='bold', pad=17)
    else:
        heading, _, subtitle = title.partition('\n')
        axis.set_title(heading, loc='left', fontsize=20, fontweight='bold', pad=36)
        axis.text(0, 1.025, subtitle, transform=axis.transAxes,
                  fontsize=subtitle_size, ha='left', va='bottom')
    axis.set_xlabel('Selected genes')
    axis.set_xscale('log', base=2)
    axis.set_xticks(PANELS, labels=[str(n) for n in PANELS])
    axis.set_xlim(PANELS[0]/1.2, PANELS[-1]*1.2)
    axis.minorticks_off()
    if metric in ('accuracy', 'macro_f1', 'cv_accuracy', 'cv_macro_f1'):
        axis.set_ylabel({
            'macro_f1': 'Macro-F1 recovery',
            'cv_macro_f1': 'CV macro-F1 recovery',
            'accuracy': 'Refit accuracy',
            'cv_accuracy': 'Conditional CV accuracy',
        }[metric])
        axis.set_ylim(0, 1.02)
    elif metric == 'elapsed_seconds':
        axis.set_ylabel('Wall time (s, log scale)')
        axis.set_yscale('log')
    else:
        axis.set_ylabel('Peak RSS (GiB)')
        axis.set_ylim(bottom=0)
    if metric in ('macro_f1', 'cv_macro_f1'):
        axis.axhline(0.9, color='#303030', linewidth=2.4,
                     linestyle=(0, (6, 3)), zorder=1.5)
        axis.text(.025, .9, '0.90', transform=axis.get_yaxis_transform(),
                  ha='left', va='bottom', fontsize=16, fontweight='bold',
                  color='#303030', bbox={'facecolor': 'white', 'edgecolor': 'none',
                                        'alpha': .9, 'pad': 1.5}, zorder=4)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', color='.88', linewidth=.55)
    axis.text(-.11, 1.06, letter, transform=axis.transAxes,
              fontweight='bold', fontsize=24, va='bottom')


def render_publication_figures(full, subsample):
    """One methods-plus-macro-F1 main figure and one resource supplement."""
    set_style()
    plt.rcParams.update({'font.size': 17, 'axes.titlesize': 20,
                         'axes.labelsize': 18, 'legend.fontsize': 17,
                         'xtick.labelsize': 16, 'ytick.labelsize': 16})
    full_table, full_run, full_manifest = full
    subset_table, subset_run, subset_manifest = subsample
    package_source = CELLSIEVE
    package_hash = sha(package_source)
    if any(manifest.get('cellsieve_package_sha256') != package_hash
           for manifest in (full_manifest, subset_manifest)):
        raise ValueError('Figure 1 requires CellSieve panels from the current package source')
    full_table = prepare_figure_table(full_table, full_manifest)
    subset_table = prepare_figure_table(subset_table, subset_manifest)
    if len(full_table) != 25 or len(subset_table) != 30:
        raise ValueError('Main figure requires 25 full and 30 subsample real rows')
    full_methods = [m for m in full_manifest['methods'] if m != 'linear_regression_de']
    subset_methods = [m for m in subset_manifest['methods'] if m != 'linear_regression_de']
    if set(subset_methods) != set(full_methods + ['comet']):
        raise ValueError('COMET must appear only in the shared subsample')
    counts_path = Path(subset_manifest['dataset']).parent / 'cohort_class_counts.csv'
    class_counts = pd.read_csv(counts_path)
    n_celltypes = len(class_counts)
    if (int(class_counts['full'].sum()) != full_manifest['input_shape'][0] or
            int(class_counts['subset'].sum()) != subset_manifest['input_shape'][0]):
        raise ValueError('Figure title counts do not match the real cohorts')
    full_title = (f"Full B6J Cortex\n{full_manifest['input_shape'][0]:,} cells "
                  f"({n_celltypes} cell types) · {full_manifest['input_shape'][1]:,} genes")
    subset_title = (f"Subsampled B6J Cortex\n{subset_manifest['input_shape'][0]:,} cells "
                    f"({n_celltypes} cell types) · {subset_manifest['input_shape'][1]:,} genes")
    directory = FIGURE / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([full_table.assign(cohort='full_cortex_without_comet'),
                          subset_table.assign(cohort='subsampled_cortex_including_comet')],
                         ignore_index=True)
    result_path = directory / 'Figure1_results.csv'
    combined.to_csv(result_path, index=False)
    resource_columns = ['cohort', 'dataset', 'method', 'requested_panel_size',
        'actual_panel_size', 'elapsed_seconds', 'peak_rss_mb', 'peak_rss_gib',
        'ranking_reused', 'result_origin', 'n_workers_requested',
        'thread_count_peak', 'output_dir']
    resource_path = directory / 'Supplemental_Figure1_resources.csv'
    combined[resource_columns].to_csv(resource_path, index=False)
    totals_path = directory / 'Supplemental_Figure1_method_resources.csv'
    combined.groupby(['cohort', 'method'], sort=False).agg(
        total_measured_panel_seconds=('elapsed_seconds', 'sum'),
        peak_rss_gib=('peak_rss_gib', 'max'),
        completed_panels=('actual_panel_size', 'count')).to_csv(totals_path)
    full_cell_run = ROOT / full_manifest['result_sources']['current_native_methods']['cell_sieve']
    subset_cell_run = ROOT / subset_manifest['result_sources']['cell_sieve']
    solver_path = directory / 'CellSieve_solver_status.csv'
    pd.DataFrame(
        cellsieve_solver_rows('full_cortex_without_comet', full_manifest, full_cell_run) +
        cellsieve_solver_rows('subsampled_cortex_including_comet', subset_manifest, subset_cell_run)
    ).to_csv(solver_path, index=False)

    method_image = plt.imread(METHODS_IMAGE)
    nonwhite = (method_image[:, :, :3] < .985).any(axis=2)
    y, x = nonwhite.nonzero()
    if len(x) == 0:
        raise ValueError('The supplied methods diagram is blank')
    border = 18
    method_image = method_image[max(0, y.min()-border):min(len(nonwhite), y.max()+border+1),
                                max(0, x.min()-border):min(nonwhite.shape[1], x.max()+border+1)]
    main = plt.figure(figsize=(22, 11.7), facecolor='white')
    grid = main.add_gridspec(2, 2, left=.045, right=.97, bottom=.20, top=.90,
                             width_ratios=(2.35, 1), hspace=.40, wspace=.16)
    methods_axis = main.add_subplot(grid[:, 0])
    methods_axis.imshow(method_image)
    methods_axis.axis('off')
    methods_axis.text(-.015, 1.03, 'A', transform=methods_axis.transAxes,
                      fontweight='bold', fontsize=24, va='bottom')
    plot_b = main.add_subplot(grid[0, 1])
    plot_c = main.add_subplot(grid[1, 1])
    plot_metric(plot_b, full_table, full_methods, 'macro_f1', 'Benchmark in '+full_title, 'B', subtitle_size=15)
    plot_metric(plot_c, subset_table, subset_methods, 'macro_f1', 'Benchmark in '+subset_title, 'C', subtitle_size=15)
    handles, names = plot_c.get_legend_handles_labels()
    main.legend(handles, names, loc='lower center', bbox_to_anchor=(.52, .077),
                ncol=6, frameon=False, fontsize=17)
    save(main, 'figure1_method_benchmark/figures', 'Figure1')

    supplement = plt.figure(figsize=(18, 28.5), facecolor='white')
    supp_grid = supplement.add_gridspec(5, 2, left=.075, right=.975, bottom=.11,
                                        top=.95, hspace=.48, wspace=.23)
    supp_metrics = ('cv_macro_f1', 'accuracy', 'cv_accuracy',
                    'elapsed_seconds', 'peak_rss_gib')
    supp_axes = []
    for row, metric in enumerate(supp_metrics):
        for col, (table, methods, cohort_title) in enumerate((
                (full_table, full_methods, full_title),
                (subset_table, subset_methods, subset_title))):
            axis = supplement.add_subplot(supp_grid[row, col])
            plot_metric(axis, table, methods, metric, cohort_title,
                        chr(ord('A') + 2*row + col))
            supp_axes.append(axis)
    handles, names = supp_axes[1].get_legend_handles_labels()
    supplement.legend(handles, names, loc='lower center',
                      bbox_to_anchor=(.52, .055), ncol=6, frameon=False, fontsize=17)
    save(supplement, 'figure1_method_benchmark/figures', 'Supplemental_Figure1')

    shared = {
        'renderer_sha256': sha(Path(__file__)),
        'methods_source_pdf': str(METHODS_PDF.relative_to(ROOT)),
        'methods_source_pdf_sha256': sha(METHODS_PDF),
        'methods_panel_image_sha256': sha(METHODS_IMAGE),
        'full_results_source': str(full_run.relative_to(ROOT)),
        'full_results_sha256': sha(full_run / 'benchmark_summary_all.csv'),
        'subsample_results_source': str(subset_run.relative_to(ROOT)),
        'subsample_results_sha256': sha(subset_run / 'benchmark_summary_all.csv'),
        'plotted_methods_full': full_methods,
        'plotted_methods_subsample': subset_methods,
        'combined_results_sha256': sha(result_path),
        'resource_table_sha256': sha(resource_path),
        'cellsieve_solver_table_sha256': sha(solver_path),
        'full_l1_source': full_manifest.get('result_sources', {}).get('saved_l1_table'),
        'full_l1_input_limit': full_manifest.get('saved_l1_input_limit'),
        'cellsieve_package_source': full_manifest.get('cellsieve_package_source'),
        'cellsieve_package_sha256': full_manifest.get('cellsieve_package_sha256'),
        'cellsieve_full_run': full_manifest.get('result_sources', {}).get('current_native_methods', {}).get('cell_sieve'),
        'cellsieve_subsample_run': subset_manifest.get('result_sources', {}).get('cell_sieve'),
        'cellsieve_full_time_limit_seconds': full_manifest.get('cellsieve_time_limit_seconds'),
        'cellsieve_subsample_time_limit_seconds': subset_manifest.get('cellsieve_time_limit_seconds'),
    }
    for stem, role, panels in (
        ('Figure1', 'main', {'A': 'provided methods outline',
            'B': 'full refit macro-F1',
            'C': 'subsample refit macro-F1'}),
        ('Supplemental_Figure1', 'supplement', {
            'A-B': 'conditional CV macro-F1 moved from earlier main layout',
            'C-D': 'refit accuracy', 'E-F': 'conditional CV accuracy',
            'G-H': 'runtime', 'I-J': 'peak memory'})):
        write_json(directory / (stem + '.provenance.json'), {
            **shared, 'role': role, 'panels': panels,
            'artifacts': {ext: sha(directory / (stem + '.' + ext))
                          for ext in ('pdf', 'svg', 'png')}})
    (directory / 'README.md').write_text(
        '# Figure 1 publication figures\n\n'
        '`Figure1.pdf/svg/png` places the supplied CellSieve methods outline at '
        'left (A), with full-cortex refit macro-F1 (B) and common COMET-inclusive '
        'subsample refit macro-F1 (C) stacked at right.\n\n'
        '`Supplemental_Figure1.pdf/svg/png` contains ten plots: the two moved '
        'conditional CV macro-F1 panels, then refit '
        'accuracy, conditional CV accuracy, observed runtime, and peak '
        'process-tree memory for each cohort. The 10/20/40/80/160 panel sizes '
        'and original numerical evaluator are retained; panel-size axes use '
        'base-2 log spacing. Dashed lines mark macro-F1 = 0.90 on macro-F1 '
        'plots. All newly typeset figure text uses enlarged Helvetica.\n\n'
        '`Figure1_results.csv` contains every plotted real-data row; '
        '`Supplemental_Figure1_resources.csv` and '
        '`Supplemental_Figure1_method_resources.csv` contain measured resource '
        'values. `CellSieve_solver_status.csv` records actual Gurobi status, time '
        'and gap for every native package panel. Each figure has a provenance JSON. '
        'Full-cortex L1 uses validated '
        'saved real runs whose historical input checksum was not recorded. '
        'Current CellSieve points use only the native package at '
        '`external/cell_sieve_package/cellsieve.py`, with its source hash '
        'recorded in provenance. Gurobi limits are 600 seconds per full '
        'panel and 300 seconds per subset panel. Older v4 CellSieve rows are excluded. '
        'The original methods PDF is preserved unchanged in the parent folder.\n')
    print('Saved publication figures: ' + str(directory / 'Figure1.pdf') +
          ' and ' + str(directory / 'Supplemental_Figure1.pdf'), flush=True)


def finalize_batch(batch, only=None):
    """Validate completed real native outputs after a repaired failed method is rerun."""
    batch = Path(batch).resolve()
    if not batch.is_relative_to((NATIVE / 'runs').resolve()):
        raise ValueError('Batch must be inside Figure 1 native/runs')
    for name in ([only] if only else ('full', 'subsample')):
        run = batch / name
        state = json.loads((run / 'run_manifest.json').read_text())
        frames = []
        for method in state['methods']:
            provenance = json.loads((run / method / 'execution_provenance.json').read_text())
            if not provenance.get('native_only') or not provenance.get('evaluation_unchanged'):
                raise ValueError('Missing native provenance: ' + method)
            frames.append(pd.read_csv(run / method / 'benchmark_summary_all.csv', dtype={'requested_panel_size': str}))
        table = validate_table(pd.concat(frames, ignore_index=True), run, state)
        summary = run / 'benchmark_summary_all.csv'; table.to_csv(summary, index=False)
        state.setdefault('initial_execution_attempts', state.get('completed_methods', []))
        state['repaired_failures'] = sorted(set(state.get('repaired_failures', []) + state.get('failed_methods', [])))
        state['failed_methods'] = []
        state['completed_methods'] = [{'method': m, 'validated_status': 'ok'} for m in state['methods']]
        state.pop('error', None)
        state.update(status='complete', summary_sha256=sha(summary), successful_rows=len(table),
                     validated_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / 'run_manifest.json', state)
        write_json(FIGURE / f'results/native_{name}_run.json', {
            'run_directory': str(run.relative_to(ROOT)), 'summary_sha256':sha(summary),
            'input_sha256':state['input_sha256']})
    complete = all(json.loads((batch / n / 'run_manifest.json').read_text())['status'] == 'complete' for n in ('full', 'subsample'))
    write_json(FIGURE / 'results/native_active_batch.json', {'batch':str(batch), 'status':'complete' if complete else 'running'})
    if complete:
        render_publication_figures(load_native('full'), load_native('subsample'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-native', action='store_true', help='Run real cortex benchmark and then render')
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--finalize-batch', type=Path, help='Validate and render an existing real batch after a repaired method rerun')
    parser.add_argument('--analysis', choices=['full','subsample'], help='Finalize just one completed cohort')
    parser.add_argument('--check', action='store_true', help='Check the physical native runtime without running models')
    parser.add_argument('--reuse-saved-full-l1', action='store_true', help='Render full cortex from current native methods and the validated saved L1 panels')
    args = parser.parse_args()
    if args.n_workers < 1:
        parser.error('--n-workers must be positive')
    if args.check:
        preflight()
        return
    if args.reuse_saved_full_l1:
        full = compile_saved_full_l1()
        render_publication_figures(full, compile_latest_subsample())
    elif args.finalize_batch:
        finalize_batch(args.finalize_batch, args.analysis)
    elif args.run_native:
        execute_native(args.n_workers)
    else:
        full = compile_saved_full_l1()
        render_publication_figures(full, compile_latest_subsample())


if __name__ == '__main__':
    main()
