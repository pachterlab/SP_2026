"""Verify all saved results/supplementary predictions and active benchmark artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from workspace_paths import local_path

from run_8cube_supplements import ROOT, TISSUES, METHODS, PANELS, digest, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'results/supplementary/verification.json')
    args = parser.parse_args()
    root = ROOT / 'results/supplementary'
    inputs = root / 'fullinput_benchmarks/h5ads'
    cohort_hashes = {}
    for path in sorted(inputs.glob('*.h5ad')):
        expected = json.loads(path.with_suffix('.json').read_text())['benchmark_sha256']
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'Full-input cohort hash mismatch: {path}')
        cohort_hashes[path.name] = actual
    if len(cohort_hashes) != 16:
        raise ValueError('Expected 16 unique prepared cohorts')
    n = 0
    for tissue in TISSUES:
        folder = root / 'alpha_subsampling' / tissue
        scores = pd.read_csv(folder / 'results.csv')
        if len(scores) != 100 or not scores.status.eq('ok').all() or scores[['repeat','alpha']].duplicated().any():
            raise ValueError(f'Incomplete sensitivity run: {tissue}')
        config = json.loads((folder / 'configuration.json').read_text())
        if config['input_sha256'] != cohort_hashes[f'{tissue}_B6J.h5ad']:
            raise ValueError(f'Sensitivity input does not match retained cohort: {tissue}')
        for repeat, runs in scores.groupby('repeat'):
            directory = folder / f'repeat_{int(repeat):03d}'
            membership = pd.read_csv(directory / 'membership.csv')
            if len(membership) != 5000 or membership.cell_id.duplicated().any():
                raise ValueError(f'Invalid subsample membership: {directory}')
            test_ids = set(membership.loc[membership.split.eq('test'), 'cell_id'])
            for row in runs.itertuples():
                run = directory / f'alpha_{row.alpha:.2f}'
                predictions = pd.read_csv(run / 'heldout_predictions.csv')
                if set(predictions.cell_id) != test_ids or len(predictions) != len(test_ids):
                    raise ValueError(f'Incorrect held-out cells: {run}')
                observed = f1_score(predictions.true, predictions.predicted, average='macro', zero_division=0)
                if not np.isclose(observed, row.test_macro_f1, rtol=0, atol=1e-12):
                    raise ValueError(f'Held-out macro-F1 mismatch: {run}')
                if not np.isclose(accuracy_score(predictions.true,predictions.predicted), row.test_accuracy, atol=1e-12):
                    raise ValueError(f'Held-out accuracy mismatch: {run}')
                native = run / f'cell_sieve_package_runs/Package_0.1.0/panel_size_40/panel_size_40_alpha_{row.alpha:.2f}'
                genes = (native / 'selected_genes.txt').read_text().splitlines()
                coefficients = pd.read_csv(native / 'model_coefficients.csv', index_col=0)
                if len(set(genes)) != 40 or coefficients.columns.tolist() != genes:
                    raise ValueError(f'Native sensitivity panel mismatch: {run}')
                if not (run / 'cell_sieve_package_train.h5ad').is_file():
                    raise ValueError(f'Missing retained training matrix: {run}')
                n += 1
        print(f'{tissue}: all 100 held-out evaluations verified', flush=True)
    selected = root / 'fullinput_benchmarks'
    results = pd.read_csv(selected / 'benchmark_results_both.csv')
    choices = pd.read_csv(selected / 'selected_alphas.csv')
    if len(results) != 255 or not results.status.eq('ok').all():
        raise ValueError('Expected 255 successful selected-alpha benchmark rows')
    for (analysis, cohort), group in results.groupby(['analysis','cohort']):
        if set(zip(group.method,group.requested_panel_size)) != {(m,k) for m in METHODS for k in PANELS}:
            raise ValueError(f'Missing method/budget: {analysis}/{cohort}')
        selection = choices[(choices.analysis == analysis) & ((choices.tissue if analysis=='tissues' else choices.strain)==cohort)].iloc[0]
        for row in group.itertuples():
            output = local_path(row.output_dir)
            genes = (output / 'selected_genes.txt').read_text().splitlines()
            if len(set(genes)) != row.actual_panel_size:
                raise ValueError(f'Selected panel size mismatch: {output}')
            for file in ['cv_metrics.json', 'classification_report.json']:
                if not (output / file).is_file():
                    raise ValueError(f'Missing native result: {output/file}')
            if row.method == 'cell_sieve':
                if row.selected_alpha != selection.selected_alpha:
                    raise ValueError(f'Wrong selected alpha: {output}')
                native = output.parent.parent / ('cell_sieve_package_runs/Package_0.1.0/'
                    f'panel_size_{row.requested_panel_size}/panel_size_{row.requested_panel_size}_alpha_{row.selected_alpha:.2f}')
                if (native / 'selected_genes.txt').read_text().splitlines() != genes:
                    raise ValueError(f'Adapter differs from native panel: {output}')
                if not (native / 'model_coefficients.csv').is_file():
                    raise ValueError(f'Missing native coefficients: {native}')
    artifacts = {}
    for folder in [ROOT/'figs']:
        for path in folder.iterdir():
            if path.suffix in ('.pdf','.svg','.png'):
                artifacts[str(path.relative_to(ROOT))] = digest(path)
    atomic_json(args.report, {'status':'complete', 'paired_alpha_runs_verified':n,
        'selected_alpha_benchmark_rows_verified':len(results), 'full_input_cohort_hashes':cohort_hashes,
        'figures_sha256':artifacts, 'verifier_sha256':digest(__file__)})
    print('Verified 900 paired runs, 255 benchmark rows and all 16 full-input cohorts.', flush=True)


if __name__ == '__main__':
    main()
