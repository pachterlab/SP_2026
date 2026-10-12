"""Refit fixed 50-gene MERFISH classifier and audit against historical predictions.

No gene selection, no hyperparameter search. Source datasets are read-only.
The original MERFISH fitting script is unavailable. This explicit reconstruction
uses the classifier defaults in the original CellSieve output code, with scaling
fitted independently within each validation fold.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/figure2/results'
SOURCE = ROOT / 'h5ads/Merfish_cortex.h5ad'
DEST = BASE / 'merfish_panel50_coefficients'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scores(y, pred):
    return {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
        'macro_recall': recall_score(y, pred, average='macro', zero_division=0),
        'weighted_recall': recall_score(y, pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y, pred, average='macro', zero_division=0),
    }


def estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs', class_weight='balanced',
        max_iter=2000, tol=1e-4, random_state=42))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', type=float, help='Explicit expected selection alpha; default is active Figure 1 alpha')
    parser.add_argument('--force', action='store_true', help='Refit even when matching saved outputs exist')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    genes_path = BASE / 'panel50/selected_genes.txt'
    selection_provenance = json.loads((BASE / 'panel50/provenance.json').read_text())
    pointer = json.loads((ROOT / 'results/figure1/results/latest_cellsieve_batch.json').read_text())
    alpha = float(pointer['alpha']) if args.alpha is None else args.alpha
    if pointer.get('status') != 'complete' or selection_provenance.get('cell_sieve_alpha') != alpha:
        raise ValueError('MERFISH panel alpha must match the active Figure 1 run')
    source_sha = sha256(SOURCE)
    genes_sha = sha256(genes_path)
    prior_path = DEST / 'provenance.json'
    if not args.force and prior_path.exists():
        prior = json.loads(prior_path.read_text())
        expected = {'cell_sieve_selection_alpha': alpha, 'source_sha256': source_sha,
                    'selected_genes_sha256': genes_sha, 'script_sha256': sha256(__file__),
                    'sklearn_version': sklearn.__version__}
        outputs = ['metrics.json', 'cv_spatial_predictions.csv', 'model_coefficients.csv',
                   'historical_metric_comparison.csv', 'cv_confusion_matrix_row_normalized.csv']
        if all(prior.get(k) == v for k, v in expected.items()) and all((DEST / name).is_file() for name in outputs):
            print(f'Reusing validated MERFISH refit for alpha {alpha:g} and identical genes/input/runtime', flush=True)
            return
    genes = genes_path.read_text().splitlines()
    saved_path = BASE / 'reference_checks/merfish_previous_predictions.csv'
    saved = pd.read_csv(saved_path, dtype={'cell_label': str})
    old_metrics = json.loads((BASE / 'reference_checks/merfish_previous_metrics.json').read_text())
    a = ad.read_h5ad(SOURCE, backed='r')
    counts = a.obs['class'].astype(str).value_counts()
    keep = a.obs['class'].astype(str).isin(counts[counts >= 10].index)
    obs = a.obs.loc[keep].copy()
    if obs.index.tolist() != saved.cell_label.tolist():
        raise ValueError('Cell cohort/order differs from saved spatial predictions')
    if not np.array_equal(obs['class'].astype(str).to_numpy(), saved.true_class.to_numpy()):
        raise ValueError('Cell labels differ from saved spatial predictions')
    if not np.allclose(obs[['x', 'y']], saved[['x', 'y']]):
        raise ValueError('Spatial coordinates differ from saved predictions')
    if len(genes) != 50 or len(set(genes)) != 50 or not set(genes).issubset(a.var_names):
        raise ValueError('Missing or duplicated panel genes')
    X = a[:, genes].to_memory().X
    X = X[keep.to_numpy()]
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    X = X.astype(np.float64)
    normalization = dict(a.uns)
    n_total = a.n_obs
    a.file.close()
    if not np.isfinite(X).all():
        raise ValueError('Non-finite expression')
    y = obs['class'].astype(str).to_numpy()
    if X.shape != (8891, 50) or len(np.unique(y)) != 9:
        raise ValueError('Unexpected MERFISH cohort dimensions')
    saved_scores = scores(y, saved.predicted_class)
    if not all(np.isclose(saved_scores[k], old_metrics[k], atol=1e-12, rtol=0) for k in saved_scores):
        raise ValueError('Historical metrics do not match saved predictions')
    print(f'Validated source: {X.shape[0]} cells, {X.shape[1]} genes, 9 classes; normalization={normalization}', flush=True)
    pred = np.empty(len(y), dtype=object)
    fold_ids = np.empty(len(y), dtype=int)
    fold_metrics = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    for fold, (train, test) in enumerate(cv.split(X, y), 1):
        model = estimator().fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        fold_ids[test] = fold
        fold_metrics.append({'fold': fold, 'n_train': len(train), 'n_test': len(test), **scores(y[test], pred[test])})
        print(f'Fold {fold}: accuracy={fold_metrics[-1]["accuracy"]:.8f}, macro_f1={fold_metrics[-1]["macro_f1"]:.8f}', flush=True)
    new_scores = scores(y, pred)
    comparison = pd.DataFrame([
        {'metric': k, 'historical': old_metrics[k], 'refit_cv': new_scores[k],
         'difference': new_scores[k] - old_metrics[k],
         'match_at_1e-12': bool(np.isclose(new_scores[k], old_metrics[k], atol=1e-12, rtol=0))}
        for k in new_scores
    ])
    comparison.to_csv(DEST / 'historical_metric_comparison.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(DEST / 'cv_fold_metrics.csv', index=False)
    result = saved[['cell_label', 'x', 'y', 'true_class']].copy()
    result['predicted_class'] = pred
    result['fold'] = fold_ids
    result['historical_predicted_class'] = saved.predicted_class
    result.to_csv(DEST / 'cv_spatial_predictions.csv', index=False)
    pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T.to_csv(DEST / 'cv_classification_report.csv')
    # Display coefficients from a separate fit on all eligible cells; CV scores
    # above always use held-out predictions, never these training predictions.
    model = estimator().fit(X, y)
    scaler = model.named_steps['standardscaler']
    classifier = model.named_steps['logisticregression']
    coefficients = pd.DataFrame(classifier.coef_, index=classifier.classes_, columns=genes)
    coefficients.index.name = 'class'
    coefficients.to_csv(DEST / 'model_coefficients.csv')
    pd.DataFrame({'gene': genes, 'mean': scaler.mean_, 'scale': scaler.scale_}).to_csv(DEST / 'feature_scaling.csv', index=False)
    pd.DataFrame({'class': classifier.classes_, 'intercept': classifier.intercept_}).to_csv(DEST / 'model_intercepts.csv', index=False)
    pd.DataFrame(confusion_matrix(y, pred, labels=classifier.classes_, normalize='true'),
                 index=classifier.classes_, columns=classifier.classes_).to_csv(DEST / 'cv_confusion_matrix_row_normalized.csv')
    all_pred = model.predict(X)
    metrics = {'n_cells_total': n_total, 'n_cells_evaluated': len(y), 'n_labels_evaluated': 9,
               'n_genes': 50, 'n_splits': 5, 'cv': new_scores, 'training_only': scores(y, all_pred),
               'historical_prediction_agreement': float(np.mean(pred == saved.predicted_class.to_numpy())),
               'historical_prediction_differences': int(np.sum(pred != saved.predicted_class.to_numpy())),
               'all_historical_metrics_match_at_1e-12': bool(comparison['match_at_1e-12'].all())}
    (DEST / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    provenance = {'source': str(SOURCE), 'source_sha256': source_sha,
                  'cell_sieve_selection_alpha': alpha,
                  'selection_provenance': selection_provenance,
                  'selected_genes_source': str(genes_path), 'selected_genes_sha256': genes_sha,
                  'historical_predictions_source': str(saved_path), 'historical_predictions_sha256': sha256(saved_path),
                  'normalization_metadata': normalization,
                  'expression': 'AnnData X (already log2p); no further log transform',
                  'cohort': 'class label, retain classes with >=10 cells; exact historical cell order, labels, coordinates verified',
                  'cv': 'StratifiedKFold(5, shuffle=True, random_state=42); fold-specific StandardScaler',
                  'model': classifier.get_params(), 'sklearn_version': sklearn.__version__,
                  'coefficient_fit': 'Separate fit on all 8891 eligible cells; coefficients per standardized feature',
                  'original_recipe_status': 'Unavailable; explicit reconstruction based on original CellSieve classifier defaults; no hyperparameter search',
                  'script_sha256': sha256(__file__)}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
