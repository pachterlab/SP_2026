#!/usr/bin/env python3
"""Paired stratified resampling and held-out evaluation of native CellSieve alpha."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import run_8cube_supplements as shared
import supplementary_helpers as native

ALPHAS = native.ALPHAS


def stratified_sample(labels, size, minimum, rng):
    """Reserve each class, then sample the remaining cells without replacement."""
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    if size >= len(labels):
        raise ValueError('Subsample size must be smaller than the original cohort')
    if size < minimum * len(groups) or any(len(g) < minimum for g in groups):
        raise ValueError('Cannot represent every class at the requested minimum and size')
    reserved = np.concatenate([rng.choice(g, minimum, replace=False) for g in groups])
    remaining = np.setdiff1d(np.arange(len(labels)), reserved)
    extra = rng.choice(remaining, size - len(reserved), replace=False)
    return np.sort(np.concatenate([reserved, extra]))


def paired_split(labels, fraction, rng):
    train, test = [], []
    for cls in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == cls))
        ntest = min(len(members)-3, max(2, int(round(len(members)*fraction))))
        if ntest < 2:
            raise ValueError('Each type needs at least three train and two test cells')
        test.extend(members[:ntest]); train.extend(members[ntest:])
    return np.sort(train), np.sort(test)


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    adjusted[order] = np.minimum(1, np.minimum.accumulate(
        (pvalues[order] * len(order) / np.arange(1, len(order)+1))[::-1])[::-1])
    return adjusted


def statistics(scores, permutations, seed, threshold=.05):
    """Blocked rank permutation omnibus, then all-pairs sign-flip tests + Benjamini–Hochberg."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 5 or scores.shape[0] < 10 or not np.isfinite(scores).all():
        raise ValueError('Require at least ten complete paired repeats and five finite scores')
    rng = np.random.default_rng(seed)
    ranks = rankdata(scores, axis=1)
    n, k = scores.shape
    centered = ranks - (k+1)/2
    observed = float(np.sum(centered.sum(axis=0)**2))
    exceed = 0
    for first in range(0, permutations, 500):
        count = min(500, permutations-first)
        ordering = np.argsort(rng.random((count, n, k)), axis=2)
        shuffled = np.take_along_axis(np.broadcast_to(centered, (count, n, k)), ordering, axis=2)
        exceed += int((np.sum(shuffled.sum(axis=1)**2, axis=1) >= observed-1e-12).sum())
    omnibus = (exceed+1)/(permutations+1)
    rows = []
    for a, b in itertools.combinations(range(k), 2):
        differences = scores[:, a] - scores[:, b]
        effect = float(differences.mean())
        hits = 0
        if np.all(differences == 0):
            pvalue = 1.
        elif n <= 20:
            # Exact sign-flip enumeration; process in bounded batches.
            total = 1 << n
            for start in range(0, total, 4096):
                ids = np.arange(start, min(start+4096, total), dtype=np.uint64)
                signs = 2*((ids[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)-1
                hits += int((np.abs(signs @ differences / n) >= abs(effect)-1e-12).sum())
            pvalue = hits/total
        else:
            for start in range(0, permutations, 500):
                count = min(500, permutations-start)
                signs = rng.choice([-1, 1], (count, n))
                hits += int((np.abs(signs @ differences / n) >= abs(effect)-1e-12).sum())
            pvalue = (hits+1)/(permutations+1)
        rows.append({'alpha_a': ALPHAS[a], 'alpha_b': ALPHAS[b],
                     'mean_macro_f1_difference_a_minus_b': effect, 'p_value': pvalue})
    pairs = pd.DataFrame(rows)
    pairs['p_bh'] = benjamini_hochberg(pairs.p_value)
    means = scores.mean(axis=0)
    # Exact mean ties prefer the lower alpha.
    leaders = np.flatnonzero(means == means.max())
    best = int(leaders[0])
    candidate = ALPHAS[best]
    relevant = pairs[(pairs.alpha_a == candidate) | (pairs.alpha_b == candidate)]
    beats_all = len(leaders) == 1 and all(
        row.p_bh < threshold and
        (row.mean_macro_f1_difference_a_minus_b > 0 if row.alpha_a == candidate
         else row.mean_macro_f1_difference_a_minus_b < 0)
        for row in relevant.itertuples())
    supported = omnibus < threshold and beats_all
    recommendation = {'omnibus_test': 'within-repeat alpha-label permutation of Friedman rank statistic',
                      'omnibus_p_value': omnibus, 'permutations': permutations,
                      'pairwise_test': 'paired two-sided sign-flip test; exact when repeats <=20',
                      'multiplicity': 'Benjamini–Hochberg adjustment over all ten alpha pairs',
                      'threshold': threshold, 'n_complete_repeats': n,
                      'best_mean_alpha': candidate, 'selected_alpha': candidate,
                      'supported_unique_winner': bool(supported),
                      'reason': 'highest mean beats every alternative after Benjamini–Hochberg correction' if supported
                      else 'highest observed mean selected empirically; no supported unique winner',
                      'scope': 'resampling stability conditional on this finite atlas; not independent mice or population-level biological replication'}
    summary = pd.DataFrame({'alpha': ALPHAS, 'mean_test_macro_f1': means,
                            'sd_test_macro_f1': scores.std(axis=0, ddof=1),
                            'n_repeats': n})
    return recommendation, pairs, summary


def analyze(output, repeats, permutations, seed, tissue):
    data = pd.read_csv(output / 'results.csv')
    if 'test_macro_f1' not in data:
        shared.atomic_json(output / 'analysis.json', {'status': 'incomplete',
                           'selected_alpha': None, 'reason': 'no successful held-out evaluations'})
        raise ValueError('No successful held-out evaluations. Inspect results.csv and rerun.')
    good = data[data.status == 'ok']
    matrix = good.pivot(index='repeat', columns='alpha', values='test_macro_f1').reindex(
        index=range(repeats), columns=ALPHAS)
    if matrix.isna().any().any():
        shared.atomic_json(output / 'analysis.json', {'status': 'incomplete',
                           'selected_alpha': None, 'reason': 'all planned paired runs must succeed'})
        raise ValueError('Incomplete paired benchmark. Rerun the same command; no alpha selected.')
    decision, pairs, summary = statistics(matrix.to_numpy(), permutations, seed)
    decision.update(tissue=tissue, strain='B6J', panel_size=40,
                    primary_metric='held-out macro-F1', status='complete')
    shared.atomic_json(output / 'analysis.json', decision)
    pairs.to_csv(output / 'paired_alpha_tests.csv', index=False)
    summary.to_csv(output / 'alpha_summary.csv', index=False)
    from compile_alpha_sensitivity import render_tissue
    render_tissue(matrix, summary, tissue, output)
    print(json.dumps(decision, indent=2), flush=True)


def run_tissue(args, tissue):
    import anndata as ad
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score, accuracy_score
    output = native.inside_workspace(args.out_dir / tissue)
    if args.stage == 'analyze':
        config = json.loads((output / 'configuration.json').read_text())
        analyze(output, config['repeats'], config['permutations'], config['seed'], tissue)
        return
    input_path = args.inputs_dir / 'h5ads' / f'{tissue}_B6J.h5ad'
    native.inside_workspace(args.inputs_dir)
    native.prepare_full_input(native.BASE, args.data_dir, tissue, 'B6J', input_path)
    input_path, checksum = native.get_subset(native.BASE, tissue, 'B6J', input_path)
    engine = native.engine_module()
    native.locked_config(output, {'tissue': tissue, 'strain': 'B6J', 'input_sha256': checksum,
                        'repeats': args.repeats, 'sample_size': args.sample_size,
                        'min_per_class': args.min_per_class, 'test_fraction': args.test_fraction,
                        'alphas': list(ALPHAS), 'panel_size': 40, 'seed': args.seed,
                        'permutations': args.permutations, 'workers': args.workers,
                        'time_limit': args.time_limit, 'script_sha256': shared.digest(__file__),
                        'engine_sha256': shared.digest(engine.__file__),
                        'cellsieve_sha256': shared.digest(engine.CELLSIEVE_PACKAGE_SCRIPT),
                        'gene_filter': 'name only; no variance/HVG filter',
                        'primary_metric': 'held-out macro-F1; panel selection restricted to training rows'})
    result_path = output / 'results.csv'
    previous = pd.read_csv(result_path).to_dict('records') if result_path.exists() else []
    records = {(int(r['repeat']), float(r['alpha'])): r for r in previous}
    atlas = ad.read_h5ad(input_path, backed='r')
    labels = atlas.obs.celltype.astype(str).to_numpy()
    try:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            rng = np.random.default_rng(seed)
            rows = stratified_sample(labels, args.sample_size, args.min_per_class, rng)
            local_labels = labels[rows]
            train, test = paired_split(local_labels, args.test_fraction, rng)
            directory = output / f'repeat_{repeat:03d}'
            directory.mkdir(exist_ok=True)
            membership = pd.DataFrame({'source_row': rows, 'cell_id': atlas.obs_names[rows],
                                       'celltype': local_labels, 'split': 'train'})
            membership.loc[test, 'split'] = 'test'
            membership.to_csv(directory / 'membership.csv', index=False)
            pd.crosstab(membership.celltype, membership.split).to_csv(directory / 'class_counts.csv')
            if all(records.get((repeat, a), {}).get('status') == 'ok' for a in ALPHAS):
                continue
            subset = atlas[rows].to_memory()
            subset.obs['source_row'] = rows
            subset_path = directory / f'{tissue}_B6J_repeat_{repeat:03d}.h5ad'
            if not subset_path.exists():
                subset.write_h5ad(subset_path, compression='gzip')
            encoder = LabelEncoder()
            y = encoder.fit_transform(local_labels)
            for alpha in ALPHAS:
                if records.get((repeat, alpha), {}).get('status') == 'ok':
                    continue
                record = {'repeat': repeat, 'alpha': alpha, 'seed': seed, 'status': 'error',
                          'n_train': len(train), 'n_test': len(test), 'n_classes': len(encoder.classes_)}
                try:
                    run_dir = directory / f'alpha_{alpha:.2f}'
                    run_dir.mkdir(exist_ok=True)
                    settings = native.settings(engine, subset_path, run_dir, alpha, [40], args.workers, args.time_limit)
                    with engine.ResourceMonitor() as monitor:
                        genes = engine.run_cell_sieve(subset, train, 40, settings, run_dir, subset_path)
                        if len(genes) != 40 or len(set(genes)) != 40 or any(g not in subset.var_names for g in genes):
                            raise ValueError('Expected 40 distinct measured genes')
                        X = engine.as_dense(subset[:, genes].X)
                        scaler = StandardScaler().fit(X[train])
                        model = LogisticRegression(class_weight='balanced', max_iter=2000,
                                  n_jobs=args.workers, random_state=42)
                        model.fit(scaler.transform(X[train]), y[train])
                        predictions = model.predict(scaler.transform(X[test]))
                        score = f1_score(y[test], predictions, average='macro', zero_division=0)
                    pd.DataFrame({'source_row': rows[test], 'cell_id': subset.obs_names[test],
                                  'true': local_labels[test], 'predicted': encoder.inverse_transform(predictions)}).to_csv(run_dir / 'heldout_predictions.csv', index=False)
                    record.update(status='ok', test_macro_f1=float(score),
                                  test_accuracy=float(accuracy_score(y[test], predictions)),
                                  actual_panel_size=40, error='', **monitor.summary())
                except Exception:
                    record['error'] = traceback.format_exc()
                    print(record['error'], flush=True)
                records[(repeat, alpha)] = record
                temporary = result_path.with_suffix('.tmp.csv')
                pd.DataFrame(records.values()).sort_values(['repeat', 'alpha']).to_csv(temporary, index=False)
                temporary.replace(result_path)
                print(f'{tissue} repeat {repeat+1}/{args.repeats}, alpha {alpha}: {record["status"]}', flush=True)
            del subset
    finally:
        atlas.file.close()
    analyze(output, args.repeats, args.permutations, args.seed, tissue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tissue', choices=shared.TISSUES + ('all',), default='CortexHippocampus')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--min-per-class', type=int, default=10)
    parser.add_argument('--test-fraction', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--permutations', type=int, default=19999)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--time-limit', type=float, default=300)
    parser.add_argument('--stage', choices=['all', 'analyze'], default='all')
    parser.add_argument('--out-dir', type=Path, default=shared.ROOT / 'results/supplementary/alpha_subsampling')
    parser.add_argument('--inputs-dir', type=Path, default=shared.ROOT / 'results/supplementary/fullinput_benchmarks')
    parser.add_argument('--data-dir', type=Path, default=Path('/home/igvf-analysis/projects/8cube/processed_data/uci_submitted/annotated'))
    args = parser.parse_args()
    if args.repeats < 10 or args.min_per_class < 5 or not 0 < args.test_fraction < 1 or args.permutations < 999:
        parser.error('Require >=10 repeats, >=5 cells/type, 0<test-fraction<1, and >=999 permutations')
    for tissue in shared.TISSUES if args.tissue == 'all' else [args.tissue]:
        run_tissue(args, tissue)


if __name__ == '__main__':
    main()
