"""Validate and summarize completed paired alpha subsamples; never fit models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_8cube_supplements as shared
from figure_style import plt, set_style

NAMES = {'Adrenal': 'Adrenal', 'CortexHippocampus': 'Cortex / hippocampus',
         'DiencephalonPituitary': 'Diencephalon / pituitary', 'Gastrocnemius': 'Gastrocnemius',
         'GonadsFemale': 'Female gonads', 'GonadsMale': 'Male gonads',
         'Heart': 'Heart', 'Kidney': 'Kidney', 'Liver': 'Liver'}
ALPHAS = [.1, .3, .5, .7, .9]


def decorate(ax, summary, tissue):
    ax.errorbar(summary.alpha, summary.mean_test_macro_f1, yerr=summary.sd_test_macro_f1,
                color='#266a9e', marker='o', markersize=5, linewidth=1.7, capsize=3, zorder=3)
    winner = summary.loc[summary.mean_test_macro_f1.idxmax()]
    ax.scatter([winner.alpha], [winner.mean_test_macro_f1], marker='*', s=180,
               color='#dc8c20', edgecolors='#694410', linewidths=.5, zorder=5)
    ax.set(title=NAMES[tissue], ylim=(0, 1), xlim=(.04, .96))
    ax.title.set_fontsize(14)
    ax.title.set_fontweight('bold')
    ax.set_xticks(ALPHAS)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)


def save(fig, folder, stem):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(folder / f'{stem}.{ext}', dpi=350, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def render_tissue(matrix, summary, tissue, folder):
    set_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for row in matrix.to_numpy():
        ax.plot(ALPHAS, row, color='#266a9e', alpha=.15, linewidth=.6)
    decorate(ax, summary, tissue)
    ax.set(xlabel=r'$\alpha$', ylabel='Held-out macro-F1')
    fig.tight_layout()
    destination = shared.ROOT / 'figs/diagnostics/alpha_sensitivity'
    destination.mkdir(parents=True, exist_ok=True)
    save(fig, destination, tissue)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=shared.ROOT / 'results/supplementary/alpha_subsampling')
    args = p.parse_args()
    root = args.out_dir.resolve()
    if not root.is_relative_to(shared.ROOT):
        p.error('Keep outputs inside SP_2026')
    out = root / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    set_style()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    summaries, pairs, decisions = [], [], []
    for ax, tissue in zip(axes.flat, shared.TISSUES):
        folder = root / tissue
        data = pd.read_csv(folder / 'results.csv')
        config = json.loads((folder / 'configuration.json').read_text())
        repeats = config['repeats']
        if len(data) != repeats*5 or not data.status.eq('ok').all() or data[['repeat','alpha']].duplicated().any():
            raise ValueError(f'Incomplete or duplicated runs: {tissue}')
        matrix = data.pivot(index='repeat', columns='alpha', values='test_macro_f1').reindex(index=range(repeats), columns=ALPHAS)
        if matrix.isna().any().any():
            raise ValueError(f'Missing paired scores: {tissue}')
        summary = pd.read_csv(folder / 'alpha_summary.csv').sort_values('alpha').reset_index(drop=True)
        if (not np.allclose(summary.mean_test_macro_f1, matrix.mean()) or
                not np.allclose(summary.sd_test_macro_f1, matrix.std(ddof=1))):
            raise ValueError(f'Summary differs from per-repeat scores: {tissue}')
        summary['highest_mean'] = summary.mean_test_macro_f1.eq(summary.mean_test_macro_f1.max())
        summary['lowest_sd'] = summary.sd_test_macro_f1.eq(summary.sd_test_macro_f1.min())
        summaries.append(summary.assign(tissue=tissue)[['tissue','alpha','mean_test_macro_f1','sd_test_macro_f1','n_repeats','highest_mean','lowest_sd']])
        tests = pd.read_csv(folder / 'paired_alpha_tests.csv').drop(columns=['p_holm'], errors='ignore')
        if len(tests) != 10:
            raise ValueError(f'Expected ten saved raw pairwise tests: {tissue}')
        from scipy.stats import false_discovery_control
        tests['p_bh'] = false_discovery_control(tests.p_value.to_numpy())
        tests.to_csv(folder / 'paired_alpha_tests.csv', index=False)
        winner = float(summary.loc[summary.mean_test_macro_f1.idxmax(), 'alpha'])
        relevant = tests[(tests.alpha_a == winner) | (tests.alpha_b == winner)]
        effect = relevant.mean_macro_f1_difference_a_minus_b * np.where(relevant.alpha_a == winner, 1, -1)
        supported = bool(((effect > 0) & (relevant.p_bh < .05)).all())
        prior = folder / 'analysis.json'
        if not prior.exists():
            prior = folder / 'recommendation.json'
        decision = json.loads(prior.read_text()) if prior.exists() else {}
        decision.update(tissue=tissue, status='complete', best_mean_alpha=winner, selected_alpha=winner,
                        supported_unique_winner=supported,
                        multiplicity='Benjamini-Hochberg over all ten alpha pairs within tissue',
                        selection_rule='highest observed mean held-out macro-F1; exact ties choose lower alpha',
                        reason='empirical sensitivity tuning; not proof of a unique optimum',
                        scope='paired subsamples of one atlas; overlapping cells, not independent animals')
        decision.pop('recommended_alpha', None)
        shared.atomic_json(folder / 'analysis.json', decision)
        decisions.append({'tissue': tissue, 'selected_alpha': winner,
                          'supported_unique_winner': supported, 'n_repeats': repeats})
        pairs.append(tests.assign(tissue=tissue))
        decorate(ax, summary, tissue)
        render_tissue(matrix, summary, tissue, folder)
    fig.suptitle('Alpha sensitivity across tissues', fontsize=20, fontweight='bold', y=.99)
    fig.supxlabel(r'$\alpha$', fontsize=16)
    fig.supylabel('Held-out macro-F1', fontsize=16)
    fig.tight_layout(rect=(.025, .025, 1, .955), h_pad=2, w_pad=2)
    save(fig, shared.ROOT / 'figs', 'Supp4')
    pd.concat(summaries, ignore_index=True).to_csv(out / 'alpha_sensitivity_summary.csv', index=False)
    pd.concat(pairs, ignore_index=True).to_csv(out / 'all_pairwise_tests_bh.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'selected_alphas.csv', index=False)
    shared.atomic_json(out / 'provenance.json', {
        'renderer_sha256': shared.digest(__file__),
        'results_sha256': {t: shared.digest(root / t / 'results.csv') for t in shared.TISSUES},
        'caption': 'Mean ± SD over 20 paired 5,000-cell subsamples. Star marks highest observed mean; not statistical significance. All y axes span 0–1. No emphasis on alpha 0.7.',
        'artifacts': {str(p.relative_to(shared.ROOT)): shared.digest(p) for p in sorted(list(out.iterdir()) + list((shared.ROOT / 'figs').iterdir()))
                      if p.suffix in ('.pdf', '.png', '.svg', '.csv') and p.name in
                      ['Supp4.pdf','Supp4.png','Supp4.svg',
                       'alpha_sensitivity_summary.csv','all_pairwise_tests_bh.csv','selected_alphas.csv']}})
    print('Validated 900 scores; saved summary, nine tissue figures, BH tests and selected alphas.', flush=True)


if __name__ == '__main__':
    main()
