"""Describe panel40 marker properties and recovery on the Figure 1 gene universe."""
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import false_discovery_control

from build_fig2_panel40 import sha256
from figure_style import ROOT, plt, set_style
from workspace_paths import local_path

DEST = ROOT / 'results/figure2/results/panel40_marker_audit'
PANEL = ROOT / 'results/figure2/results/biology_panel40_fig1/selected_genes.txt'
DATA = ROOT / 'h5ads/B6J_CortexHippocampus_subset.h5ad'


def rank(values, genes, descending=True):
    order = pd.DataFrame({'gene': genes, 'value': values}).sort_values(
        ['value', 'gene'], ascending=[not descending, True], na_position='last')
    return pd.Series(np.arange(1, len(genes)+1), index=order.gene).reindex(genes).to_numpy()


def export_display(selected):
    columns = ['highly_specific', 'original_hvg', 'qualifies_de', 'recovered_specificity40', 'recovered_hvg40', 'recovered_de40']
    selected.sort_values(['category', 'gene']).to_csv(DEST / 'panel40_grouped.csv', index=False)
    summary_columns = columns + ['hvg_top2000', 'de_top40_any_celltype'] + [c for c in selected if c.startswith('saved_')]
    pd.DataFrame([{'criterion': col, 'n_panel_genes': int(selected[col].sum()),
                  'panel_genes': ';'.join(selected.loc[selected[col], 'gene'])} for col in summary_columns]).to_csv(DEST / 'recovery_summary.csv', index=False)
    set_style()
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(selected[columns].to_numpy(dtype=int), cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax.set_yticks(range(len(selected)), selected.gene, fontsize=11)
    ax.set_xticks(range(6), ['Specific', 'Original HVG', 'DE', 'Specificity\ntop 40', 'HVG\ntop 40', 'DE\ntop 40'], fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.axvline(2.5, color='.3', linewidth=1.5)
    for i in range(40):
        for j in range(6):
            if selected.iloc[i][columns[j]]:
                ax.plot(j, i, marker='o', color='white', markersize=4, linestyle='none')
    ax.set_title('Cortex 40-gene panel: marker properties and recovery', pad=45, fontsize=16)
    fig.tight_layout()
    for ext in ['pdf','png','svg']:
        (ROOT / 'figs/diagnostics').mkdir(parents=True, exist_ok=True)
        fig.savefig(ROOT / 'figs/diagnostics' / f'marker_properties.{ext}', dpi=350)
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    panel = PANEL.read_text().splitlines()
    origin = json.loads((PANEL.parent / 'provenance.json').read_text())
    if sha256(DATA) != origin['expression_source_sha256'] or sha256(PANEL) != origin['selected_genes_sha256']:
        raise ValueError('Panel or input differs from saved Figure 2 provenance')
    print('Loading full cortex matrix', flush=True)
    x = ad.read_h5ad(DATA)
    if x.shape != (86427, 5083) or not set(panel).issubset(x.var_names):
        raise ValueError('Unexpected input dimensions or panel')
    genes = x.var_names.astype(str).to_numpy()
    labels = x.obs.celltype.astype(str).to_numpy()
    classes = sorted(set(labels))
    means, detection = [], []
    for celltype in classes:
        block = x.X[labels == celltype]
        means.append(np.asarray(block.mean(axis=0)).ravel())
        detection.append(np.asarray((block > 0).mean(axis=0)).ravel())
    means, detection = np.array(means), np.array(detection)
    saved = pd.read_csv(PANEL.parent / 'heatmaps/mean_expression_by_celltype.csv', index_col=0)
    calculated = pd.DataFrame(means, index=classes, columns=genes)
    if not np.allclose(calculated.loc[saved.index, panel], saved, atol=1e-10):
        raise ValueError('Cell-type means do not reproduce saved panel summaries')
    peak = means.argmax(axis=0)
    sorted_means = np.sort(means, axis=0)
    maxmean, second = sorted_means[-1], sorted_means[-2]
    tau = np.sum(1-means / np.maximum(maxmean, 1e-15), axis=0) / (len(classes)-1)
    tau[maxmean == 0] = 0
    ratio = maxmean / np.maximum(second, 1e-15)
    peak_detection = detection[peak, np.arange(len(genes))]
    specific = (tau >= .85) & (ratio >= 2) & (peak_detection >= .25)
    # Preserve the actual upstream HVG decision as the primary variability label.
    original_hvg = x.var.highly_variable.astype(bool).to_numpy()
    original_dispersion = x.var.dispersions_norm.to_numpy().copy()
    print('Recomputing dispersion rankings on the matched 5,083-gene universe', flush=True)
    sc.pp.highly_variable_genes(x, flavor='seurat', n_top_genes=2000, n_bins=20, subset=False)
    hvg = x.var.dispersions_norm.to_numpy()
    hvg_rank = rank(hvg, genes)
    print('Wilcoxon one-versus-rest markers, all 23 classes and 5,083 genes', flush=True)
    sc.tl.rank_genes_groups(x, groupby='celltype', method='wilcoxon', reference='rest',
        n_genes=x.n_vars, use_raw=False, pts=True, tie_correct=True,
        corr_method='benjamini-hochberg', key_added='marker_audit')
    de = sc.get.rank_genes_groups_df(x, group=None, key='marker_audit')
    de = de.rename(columns={'names': 'gene', 'group': 'celltype', 'pvals': 'p_value',
                            'pvals_adj': 'q_within_celltype', 'logfoldchanges': 'log2fc'})
    # Correct across all 23 x 5,083 comparisons, not just panel genes.
    de['q_global_bh'] = false_discovery_control(de.p_value.to_numpy())
    de['rank_in_celltype'] = de.groupby('celltype', observed=True).cumcount()+1
    de['qualifies_de'] = (de.q_global_bh < .05) & (de.log2fc >= 1) & (de.pct_nz_group >= .25)
    de.to_csv(DEST / 'all_gene_celltype_de.csv.gz', index=False)
    eligible = de[de.qualifies_de].sort_values(['scores', 'gene', 'celltype'], ascending=[False, True, True])
    best_de = eligible.drop_duplicates('gene').set_index('gene')
    best_any = de.sort_values(['scores', 'gene', 'celltype'], ascending=[False, True, True]).drop_duplicates('gene').set_index('gene')
    result = pd.DataFrame({'gene': genes, 'peak_mean_celltype': np.array(classes)[peak],
        'peak_mean_log_expression': maxmean, 'second_highest_mean': second,
        'peak_to_second_mean_ratio': ratio, 'tau_specificity': tau,
        'peak_detection_fraction': peak_detection, 'highly_specific': specific,
        'original_hvg': original_hvg, 'original_normalized_dispersion': original_dispersion,
        'original_dispersion_rank': rank(original_dispersion, genes),
        'recomputed_normalized_dispersion': hvg, 'hvg_rank': hvg_rank,
        'hvg_top2000': hvg_rank <= 2000})
    result['specificity_rank'] = rank(np.where(peak_detection >= .25, tau, -np.inf), genes)
    result['qualifies_de'] = result.gene.isin(best_de.index)
    for source, target in [('celltype','best_de_celltype'), ('scores','best_de_score'),
                           ('log2fc','best_de_log2fc'), ('q_global_bh','best_de_q_global_bh'),
                           ('pct_nz_group','best_de_detection'), ('rank_in_celltype','best_de_rank_in_celltype')]:
        result[target] = result.gene.map(best_de[source])
    result['max_positive_wilcoxon_score'] = result.gene.map(best_any.scores)
    de_scores = result.gene.map(best_de.scores).fillna(-np.inf).to_numpy()
    result['de_global_rank'] = rank(de_scores, genes)
    result['de_top40_any_celltype'] = result.gene.isin(set(eligible[eligible.rank_in_celltype <= 40].gene))
    result['recovered_specificity40'] = (result.specificity_rank <= 40) & (peak_detection >= .25)
    result['recovered_hvg40'] = result.hvg_rank <= 40
    result['recovered_de40'] = (result.de_global_rank <= 40) & result.qualifies_de
    for i, row in result.iterrows():
        flags = [name for name, field in [('specific', 'highly_specific'), ('HVG', 'original_hvg'), ('DE', 'qualifies_de')] if row[field]]
        result.loc[i, 'category'] = ' + '.join(flags) if flags else 'Outside these definitions'
    sources = {}
    comparator = json.loads((ROOT / 'results/figure1/results/native_comparator_batch.json').read_text())
    batch = local_path(comparator['batch']) / 'full'
    for method in ['nsforest', 'scgenefit', 'logistic_regression_de']:
        for budget in [40, 160]:
            path = batch / method / 'B6J_CortexHippocampus_subset' / method / f'panel_size_{budget}/selected_genes.txt'
            if not path.exists():
                raise ValueError(f'Missing saved comparator panel: {path}')
            selected = path.read_text().splitlines()
            result[f'saved_{method}_{budget}'] = result.gene.isin(selected)
            sources[str(path.relative_to(ROOT))] = {'sha256': sha256(path), 'actual_genes': len(selected)}
    for budget in [40, 160]:
        path = ROOT / f'results/figure1/results/l1_reference/panel_size_{budget}/selected_genes.txt'
        result[f'saved_historical_l1_{budget}'] = result.gene.isin(path.read_text().splitlines())
        sources[str(path.relative_to(ROOT))] = {'sha256': sha256(path), 'limitation': 'Historical L1 run has no input checksum'}
    result.to_csv(DEST / 'all_gene_properties.csv', index=False)
    selected = result.set_index('gene').loc[panel].reset_index()
    selected.to_csv(DEST / 'panel40_gene_classification.csv', index=False)
    selected[['gene', 'peak_mean_celltype', 'category', 'highly_specific', 'original_hvg', 'qualifies_de',
        'specificity_rank', 'hvg_rank', 'de_global_rank', 'recovered_specificity40', 'recovered_hvg40', 'recovered_de40']].to_csv(DEST / 'panel40_simple_summary.csv', index=False)
    selected.groupby('category').agg(n_genes=('gene','size'), genes=('gene', lambda s: ';'.join(s))).reset_index().to_csv(DEST / 'category_summary.csv', index=False)
    columns = ['highly_specific', 'original_hvg', 'qualifies_de', 'recovered_specificity40', 'recovered_hvg40', 'recovered_de40']
    pd.DataFrame([{'criterion': col, 'n_panel_genes': int(selected[col].sum()),
                   'panel_genes': ';'.join(selected.loc[selected[col], 'gene'])} for col in columns]).to_csv(DEST / 'recovery_summary.csv', index=False)
    export_display(selected)
    provenance = {'expression_source': str(DATA.relative_to(ROOT)), 'expression_sha256': sha256(DATA),
        'panel_sha256': sha256(PANEL), 'cell_sieve_alpha': origin['cell_sieve_alpha'],
        'n_cells': x.n_obs, 'n_genes': x.n_vars, 'n_celltypes': len(classes),
        'scope': 'Descriptive in-cohort marker audit; all 5,083 candidate genes, no subsampling; not replicate-level inference',
        'specificity': 'Tau over equally weighted class means >=0.85, peak/second mean >=2, peak detection >=0.25',
        'hvg': 'Primary label is the saved upstream highly_variable flag (Seurat); matched-universe ranking recomputed with Scanpy Seurat, 20 bins, top 2000; exact ranks break ties by gene name',
        'de': 'Positive one-vs-rest tie-corrected Wilcoxon on log1p AnnData X; BH across all 23*5083 comparisons; q<0.05, approximate log2FC>=1, detection>=0.25',
        'recovery': 'Global 40-gene lists ranked by tau with detection>=0.25, recomputed normalized dispersion, or maximum qualifying Wilcoxon score. Per-class top40 DE union is separate and has a larger budget.',
        'comparators': sources, 'script_sha256': sha256(Path(__file__)),
        'versions': {'scanpy': sc.__version__, 'anndata': ad.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
        'artifacts': {p.name: sha256(p) for p in DEST.iterdir() if p.is_file() and p.name not in ('provenance.json', 'run.log')}}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(selected.groupby('category').gene.apply(list).to_string(), flush=True)
    print(pd.read_csv(DEST / 'recovery_summary.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true', help='Rebuild display/summary from validated saved properties without recomputing DE')
    args = parser.parse_args()
    if args.render_only:
        provenance_path = DEST / 'provenance.json'
        provenance = json.loads(provenance_path.read_text())
        if sha256(PANEL) != provenance['panel_sha256']:
            raise ValueError('Saved audit belongs to a different panel')
        properties = DEST / 'panel40_gene_classification.csv'
        if sha256(properties) != provenance['artifacts'][properties.name]:
            raise ValueError('Saved properties do not match provenance')
        export_display(pd.read_csv(properties))
        provenance['render_script_sha256'] = sha256(Path(__file__))
        provenance['artifacts'] = {p.name: sha256(p) for p in DEST.iterdir()
            if p.is_file() and p.name not in ('provenance.json', 'run.log')}
        provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    else:
        main()
