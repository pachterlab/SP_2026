#!/usr/bin/env python3
"""Shared cohort preparation and figure rendering for the unified benchmark workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'scripts/native/scripts'
TISSUES = ('Adrenal', 'CortexHippocampus', 'DiencephalonPituitary', 'Gastrocnemius',
           'GonadsFemale', 'GonadsMale', 'Heart', 'Kidney', 'Liver')
STRAINS = ('129S1J', 'AJ', 'B6J', 'CASTJ', 'NODJ', 'NZOJ', 'PWKJ', 'WSBJ')
METHODS = ('cell_sieve', 'nsforest', 'scgenefit')
PANELS = (10, 20, 40, 80, 160)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prepare(source, strain, destination, max_genes, min_cells):
    """Read bounded CSR blocks; never load the full atlas expression matrix."""
    import anndata as ad
    from anndata.io import read_elem
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import sparse

    if destination.exists():
        metadata = json.loads(destination.with_suffix('.json').read_text())
        stat = source.stat()
        expected = {'source': str(source), 'source_size': stat.st_size,
                    'source_mtime_ns': stat.st_mtime_ns, 'strain': strain,
                    'max_genes': max_genes, 'min_cells': min_cells}
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Subset configuration changed: use a fresh output directory ({destination})')
        print(f'Reusing {destination}', flush=True)
        return
    with h5py.File(source, 'r') as f:
        obs = read_elem(f['obs'])
        var = read_elem(f['var'])
        labels = obs['celltype'].astype('string')
        keep = obs['Genotype'].astype('string').eq(strain) & labels.notna()
        keep &= ~labels.str.strip().str.lower().isin(['low quality', 'unknown', 'unassigned', 'nan', ''])
        counts = labels[keep].value_counts()
        eligible = counts[counts >= min_cells].index
        keep &= labels.isin(eligible)
        rows = np.flatnonzero(keep.fillna(False).to_numpy(dtype=bool))
        if len(rows) == 0 or len(eligible) < 2:
            raise ValueError(f'{source.name}/{strain}: fewer than two eligible classes')
        names = var['gene_name'].astype(str) if 'gene_name' in var else var.index.astype(str)
        names = np.asarray(names)
        gene_mask = np.array([
            not (g.startswith(('Gm', 'mt-', 'Rps', 'Rpl', 'ENSMUSG')) or g.endswith('Rik')) for g in names])
        columns = np.flatnonzero(gene_mask)
        if len(set(names[columns])) != len(columns):
            raise ValueError('Duplicate gene symbols after name filtering; resolve explicitly')
        parts = []
        for first in range(0, len(obs), 2500):
            last = min(first + 2500, len(obs))
            local = rows[(rows >= first) & (rows < last)] - first
            if not len(local):
                continue
            group = f['X']
            if group.attrs.get('encoding-type') != 'csr_matrix':
                raise ValueError('Expected original CSR X')
            ptr = group['indptr'][first:last + 1]
            start, end = int(ptr[0]), int(ptr[-1])
            block = sparse.csr_matrix((group['data'][start:end], group['indices'][start:end],
                                       ptr - start), shape=(last-first, len(var)))
            parts.append(block[local][:, columns])
        X = sparse.vstack(parts, format='csr')
        if not np.isfinite(X.data).all():
            raise ValueError('Nonfinite original expression values')
        subset_obs = obs.iloc[rows].copy()
        subset_obs['source_row'] = rows
        subset_var = var.iloc[columns].copy()
        subset_var['source_gene_column'] = columns
        subset_var.index = pd.Index(names[columns], name='gene_name')
        full = ad.AnnData(X=X, obs=subset_obs, var=subset_var)
        stat = source.stat()
        full.uns['supplement_provenance'] = {
            'source': str(source), 'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'strain': strain, 'matrix': 'original log1p normalized X; no renormalization',
            'gene_filter': 'exclude Gm/mt-/Rps/Rpl/ENSMUSG prefixes and Rik suffix',
            'min_cells_per_class': min_cells,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        full_path = destination if max_genes == 0 else destination.with_name(destination.stem + '_all_namefiltered_genes.h5ad')
        if max_genes != 0:
            temp = full_path.with_suffix('.tmp.h5ad')
            full.write_h5ad(temp, compression='gzip')
            temp.replace(full_path)
        # Same fixed universe for all three methods within each cohort. Sparse
        # population variance equals the original harness's dense variance rule.
        if max_genes > 0 and X.shape[1] > max_genes:
            variance = np.asarray(X.power(2).mean(axis=0)).ravel() - np.asarray(X.mean(axis=0)).ravel() ** 2
            selected = np.argsort(variance, kind='stable')[-max_genes:]
            selected.sort()
            benchmark = full[:, selected].copy()
        else:
            benchmark = full
        if benchmark.n_vars < max(PANELS):
            raise ValueError(f'Only {benchmark.n_vars} genes; cannot preserve panel grid')
        benchmark.uns['supplement_provenance']['shared_universe'] = 'all name-filtered genes; no variance/HVG filter' if max_genes == 0 else 'highest original-X population variance'
        temp = destination.with_suffix('.tmp.h5ad')
        benchmark.write_h5ad(temp, compression='gzip')
        temp.replace(destination)
        pd.DataFrame({'celltype': counts.index, 'n_cells': counts.values,
                      'retained': counts.index.isin(eligible)}).to_csv(
                          destination.with_suffix('.class_counts.csv'), index=False)
        atomic_json(destination.with_suffix('.json'), {
            'source': str(source), 'source_size': stat.st_size,
            'source_mtime_ns': stat.st_mtime_ns, 'strain': strain,
            'max_genes': max_genes, 'min_cells': min_cells, 'n_cells': benchmark.n_obs,
            'all_namefiltered_genes': max_genes == 0,
            'n_classes': len(eligible), 'n_genes': benchmark.n_vars,
            'all_namefiltered_genes_h5ad': str(full_path),
            'benchmark_h5ad': str(destination), 'benchmark_sha256': digest(destination),
        })
        print(f'Saved {destination.name}: {benchmark.shape}', flush=True)


def render(table, output, cohorts, stem):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from figure_style import set_style
    set_style()
    plt.rcParams.update({'font.size':12, 'axes.titlesize':13, 'axes.labelsize':12,
                         'xtick.labelsize':11, 'ytick.labelsize':11, 'legend.fontsize':12})
    styles = {'cell_sieve': ('CellSieve', '#266a9e'), 'nsforest': ('NS-Forest', '#9467a4'),
              'scgenefit': ('scGeneFit', '#ca7425')}
    metrics = {'macro_f1': 'Refit macro-F1', 'cv_macro_f1': 'Conditional CV macro-F1',
               'accuracy': 'Refit accuracy', 'cv_accuracy': 'Conditional CV accuracy',
               'elapsed_seconds': 'Wall time (seconds)', 'peak_rss_mb': 'Peak process-tree RSS (MB)'}
    output.mkdir(parents=True, exist_ok=True)
    with PdfPages(output / f'{stem}.pdf') as pdf:
        for metric, title in metrics.items():
            fig, axes = plt.subplots(3, 3, figsize=(13, 10))
            for i, (ax, cohort) in enumerate(zip(axes.flat, cohorts)):
                group = table[table['cohort'] == cohort]
                for method, (name, color) in styles.items():
                    d = group[(group.method == method) & (group.status == 'ok')].copy()
                    d = d.sort_values('requested_panel_size').drop_duplicates('requested_panel_size', keep='last')
                    # Keep failed or missing budgets as gaps, never zeros or connected inferred results.
                    values = [d.loc[d.requested_panel_size == k, metric].iloc[0]
                              if (d.requested_panel_size == k).any() else np.nan for k in PANELS]
                    actual = [d.loc[d.requested_panel_size == k, 'actual_panel_size'].iloc[0]
                              if (d.requested_panel_size == k).any() else k for k in PANELS]
                    ax.plot(actual, values, '-o', color=color, label=name, ms=4)
                title_cohort = {'CortexHippocampus':'Cortex / hippocampus',
                                'DiencephalonPituitary':'Diencephalon / pituitary',
                                'GonadsFemale':'Female gonads', 'GonadsMale':'Male gonads'}.get(cohort, cohort)
                if 'selected_alpha' in group:
                    chosen = group.loc[group.method == 'cell_sieve', 'selected_alpha'].dropna().unique()
                    if len(chosen) == 1:
                        title_cohort += rf' · $\alpha={chosen[0]:g}$'
                ax.set_title(title_cohort)
                ax.text(-.13, 1.07, chr(65+i), transform=ax.transAxes, weight='bold', fontsize=14)
                ax.set(xlabel='Selected genes (actual)', ylabel=title, xscale='log')
                ax.set_xticks(PANELS, labels=[str(k) for k in PANELS])
                if metric not in ('elapsed_seconds', 'peak_rss_mb'):
                    ax.set_ylim(0, 1)
                ax.grid(axis='y', alpha=.2)
                ax.spines[['top', 'right']].set_visible(False)
                failed = group[group.status != 'ok']
                if len(failed):
                    ax.text(.02, .02, f'{len(failed)} failed rows; see CSV', transform=ax.transAxes, fontsize=7)
            for ax in list(axes.flat)[len(cohorts):]:
                ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .035))
            heading = 'B6J tissues' if len(cohorts) == 9 else 'Cortex / hippocampus strains'
            fig.suptitle(heading + ' — ' + title, fontsize=18, fontweight='bold')
            fig.tight_layout(rect=(0, .065, 1, .95))
            pdf.savefig(fig)
            for ext in ('png', 'svg'):
                if metric == 'macro_f1':
                    destination = output / f'{stem}.{ext}'
                else:
                    diagnostic = output / 'diagnostics' / stem
                    diagnostic.mkdir(parents=True, exist_ok=True)
                    destination = diagnostic / f'{metric}.{ext}'
                fig.savefig(destination, dpi=350)
            plt.close(fig)



if __name__ == '__main__':
    from fullinput_benchmarks import main
    main()
