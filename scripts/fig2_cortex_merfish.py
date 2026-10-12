"""Render Figure 2 from saved cortex panels and the documented MERFISH refit."""

import json
import hashlib

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figure_style import ROOT, plt, set_style, save


def celltype_label(value):
    """Compact display labels; stored cell-type identifiers are unchanged."""
    return (str(value).replace('GABAergic cortical interneuron', 'GABA interneuron')
            .replace('GABAergic interneuron', 'GABA interneuron')
            .replace('glutamatergic neuron', 'Glut neuron')
            .replace('committed oligodendrocyte precursor', 'committed OPC')
            .replace('oligodendrocyte precursor cell', 'OPC')
            .replace('vascular leptomeningeal cell', 'vascular leptomeningeal')
            .replace('hippocampal CA1-3 neuron', 'CA1–3 neuron')
            .replace('choroid plexus epithelial cell', 'choroid plexus epithelium'))


def matrix(ax, frame, title, *, diverging=True, ticks=7, bound=None):
    values = frame.to_numpy()
    if bound is None:
        bound = float(np.nanmax(np.abs(values))) if diverging else 1.0
    im = ax.imshow(
        values, aspect='auto', cmap='RdBu_r' if diverging else 'Blues',
        vmin=-bound if diverging else 0, vmax=bound, interpolation='nearest'
    )
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=90, fontsize=ticks)
    ax.set_yticks(range(len(frame)), [celltype_label(x) for x in frame.index], fontsize=16)
    ax.tick_params(length=3, pad=2)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_y(-.30 if i % 2 else 0)
    ax.set_title(title, fontsize=24, fontweight='bold', pad=12)
    return im


def main():
    set_style()
    plt.rcParams.update({
        "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
        "legend.fontsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })
    base = ROOT / 'results/figure2/results'
    biology = base / 'biology_panel40_fig1'
    selections = {
        'cortex_panel40': json.loads((biology / 'provenance.json').read_text()),
        'overlap_panel50': json.loads((base / 'panel50/provenance.json').read_text()),
    }
    pointer = json.loads((ROOT / 'results/figure1/results/latest_cellsieve_batch.json').read_text())
    alpha = float(pointer['alpha'])
    if pointer.get('status') != 'complete' or any(p.get('cell_sieve_alpha') != alpha for p in selections.values()):
        raise ValueError('Figure 2 selections must match the active Figure 1 alpha')
    genes = (biology / 'selected_genes.txt').read_text().splitlines()
    expr = pd.read_csv(biology / 'heatmaps/mean_expression_by_celltype_gene_zscore.csv', index_col=0)
    coeff = pd.read_csv(biology / 'heatmaps/model_coefficients_by_celltype.csv', index_col=0)
    if len(genes) != 40 or expr.shape != coeff.shape or expr.columns.tolist() != genes or coeff.columns.tolist() != genes:
        raise ValueError('Cortex coefficients and violin source summaries must match the native 40-gene panel')
    refit_dir = base / 'merfish_panel50_coefficients'
    spatial_path = refit_dir / 'cv_spatial_predictions.csv'
    spatial = pd.read_csv(spatial_path)
    refit_provenance = json.loads((refit_dir / 'provenance.json').read_text())
    if (refit_provenance.get('cell_sieve_selection_alpha') != alpha or
            refit_provenance['selected_genes_sha256'] != hashlib.sha256((base / 'panel50/selected_genes.txt').read_bytes()).hexdigest()):
        raise ValueError('MERFISH predictions must be refitted for the current selected panel')
    refit_metrics = json.loads((refit_dir / 'metrics.json').read_text())['cv']
    genes50 = (base / 'panel50/selected_genes.txt').read_text().splitlines()
    cortex50 = pd.read_csv(base / 'panel50/model_coefficients.csv', index_col=0)
    if len(genes50) != 50 or set(cortex50.columns) != set(genes50):
        raise ValueError('Expected the saved 50-gene 8cube model')
    cortex50 = cortex50.loc[:, genes50]
    merfish_path = base / 'merfish_panel50_coefficients/model_coefficients.csv'
    merfish50 = pd.read_csv(merfish_path, index_col=0)
    if set(merfish50.columns) != set(genes50):
        raise ValueError('MERFISH heatmap must contain the same 50 genes as the 8cube heatmap')
    merfish50 = merfish50.loc[:, genes50]
    if set(merfish50.index) != set(spatial.true_class):
        raise ValueError('MERFISH coefficients and spatial predictions must use the same classes')
    violin_data = pd.read_csv(biology / 'violins/expression_by_cell.csv.gz')
    violin_genes = json.loads((biology / 'violins/provenance.json').read_text())['genes']
    if not set(violin_genes).issubset(genes) or set(violin_data.celltype) != set(expr.index):
        raise ValueError('Violin genes and cell types must match cortex coefficients')

    fig = plt.figure(figsize=(24, 14))
    # Shared column boundaries and row heights keep the six-panel grid aligned.
    left, left_width = .12, .32
    right, right_width, map_gap = .54, .45, .04
    top_bottom, top_height = .635, .315
    bottom, bottom_height = .105, .28
    ax40 = fig.add_axes([left, top_bottom, left_width, top_height])
    violins = [fig.add_axes([right, .82, right_width, .13]),
               fig.add_axes([right, top_bottom, right_width, .145])]
    heatmaps = [fig.add_axes([left, bottom, left_width, bottom_height])]
    map_width = (right_width - map_gap) / 2
    maps = [fig.add_axes([right, bottom, map_width, bottom_height]),
            fig.add_axes([right + map_width + map_gap, bottom, map_width, bottom_height])]
    coefficient_bound = max(np.abs(cortex50.to_numpy()).max(),
                            np.abs(merfish50.to_numpy()).max())
    for ax, frame, title in (
        (ax40, coeff, '40 gene panel'),
        (heatmaps[0], merfish50, '50 gene panel in MERFISH'),
    ):
        im = matrix(ax, frame, title, ticks=16,
                    bound=None if ax is ax40 else coefficient_bound)
        if ax is ax40:
            bar = fig.colorbar(im, cax=fig.add_axes([.452, .66, .008, .265]))
        else:
            # Preserve the saved coefficient limits while omitting the snRNAseq panel.
            bar = fig.colorbar(im, cax=fig.add_axes([.452, .125, .008, .24]))
        bar.set_label('Coefficient', fontsize=22, labelpad=5)
        bar.ax.tick_params(labelsize=16, length=3, pad=3)

    cats = sorted(set(spatial.true_class) | set(spatial.predicted_class))
    # Categorical greens, purples, gold, brown and charcoal stay distinct
    # from the red/blue coefficient heatmaps. Shared by E, F and the legend.
    colors = {
        '01 IT-ET Glut': '#238b45',
        '02 NP-CT-L6b Glut': '#d4a000',
        '06 CTX-CGE GABA': '#b39bd4',
        '07 CTX-MGE GABA': '#6f2385',
        '08 CNU-MGE GABA': '#333333',
        '30 Astro-Epen': '#a5bf39',
        '31 OPC-Oligo': '#78532b',
        '33 Vascular': '#bf5ba5',
        '34 Immune': '#596523',
    }
    if set(cats) != set(colors):
        raise ValueError('MERFISH class palette must match the plotted classes')
    for ax, column, title in (
        (maps[0], 'true_class', '1122 annotation genes'),
        (maps[1], 'predicted_class',
         '50 annotation genes')
    ):
        ax.scatter(spatial.x, spatial.y, c=[colors[cat] for cat in spatial[column]],
                   s=2.1, linewidths=0, rasterized=True)
        ax.set(xlabel='Spatial x', ylabel='Spatial y', title=title, aspect='equal')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(title, fontsize=24, fontweight='bold', pad=32)
        ax.tick_params(labelsize=16, length=3)
        ax.invert_yaxis()
        ax.set_anchor('N')
    # A common title offset aligns D–F above the shared lower-row boundary.
    heatmaps[0].set_title('50 gene panel in MERFISH', fontsize=24,
                          fontweight='bold', pad=32)
    maps[1].text(
        .5, 1.025,
        f"Accuracy {refit_metrics['accuracy']:.4f} · Macro-F1 {refit_metrics['macro_f1']:.4f}",
        transform=maps[1].transAxes, ha='center', va='bottom', fontsize=16
    )
    fig.legend(
        handles=[Line2D([], [], ls='', marker='o', color=colors[cat], label=cat, markersize=7)
                 for cat in cats],
        loc='upper left', bbox_to_anchor=(right, .03, right_width, .015),
        mode='expand', ncol=5, frameon=False, fontsize=15,
        handletextpad=.3, columnspacing=.8, labelspacing=.5, borderaxespad=0
    )

    for ax, gene, color in zip(violins, violin_genes, ['#9b9b9b', '#9b9b9b']):
        groups = [violin_data.loc[violin_data.celltype == celltype, gene].to_numpy()
                  for celltype in expr.index]
        parts = ax.violinplot(groups, positions=np.arange(len(groups)), widths=.85,
                              showextrema=False, showmedians=True, points=100)
        for body in parts['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.75)
        parts['cmedians'].set_color('#222222')
        parts['cmedians'].set_linewidth(.8)
        ax.set_xticks(np.arange(len(groups)), [celltype_label(x) for x in expr.index], rotation=45, ha='right', fontsize=16)
        ax.set(xlim=(-.6, len(groups)-.4), ylabel='Expression')
        ax.set_title(f'{gene} expression', fontsize=24, fontweight='bold', pad=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=1)
        ax.grid(axis='y', color='.92', linewidth=.5)
        ax.set_axisbelow(True)

    violins[0].tick_params(axis='x', labelbottom=False)

    for ax, letter in zip([ax40] + violins + heatmaps + maps, 'ABCDEF'):
        ax.text(-.04 if ax in [ax40] + violins else -.10, 1.025, letter,
                transform=ax.transAxes, weight='bold', fontsize=30,
                ha='right', va='bottom')
    fig.canvas.draw()
    positions = {name: ax.get_position().bounds for name, ax in
                 zip('ABCDEF', [ax40] + violins + heatmaps + maps)}
    if not (np.isclose(positions['A'][2], positions['D'][2])
            and np.isclose(positions['B'][2], positions['E'][2] + map_gap + positions['F'][2])
            and np.isclose(positions['A'][3], positions['B'][1] + positions['B'][3] - positions['C'][1])
            and all(np.isclose(positions['D'][3], positions[k][3]) for k in 'EF')):
        raise ValueError('Figure 2 panel dimensions do not match the requested grid')
    save(fig, 'results/figure2/exports', 'Figure2')
    directory = ROOT / 'results/figure2/exports'
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {'cell_sieve_alpha': alpha, 'selections': selections,
        'merfish_refit': json.loads((refit_dir / 'provenance.json').read_text()),
        'merfish_cv_metrics': refit_metrics,
        'violin_genes': violin_genes,
        'panel_bounds': positions, 'figure_size_inches': [24, 14],
        'pdf_pages': ['A–F composite; MERFISH heatmap and spatial maps side by side'],
        'artifacts': {ext: hashlib.sha256((ROOT / 'figs' / ('Figure2.' + ext)).read_bytes()).hexdigest()
                      for ext in ('pdf', 'svg', 'png')}}
    (directory / 'Figure2.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')



if __name__ == '__main__':
    main()
