"""Render Figure 2 from saved cortex panels and the documented MERFISH refit."""

import json

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
    ax.set_yticks(range(len(frame)), [celltype_label(x) for x in frame.index], fontsize=13)
    ax.tick_params(length=3, pad=2)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_y(-.18 if i % 2 else 0)
    ax.set_title(title, fontsize=20, pad=10)
    return im


def main():
    set_style()
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
    })
    base = ROOT / 'figure2_cortex_merfish/results'
    biology = base / 'biology_panel40_fig1'
    genes = (biology / 'selected_genes.txt').read_text().splitlines()
    expr = pd.read_csv(biology / 'heatmaps/mean_expression_by_celltype_gene_zscore.csv', index_col=0)
    coeff = pd.read_csv(biology / 'heatmaps/model_coefficients_by_celltype.csv', index_col=0)
    if len(genes) != 40 or expr.shape != coeff.shape or expr.columns.tolist() != genes or coeff.columns.tolist() != genes:
        raise ValueError('Cortex coefficients and violin source summaries must match the native 40-gene panel')
    refit_dir = base / 'merfish_panel50_coefficients'
    spatial_path = refit_dir / 'cv_spatial_predictions.csv'
    spatial = pd.read_csv(spatial_path)
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
    violin_genes = ['Npas3', 'Flt1']
    if not set(violin_genes).issubset(genes) or set(violin_data.celltype) != set(expr.index):
        raise ValueError('Violin genes and cell types must match cortex coefficients')

    fig = plt.figure(figsize=(18, 14))
    # Larger typography needs taller heatmaps and two tiers of gene labels.
    ax40 = fig.add_axes([.145, .635, .30, .315])
    violins = [fig.add_axes([.54, .84, .45, .11]),
               fig.add_axes([.54, .635, .45, .125])]
    heatmaps = [fig.add_axes([.145, .145, .25, .305]),
                fig.add_axes([.50, .145, .24, .305])]
    maps = [fig.add_axes([.83, .285, .16, .165]),
            fig.add_axes([.83, .055, .16, .165])]
    coefficient_bound = max(np.abs(cortex50.to_numpy()).max(),
                            np.abs(merfish50.to_numpy()).max())
    for ax, frame, title in (
        (ax40, coeff, 'Cortex · 40 genes'),
        (heatmaps[0], cortex50, '8cube cortex · 50 genes'),
        (heatmaps[1], merfish50, 'MERFISH refit · 50 genes'),
    ):
        im = matrix(ax, frame, title, ticks=12,
                    bound=None if ax is ax40 else coefficient_bound)
        if ax is ax40:
            bar = fig.colorbar(im, ax=ax, fraction=.018, pad=.012, shrink=.9)
        elif ax is heatmaps[1]:
            # D/E use identical limits and one shared scale to save space.
            bar = fig.colorbar(im, cax=fig.add_axes([.752, .17, .010, .255]))
        else:
            continue
        bar.set_label('Coefficient', fontsize=16, labelpad=5)
        bar.ax.tick_params(labelsize=13, length=3, pad=3)

    cats = sorted(set(spatial.true_class) | set(spatial.predicted_class))
    # Categorical greens, purples, gold, brown and charcoal stay distinct
    # from the red/blue coefficient heatmaps. Shared by F, G and the legend.
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
        (maps[0], 'true_class', 'MERFISH\nAnnotated classes'),
        (maps[1], 'predicted_class',
         f"MERFISH refit · 5-fold\nAccuracy {refit_metrics['accuracy']:.2%}\nMacro-F1 {refit_metrics['macro_f1']:.4f}")
    ):
        ax.scatter(spatial.x, spatial.y, c=[colors[cat] for cat in spatial[column]],
                   s=2.1, linewidths=0, rasterized=True)
        ax.set(xlabel='Spatial x', ylabel='Spatial y', title=title, aspect='equal')
        ax.set_title(title, fontsize=16, pad=9)
        ax.tick_params(labelsize=14, length=3)
        ax.invert_yaxis()
        ax.set_anchor('N')
    maps[0].set_xlabel('')
    maps[0].tick_params(axis='x', labelbottom=False)
    fig.legend(
        handles=[Line2D([], [], ls='', marker='o', color=colors[cat], label=cat, markersize=8)
                 for cat in cats],
        loc='upper center', bbox_to_anchor=(.57, .005), ncol=5, frameon=False, fontsize=13,
        handletextpad=.3, labelspacing=.35, borderaxespad=0
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
        ax.set_xticks(np.arange(len(groups)), [celltype_label(x) for x in expr.index], rotation=45, ha='right', fontsize=13)
        ax.set(xlim=(-.6, len(groups)-.4), ylabel='Expression')
        ax.set_title(f'{gene} · cortex expression', fontsize=20, pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=1)
        ax.grid(axis='y', color='.92', linewidth=.5)
        ax.set_axisbelow(True)

    violins[0].tick_params(axis='x', labelbottom=False)

    for ax, letter in zip([ax40] + violins + heatmaps + maps, 'ABCDEFG'):
        ax.text(-.04 if ax in [ax40] + violins else -.10, 1.025, letter,
                transform=ax.transAxes, weight='bold', fontsize=26,
                ha='right', va='bottom')
    save(fig, 'figure2_cortex_merfish/figures', 'Figure2_cortex_merfish')



if __name__ == '__main__':
    main()
