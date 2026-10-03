"""Figure 4: defensible binary-detection floor and native CellSieve comparison."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from figure_style import plt, set_style, save

RESULTS = ROOT / 'figure4_theoretical_minimum/results'


def letter(ax, value):
    ax.text(-.105 if value == 'A' else 0, 1.06, value, transform=ax.transAxes, fontsize=17,
            weight='bold', va='bottom', ha='right')


def main():
    set_style()
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    summary = json.loads((RESULTS / 'capacity_summary.json').read_text())
    genes = pd.read_csv(RESULTS / 'binary_detection_gene_capacity.csv')
    depth = pd.read_csv(RESULTS / 'depth_thinning_sensitivity.csv')
    panels = pd.read_csv(RESULTS / 'figure1_cellsieve_panel_comparison.csv')
    if len(genes) != 9004 or set(panels.panel_size) != {10, 20, 40, 80, 160}:
        raise ValueError('Expected native EMBER candidate set and actual Figure 1 CellSieve panels')
    hp = summary['H_P_bits']
    floor = summary['ideal_fixed_binary_coding_floor_genes']
    crossing = summary['additive_envelope_crossing_genes']

    fig = plt.figure(figsize=(14.2, 10.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[.86, 1.0, 1.04],
                           left=.09, right=.94, bottom=.095, top=.955,
                           hspace=.57, wspace=.36)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[2, :])

    # A: the measured partition and the two distinct mathematical statements.
    ax_a.set_title('B6J partition and capacity definition', loc='left', pad=8)
    lines = [
        r'$N=86{,}427$ cells     $G=24{,}170$ genes     $K=23$ cell types',
        rf'$H(P)=-\sum_k \pi_k\log_2\pi_k={hp:.3f}$ bits/cell',
        r'$u_g=h_2(d_g)-\sum_k\pi_k h_2(d_{gk})$  bits/cell',
        r'$U(S)=H(P)-H(P\mid Y_S)$   (joint detection pattern)',
        r'$m_\varepsilon=\min\{|S|:U(S)\geq H(P)-\varepsilon\}$',
        rf'Ideal binary coding floor: $\lceil\log_2 K\rceil={floor}$ genes',
    ]
    for y, line in zip(np.linspace(.91, .12, len(lines)), lines):
        ax_a.text(.025, y, line, transform=ax_a.transAxes, fontsize=11,
                  ha='left', va='center', color='#18374a')
    ax_a.text(.025, .01, 'Exact joint capacity requires gene-dependence data.',
              transform=ax_a.transAxes, fontsize=9.5, ha='left', va='bottom', color='#6a4642')
    ax_a.set_axis_off()

    # B: show what native Psi can and cannot measure beside per-cell detection information.
    sc = ax_b.scatter(genes.Psi_native_100_draw, genes.single_gene_entropy_reduction_bits,
                      c=genes.d_g, cmap='viridis', vmin=0, vmax=1,
                      s=8, alpha=.55, linewidths=0, rasterized=True)
    ax_b.set(xlabel='Native EMBER $\\Psi_g$ (100 draws; p $\\leq$ 0.05)',
             ylabel='Detection entropy reduction $u_g$ (bits/cell)',
             title='EMBER specificity and usable detection information')
    cbar = fig.colorbar(sc, ax=ax_b, fraction=.045, pad=.025)
    cbar.set_label('Observed detection $d_g$', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # C: an upper envelope for joint capacity ONLY under conditional independence.
    end = 10
    curve = genes.iloc[:end]
    ax_c.plot(curve['rank'], curve.additive_envelope_bits, '-o',
              color='#266a9e', lw=2, ms=4, label='Sum of best single-gene $u_g$')
    ax_c.axhline(hp, color='#b14c46', ls='--', lw=1.6, label=f'Partition complexity = {hp:.2f} bits')
    ax_c.axvline(crossing, color='#266a9e', ls=':', lw=1.5)
    ax_c.axvline(floor, color='#302f50', ls='-.', lw=1.5,
                 label=f'Ideal binary coding floor = {floor}')
    ax_c.annotate(f'Additive crossing: {crossing}', xy=(crossing, hp),
                  xytext=(6.1, hp + 1.2), fontsize=9.5,
                  arrowprops={'arrowstyle': '-', 'color': '#266a9e'})
    ax_c.set(xlabel='Top genes ranked by measured $u_g$',
             ylabel='Optimistic additive envelope (bits/cell)',
             title='Capacity threshold under conditional independence',
             xlim=(1, end), ylim=(0, float(curve.additive_envelope_bits.iloc[-1]) * 1.03))
    ax_c.legend(frameon=False, fontsize=8.8, loc='upper left')

    # D: exact thinning expectation for the observed UMI counts, no technology model.
    depth = depth.sort_values('expected_full_gene_umi_depth_per_cell')
    ax_d.plot(depth.expected_full_gene_umi_depth_per_cell,
              depth.additive_envelope_crossing, '-o', color='#ca7425', lw=2, ms=5,
              label='Conditional-independence crossing')
    ax_d.axhline(floor, ls='--', color='#302f50', lw=1.5,
                 label=f'Ideal binary coding floor = {floor}')
    for row in depth.itertuples(index=False):
        ax_d.annotate(f'{row.retained_molecule_fraction_q:g}×',
                      (row.expected_full_gene_umi_depth_per_cell,
                       row.additive_envelope_crossing),
                      xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.5)
    ax_d.set_xscale('log')
    ax_d.set(xlabel='Expected observed UMI depth per cell (all source genes)',
             ylabel='Optimistic crossing (genes)',
             title='Sensitivity to thinning observed UMIs',
             ylim=(3.5, 6.8))
    ax_d.set_yticks([4, 5, 6])
    ax_d.legend(frameon=False, fontsize=9, loc='upper right')

    # E: native CellSieve results are introduced only after the derivation.
    ax_e.plot(panels.panel_size, panels.macro_f1, '-o', color='#266a9e',
              lw=2.2, ms=7, label='Native CellSieve full-cortex macro-F1')
    ax_e.axvline(floor, ls='--', color='#302f50', lw=1.6,
                 label=f'Ideal binary-detection floor: {floor} genes')
    ax_e.axhline(.9, ls=':', color='#b14c46', lw=1.4, label='0.9 macro-F1 reference')
    first = panels.loc[panels.macro_f1.ge(.9)].sort_values('panel_size').iloc[0]
    ax_e.annotate(f'First tested panel $\\geq$0.9: {int(first.panel_size)} genes',
                  xy=(first.panel_size, first.macro_f1), xytext=(30, .66),
                  fontsize=10, arrowprops={'arrowstyle': '->', 'color': '#266a9e'})
    ax_e.set_xscale('log')
    ax_e.xaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 80, 160]))
    ax_e.xaxis.set_major_formatter(FixedFormatter(['5', '10', '20', '40', '80', '160']))
    ax_e.set(xlabel='Number of genes', ylabel='Full-cortex CellSieve macro-F1',
             title='Ideal binary-detection floor beside real Figure 1 CellSieve panels',
             xlim=(4, 190), ylim=(.4, 1.04))
    ax_e.legend(frameon=False, ncol=3, loc='lower right', fontsize=9)
    ax_e.text(.005, -.34,
              'Five genes is a necessary ideal binary-code floor, not an attainable minimum or a bound on count-valued CellSieve. '
              'The 80-gene result is the first tested panel above 0.9.',
              transform=ax_e.transAxes, fontsize=9.3, ha='left', va='top', color='#5b4d4a')

    for ax in (ax_b, ax_c, ax_d, ax_e):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='.88', linewidth=.6)
        ax.set_axisbelow(True)
    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d, ax_e), 'ABCDE'):
        letter(ax, lab)
    save(fig, 'figure4_theoretical_minimum/figures', 'Figure4_B6J_minimum_gene_bound_vs_CellSieve')


if __name__ == '__main__':
    main()
