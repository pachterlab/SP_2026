"""Export the saved whole-mouse GO results as a LaTeX table; no figures or refitting."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/figure2/results/biology_panel40_fig1/go_analysis/whole_mouse_genome'


def tex_escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def probability(value):
    if value < 0.0001:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return rf'\ensuremath{{{mantissa}\times 10^{{{int(exponent)}}}}}'
    return f'{value:.4g}'


def main():
    if not (DEST / 'all_go_results.csv').is_file():
        raise SystemExit('No GO results for the active panel. Previous-panel GO tables are archived under results/archived_go; recompute enrichment before exporting a new table.')
    provenance = json.loads((DEST / 'provenance.json').read_text())
    panel = DEST.parent.parent / 'selected_genes.txt'
    if hashlib.sha256(panel.read_bytes()).hexdigest() != provenance['selected_genes_sha256']:
        raise ValueError('GO results belong to a different selected panel; rerun enrichment')
    results_path = DEST / 'all_go_results.csv'
    if hashlib.sha256(results_path.read_bytes()).hexdigest() != provenance['artifacts']['all_go_results.csv']:
        raise ValueError('GO CSV hash does not match its provenance')
    results = pd.read_csv(results_path)
    N, n = provenance['background_size'], provenance['annotated_panel_size']
    original_terms = ['GO:0007399', 'GO:0048731', 'GO:0048856', 'GO:0007275', 'GO:0032502']
    top = results.set_index('go_id').loc[original_terms].reset_index()
    assert results.background_size.eq(N).all() and results.panel_size.eq(n).all()
    lines = [r'% Requires \usepackage{tabularx}; insert with \input{path/to/whole_genome_go_table.tex}.',
             r'\begin{table}[htbp]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
             r'\renewcommand{\arraystretch}{1.2}',
             rf'\caption{{Selected developmental GO biological processes for the cortex marker panel against the annotated whole-mouse genome ({N:,} genes). Of 40 selected genes, {n} have usable annotations. Values are raw, unadjusted $P$ values.}}',
             r'\label{tab:cortex-panel40-go-whole-genome}',
             r'\begin{tabularx}{\columnwidth}{l>{\raggedright\arraybackslash}p{0.28\columnwidth}r>{\raggedright\arraybackslash}X}', r'\hline',
             r'GO ID & Biological process & Raw $P$ & Selected genes \\', r'\hline']
    for row in top.itertuples():
        genes = ', '.join(r'\textit{' + tex_escape(gene) + '}' for gene in str(row.genes).split(';') if pd.notna(row.genes))
        term = re.sub(r'\bdevelopment(?:al)?\b', lambda match: r'\textbf{' + match.group(0) + '}', tex_escape(row.term))
        lines.append(f'{tex_escape(row.go_id)} & {term} & {probability(row.p_value)} & {genes} '+r'\\')
    lines.extend([r'\hline', r'\end{tabularx}', r'\end{table}'])
    output = DEST / 'whole_genome_go_table.tex'
    output.write_text('\n'.join(lines)+'\n')
    top.to_csv(DEST / 'selected_developmental_terms.csv', index=False)
    provenance['artifacts']['whole_genome_go_table.tex'] = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance['artifacts']['selected_developmental_terms.csv'] = hashlib.sha256((DEST / 'selected_developmental_terms.csv').read_bytes()).hexdigest()
    provenance['export_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
