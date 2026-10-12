"""Validate current Figure 1/2 outputs and refresh their audit/documentation.

Reads saved outputs only; no selection, classification, or enrichment is run.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from workspace_paths import ROOT, local_path
from fig1_method_benchmark import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Validate saved outputs without updating documentation or audit')
    args=parser.parse_args()
    results=ROOT/'results/figure1/results'
    pointer=json.loads((results/'latest_cellsieve_batch.json').read_text())
    alpha=float(pointer['alpha'])
    batch=local_path(pointer['batch'])
    meta=json.loads((batch/'rerun_manifest.json').read_text())
    if pointer['status']!='complete' or meta['status']!='complete' or meta['alpha']!=alpha:
        raise ValueError('Complete Figure 1 batch required')
    for cohort in ('full','subsample'):
        run=local_path(pointer[cohort+'_run'])
        provenance=json.loads((run/'execution_provenance.json').read_text())
        argv=provenance['argv']
        assert float(argv[argv.index('--cell-sieve-alpha')+1])==alpha
        t=pd.read_csv(run/'benchmark_summary_all.csv')
        assert len(t)==5 and t.status.eq('ok').all() and set(t.actual_panel_size)=={10,20,40,80,160}
    f2=ROOT/'results/figure2'
    for name,count in [('biology_panel40_fig1',40),('panel50',50)]:
        folder=f2/'results'/name
        provenance=json.loads((folder/'provenance.json').read_text())
        assert provenance['cell_sieve_alpha']==alpha
        assert len(set((folder/'selected_genes.txt').read_text().splitlines()))==count
    refit=f2/'results/merfish_panel50_coefficients'
    provenance=json.loads((refit/'provenance.json').read_text())
    assert provenance['cell_sieve_selection_alpha']==alpha
    assert provenance['selected_genes_sha256']==sha(f2/'results/panel50/selected_genes.txt')
    pred=pd.read_csv(refit/'cv_spatial_predictions.csv')
    metrics=json.loads((refit/'metrics.json').read_text())['cv']
    assert len(pred)==8891
    assert np.isclose((pred.true_class==pred.predicted_class).mean(),metrics['accuracy'])
    assert np.isclose(f1_score(pred.true_class,pred.predicted_class,average='macro',zero_division=0),metrics['macro_f1'])
    for directory,stem in [(ROOT/'results/figure1/exports','Figure1'),
                           (ROOT/'results/figure1/exports','Supp1'),
                           (f2/'exports','Figure2')]:
        figure=json.loads((directory/(stem+'.provenance.json')).read_text())
        assert figure.get('cellsieve_alpha',figure.get('cell_sieve_alpha'))==alpha
        for ext in ('pdf','svg','png'):
            # Figure 1 uses publication_artifacts; Figure 2 uses artifacts.
            hashes=figure.get('artifacts',figure.get('publication_artifacts',{}))
            if ext in hashes and isinstance(hashes[ext],str):
                assert sha(ROOT/'figs'/(stem+'.'+ext))==hashes[ext]
            assert (ROOT/'figs'/(stem+'.'+ext)).stat().st_size>0
    if args.check_only:
        print(f'Figure 1/2 saved outputs validated at alpha={alpha:g}; no files changed.')
        return
    audit=ROOT/'metadata/cell_sieve_alpha_audit.csv'
    with audit.open(newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    violin_genes=json.loads((f2/'results/biology_panel40_fig1/violins/provenance.json').read_text())['genes']
    for row in rows:
        if row['figure_or_analysis'] in ('Figure1','Figure2'):
            row.update(saved_alpha=str(alpha),required_alpha=str(alpha),status='rerun complete',needed_update='None for alpha')
            if row['figure_or_analysis']=='Figure1':
                row['evidence']=str(batch.relative_to(ROOT))+'; results/figure1/exports/Figure1.provenance.json'
            else:
                row['evidence']='results/figure2/exports/Figure2.provenance.json'
                if row['part'].startswith('B-C'):
                    row['part']='B-C '+ ' and '.join(violin_genes)+' violins'
    with audit.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for file in (ROOT/'README.md',ROOT/'results/figure1/README.md'):
        text=file.read_text()
        text=re.sub(r'Current Cell Sieve publication runs use alpha [0-9.]+,',f'Current Cell Sieve publication runs use alpha {alpha:g},',text)
        text=re.sub(r'Current CellSieve points use alpha \*\*[0-9.]+\*\*',f'Current CellSieve points use alpha **{alpha:g}**',text)
        text=re.sub(r'current alpha-[0-9.]+ CellSieve batch',f'current alpha-{alpha:g} CellSieve batch',text)
        text=text.replace('Superseded CellSieve runs, failed alternatives, and unused script snapshots have been deleted.', 'The active run is selected by the validated batch pointer; previous CellSieve runs are preserved for provenance.')
        text=re.sub(r'To replace only Cell Sieve in a real rerun, use `[^`]+`\.', 'To replace only Cell Sieve in a real rerun, use `bash scripts/rerun_figures12.sh --alpha '+f'{alpha:g}`.',text)
        text=text.replace('`python scripts/rerun_cellsieve_alpha07.py`', f'`bash scripts/rerun_figures12.sh --alpha {alpha:g}`')
        file.write_text(text)
    protocol=ROOT/'results/figure1/native/PROTOCOL.md'
    text=protocol.read_text()
    started=meta['started_utc']
    run_date=f'{started[:4]}-{started[4:6]}-{started[6:8]}'
    text=re.sub(r'^\d{4}-\d{2}-\d{2} (?:alpha audit|cortex alpha update):.*$',f'{run_date} cortex alpha update: Figures 1 and 2 use alpha {alpha:g}. Both fixed Figure 1 cohorts and the separate Figure 2 cortex/MERFISH overlap selection were rerun. Comparators and evaluation settings are preserved. Current batch: `{batch.relative_to(ROOT)}`. Figure 3 is unchanged. See `../../metadata/cell_sieve_alpha_audit.csv`.',text,flags=re.MULTILINE)
    text=text.replace('alpha=0.7,',f'alpha={alpha:g},').replace('Cell Sieve_solver_status.csv','CellSieve_solver_status.csv')
    protocol.write_text(text)
    readme=f2/'README.md'
    prior=readme.read_text()
    # Preserve prior detailed biological context as historical documentation.
    historical=batch/'previous_figure2/README.md'
    if not historical.exists():
        historical.parent.mkdir(parents=True,exist_ok=True);historical.write_text(prior)
    marker_section = ('\n## Marker properties and recovery of the 40-gene panel\n' +
                      prior.split('\n## Marker properties and recovery of the 40-gene panel\n', 1)[1]
                      if '\n## Marker properties and recovery of the 40-gene panel\n' in prior else '')
    marker_provenance = f2 / 'results/panel40_marker_audit/provenance.json'
    if marker_section and (not marker_provenance.exists() or
            json.loads(marker_provenance.read_text())['panel_sha256'] !=
            sha(f2 / 'results/biology_panel40_fig1/selected_genes.txt')):
        marker_section = ('\n## Marker properties and recovery of the 40-gene panel\n\n'
                          'The panel changed; rerun `python -B scripts/audit_fig2_panel40_markers.py` '
                          'before interpreting earlier marker categories or recovery ranks.\n')
    rationale = ''
    if violin_genes == ['Npas3', 'Flt1']:
        historical_text = historical.read_text()
        start = historical_text.find('The two violin genes remain')
        end = historical_text.find('## MERFISH refit', start)
        if start >= 0 and end > start:
            rationale = '## Violin-gene context\n\n' + historical_text[start:end].strip() + '\n\n'
            rationale = re.sub(r'The GO results[^\n]*', '', rationale)
    comparison=pd.read_csv(refit/'historical_metric_comparison.csv').set_index('metric')
    table='| Metric | Historical saved | Current refit CV |\n|---|---:|---:|\n'
    for label,key in [('Accuracy','accuracy'),('Macro-F1','macro_f1'),('Balanced accuracy','balanced_accuracy')]:
        row=comparison.loc[key];table+=f'| {label} | {row.historical:.6f} | {row.refit_cv:.6f} |\n'
    readme.write_text(f'''# Figure 2 — cortex biology and MERFISH

Current CellSieve selections use **alpha {alpha:g}**. The 40-gene panel comes from
Figure 1's fixed 86,427-cell, 5,083-gene full-cortex benchmark. The 50-gene panel
is a separate native selection within the 556-gene cortex/MERFISH intersection.
Both use 23 cortex cell types. The results/supplementary 24,170-gene benchmark is a
separate analysis and is not substituted into this figure.

## Panels

- **A:** Native 40-gene cortex classifier coefficients.
- **B–C:** Measured expression of {' and '.join(violin_genes)} across the same cortex cells.
  Both genes belong to A. Original AnnData X is used without further transformation.
- **D:** MERFISH coefficients refitted for the native 50-gene overlap-constrained panel.
- **E–F:** Annotated MERFISH classes and pooled held-out five-fold predictions.

The 24 × 14 inch layout aligns A above D at equal width, with the combined
B–C and E–F blocks sharing their left and right boundaries. A matches the total
B–C plot height; D, E and F have equal plot heights. The spatial maps retain
equal coordinate scaling, with limits padded to fit the aligned axes. A shared
two-row legend spans both maps. Typography follows Figure 1’s Helvetica,
bold title and larger axis-label convention, scaled for the wider canvas. Spatial coordinates and categorical colors are
preserved. F's accuracy and macro-F1 come from saved CV metrics. The native
coefficient examples are retained when selected; missing examples are filled by
maximum absolute native coefficient, as recorded in the violin provenance.

{rationale}## MERFISH evaluation

There are 8,891 eligible cells in nine classes. Cell IDs, order, labels and x/y
coordinates are checked against the preserved reference. The refit uses balanced
L2 logistic regression (C=1, lbfgs, max_iter=2000, tol=1e-4, random_state=42), with
five stratified shuffled folds and StandardScaler fitted within each training
fold. Original log2p expression is not transformed again. A separate fit on all
eligible cells supplies E's coefficients; those training scores are not CV scores.
The historical MERFISH recipe is unavailable, so this is an explicit refit rather
than an exact procedural reproduction or a frozen scRNA-to-MERFISH transfer test.

{table}
## Reproduction

Run from the repository root on the Gurobi-licensed host, using the original
benchmark environment through the launcher:

```bash
# Refit CellSieve for both Figure 1 cohorts and Figure 2's overlap; refit MERFISH.
bash scripts/rerun_figures12.sh --alpha {alpha:g}
# Rebuild measured cortex summaries and render from completed saved selections.
make figure2-panel40 PYTHON=/home/nikki/miniconda3/envs/benchmark_env/bin/python
```

`BENCHMARK_ENV_DIR` overrides the launcher's environment location. `--batch`
resumes validated Figure 1 cohorts. Native limits remain 600 seconds per full
cortex/overlap panel and 300 seconds per Figure 1 subsample panel, with MIP gap
0.10 and four workers. Renderer checks reject alpha or gene-panel mismatches.

Outputs are `figures/Figure2.{{pdf,svg,png}}` and matching provenance.
Active tables are in `results/biology_panel40_fig1`, `results/panel50`, and
`results/merfish_panel50_coefficients`; `results/reference_checks` is unchanged.
The prior figure, tables and README are preserved under
`{batch.relative_to(ROOT)}/previous_figure2/`.

## GO tables

GO enrichment is a separate, explicit step using the current Figure 1 40-gene
panel and frozen GO Consortium mouse annotations in `results/figure2/resources/go/`.
From the repository root, in `benchmark_env`:

```bash
make figure2-go-rerun PYTHON=/home/nikki/miniconda3/envs/benchmark_env/bin/python
# Export an existing, validated result without recomputing enrichment:
make figure2-go PYTHON=/home/nikki/miniconda3/envs/benchmark_env/bin/python
```

The background contains all mouse genes with usable GO annotations across
BP/MF/CC, not the 5,083-gene cortex input or the 556-gene MERFISH overlap. Mapping uses exact symbols, then unambiguous synonyms; NOT and ND
annotations are excluded. Annotations propagate through `is_a` and `part_of`.
One-sided hypergeometric tests include every annotated non-root term, including
zero-hit terms; BH correction uses this full BP/MF/CC family. The manuscript table
retains the same five developmental terms and reports raw P values, not FDR claims.

Results, the LaTeX table, gene mappings, background IDs and provenance are in
`results/biology_panel40_fig1/go_analysis/whole_mouse_genome/`. Provenance records
input/output hashes, actual annotation dates, counts and runtime versions. The
frozen input files and `sources.json` must accompany the result bundle; current
GO download URLs change and are not sufficient to reproduce this snapshot.
Previous-panel results stay in `results/archived_go/`. Panel changes require a
fresh rerun; the table exporter rejects panel or result hash mismatches.
''')
    if marker_section:
        readme.write_text(readme.read_text().rstrip() + '\n' + marker_section)
    print(f'Figures 1/2 alpha={alpha:g}: saved outputs, prediction metrics, audit and documentation validated.',flush=True)


if __name__=='__main__':
    main()
