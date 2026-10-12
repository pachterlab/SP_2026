"""Export manuscript-referenced CellSieve results; no fitting or analysis."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'supplemental_data'
MIGRATION=json.loads((ROOT/'metadata/layout_migration.json').read_text())['relocations']
INDEX=[]
SOURCES=[]

def read(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))

def resolve(value):
    p=Path(value)
    s=str(p.relative_to(ROOT) if p.is_absolute() else p)
    for old,new in sorted(MIGRATION.items(),key=lambda x:-len(x[0])):
        if s==old or s.startswith(old+'/'):
            s=new+s[len(old):];break
    return ROOT/s

def selected(p, expected=None):
    genes=[x.strip() for x in p.read_text().splitlines() if x.strip()]
    assert len(genes)==len(set(genes)),p
    if expected is not None: assert len(genes)==int(expected),(p,len(genes),expected)
    return '; '.join(genes)

def write(name,data,description,sources):
    assert data,name
    p=OUT/name
    with p.open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in data for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    INDEX.append(dict(file=name,description=description,rows=len(data),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for source in sorted(set(sources)):
        SOURCES.append(dict(table_file=name,source_file=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

def panel_results(source):
    data=[];sources=[source]
    # Keep reported performance/settings; omit machine-specific paths and internal bookkeeping.
    drop={'adata_path','output_dir','original_output_dir','source_summary','error','result_origin','cell_sieve_v2_mode'}
    for r in read(source):
        if r['method']!='cell_sieve':continue
        p=resolve(r['output_dir'])/'selected_genes.txt';sources.append(p)
        data.append({k:v for k,v in r.items() if k not in drop}|{'selected_genes':selected(p,r['actual_panel_size'])})
    return data,sources

OUT.mkdir(exist_ok=True)
f1=ROOT/'results/figure1/exports'
data,sources=panel_results(f1/'Figure1_results.csv')
solver=f1/'CellSieve_solver_status.csv'
statuses={(r['cohort'],r['panel_size']):r for r in read(solver)}
for r in data:
    r['alpha']='0.9'
    s=statuses[(r['cohort'],r['requested_panel_size'])]
    r.update({k:v for k,v in s.items() if k not in {'cohort','panel_size','native_stdout','native_stdout_sha256'}})
write('Table_S1_Figure1_Supp1.csv',data,'Figure 1 / Supp1 CellSieve panel genes, refit and CV scores, resources and solver status.',sources+[solver])

f2=ROOT/'results/figure2/results'
coeff=[];sources=[]
for panel,path,gene_path in [
    ('cortex_40',f2/'biology_panel40_fig1/heatmaps/model_coefficients_by_celltype.csv',f2/'biology_panel40_fig1/selected_genes.txt'),
    ('MERFISH_50',f2/'merfish_panel50_coefficients/model_coefficients.csv',f2/'panel50/selected_genes.txt')]:
    sources.extend([path,gene_path])
    names=selected(gene_path).split('; ')
    for r in read(path):
        label=next(k for k in r if k not in names)
        assert set(names).issubset(r),path
        for i,g in enumerate(names,1):
            coeff.append(dict(panel=panel,alpha='0.9',cell_type=r[label],gene_order=i,gene=g,coefficient=r[g]))
write('Table_S2_Figure2_coefficients.csv',coeff,'Figure 2a,d: both gene panels and their displayed classifier coefficients.',sources)
for file,source,desc in [
    ('Table_S2_Figure2_expression.csv',f2/'biology_panel40_fig1/violins/gene_celltype_summary.csv','Figure 2b,c: Npas3 and Flt1 expression summaries.'),
    ('Table_S2_Figure2_MERFISH_predictions.csv',f2/'merfish_panel50_coefficients/cv_spatial_predictions.csv','Figure 2e,f: held-out predictions, reference labels and spatial coordinates.')]:
    write(file,read(source),desc,[source])
p=f2/'merfish_panel50_coefficients/metrics.json'
metrics=json.loads(p.read_text())
# The metrics file stores the pooled held-out scores used in the figure.
pooled={k:metrics[k] for k in ['n_cells_total','n_cells_evaluated','n_labels_evaluated','n_genes','n_splits']}
pooled.update({'cv_'+k:v for k,v in metrics['cv'].items()})
write('Table_S2_Figure2_MERFISH_metrics.csv',[pooled],'Figure 2 pooled held-out MERFISH scores.',[p])

go=f2/'biology_panel40_fig1/go_analysis/whole_mouse_genome'
p=go/'all_go_results.csv'; shown=go/'selected_developmental_terms.csv'
all_go=read(p);displayed=read(shown)
id_key=next(k for k in displayed[0] if 'GO:' in str(displayed[0][k]))
ids={r[id_key] for r in displayed}
for r in all_go:r['reported_in_manuscript']=str(r[id_key] in ids).lower()
assert sum(r['reported_in_manuscript']=='true' for r in all_go)==5
write('Table_S3_GO.csv',all_go,'All GO tests with adjusted P values; five manuscript terms marked.',[p,shown])

f3=ROOT/'results/figure3'
p=f3/'results/figure3_noise_counts.csv';g=f3/'noise_benchmark/results/cortex_reference/panel80_alpha070/selected_genes.txt'
data=read(p)
for r in data:r.update(alpha='0.7',selected_genes=selected(g,80))
write('Table_S4_Figure3_noise.csv',data,'Figure 3a: noise-recovery counts and 80-gene panel.',[p,g])
p=f3/'results/figure3_unsupervised_screen.csv';data=read(p);sources=[p]
for r in data:
    g=f3/'Zhengmix4eq_single_pass_grid_60genes/cases'/r['case']/'native/round1/panel_size_60_alpha_0.70/selected_genes.txt'
    r.update(alpha='0.7',selected_genes=selected(g,60));sources.append(g)
write('Table_S4_Figure3_screen.csv',data,'Figure 3c,d: 20 initialization results and their gene panels.',sources)
p=f3/'results/figure3_three_pass_metrics.csv';data=read(p);sources=[p]
for r in data:
    r.update(n_pcs=30,n_neighbors=10,resolution='0.6',alpha='0.7',selected_genes='')
    if int(r['round']):
        g=f3/'Zhengmix4eq_three_pass_60genes/native'/f"round{r['round']}"/'panel_size_60_alpha_0.70/selected_genes.txt'
        r['selected_genes']=selected(g,60);sources.append(g)
write('Table_S4_Figure3_passes.csv',data,'Figure 3e: initial state and three refinement passes with gene panels.',sources)
p=f3/'results/figure3_centroid_distances.csv'
write('Table_S4_Figure3_distances.csv',read(p),'Figure 3b: centroid distances for the 1,251 changed unperturbed cells.',[p])

base=ROOT/'results/supplementary/fullinput_benchmarks'
for number,analysis,fig in [(5,'tissues','Supp2'),(6,'strains','Supp3')]:
    data,sources=panel_results(base/f'benchmark_results_{analysis}.csv')
    write(f'Table_S{number}_{fig}.csv',data,f'{fig}: CellSieve scores, resources and genes by cohort and budget.',sources)

base=ROOT/'results/supplementary/alpha_subsampling'
data=[];sources=[]
for d in sorted(base.iterdir()):
    p=d/'results.csv'
    if not p.exists():continue
    sources.append(p)
    for row in read(p):
        candidates=list((d/f"repeat_{int(row['repeat']):03d}"/f"alpha_{float(row['alpha']):.2f}").rglob('selected_genes.txt'))
        assert len(candidates)==1
        g=candidates[0];sources.append(g)
        data.append({'tissue':d.name,**{k:v for k,v in row.items() if k!='error'},'selected_genes':selected(g,row['actual_panel_size'])})
assert len(data)==900
write('Table_S7_Supp4_runs.csv',data,'Supp4: all 900 sensitivity runs, held-out scores and selected genes.',sources)
for suffix,source in [('summary','alpha_sensitivity_summary'),('paired_tests','all_pairwise_tests_bh')]:
    p=base/'summary'/f'{source}.csv'
    write(f'Table_S7_Supp4_{suffix}.csv',read(p),'Supp4: '+suffix.replace('_',' ')+'.',[p])

write('source_manifest.csv',SOURCES.copy(),'Source paths and checksums for retained tables.',[])
# Remove only obsolete exports from the previous generated catalog.
old_index=OUT/'table_index.csv'
if old_index.exists():
    retained={r['file'] for r in INDEX}
    for old in read(old_index):
        if old['file'] not in retained:
            p=OUT/old['file']
            assert p.parent==OUT and p.suffix=='.csv',p
            p.unlink(missing_ok=True)
with old_index.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['file','description','rows','sha256']);w.writeheader();w.writerows(INDEX)
(OUT/'README.md').write_text('''# Supplemental tables

Seven table groups contain the CellSieve results underlying the manuscript and current figures. Each group consists of CSV files. Selected genes appear with each run rather than in separate tables; gene lists preserve saved order and do not imply a ranking.

| Table | Figure / text | Contents |
|---|---|---|
| S1 | Figure 1, Supp1; benchmarking and solver discussion | Performance, conditional CV, resources, solver gaps and panel genes |
| S2 | Figure 2; cortex interpretation and MERFISH transfer | Both panel coefficient matrices in long format, Npas3/Flt1 summaries, pooled metrics and spatial held-out predictions |
| S3 | Main-text GO table and GO methods | All GO tests with raw/adjusted P values; five reported developmental terms marked |
| S4 | Figure 3; annotation refinement | Cortex recovery counts, individual centroid distances, 20 initialization results and three refinement passes, with gene panels |
| S5 | Supp2; tissue benchmark | CellSieve scores, resources and panel genes for nine tissues |
| S6 | Supp3; strain benchmark | CellSieve scores, resources and panel genes for eight strains |
| S7 | Supp4; alpha sensitivity | 900 paired runs with genes, summary statistics and corrected paired tests |

Figure 1 and tissue/strain CV scores are conditional on panels selected on the full cohort. Sensitivity runs report held-out test scores. The initial Figure 3 state has no selected panel. B6J cortex appears in both tissue and strain comparisons.

No models were refitted. Unreferenced marker audits, historical comparisons, intermediate mappings, duplicate summaries and per-cell iterative evaluation files are omitted; their original results remain in `results/`. Comparator outputs remain with the numerical reproduction bundle.

`table_index.csv` lists row counts and checksums. `source_manifest.csv` records provenance and is supporting metadata, not an additional supplemental table. These proposed S1–S7 table numbers are separate from the S1–S4 figure numbers; the supplied manuscript does not yet cite supplemental table numbers.

Regenerate with `python scripts/prepare_supplemental_data.py`. This folder is Git-ignored and ready to archive separately for submission.
''')
print(f'Exported {len(INDEX)-1} data files in seven table groups.')
