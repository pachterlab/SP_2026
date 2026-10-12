#!/usr/bin/env python3
"""One selected-alpha full-input workflow: arrange saved fits, render, or fit fresh."""
import argparse
import json
import os
import shutil
from pathlib import Path
import pandas as pd
import run_8cube_supplements as shared
import supplementary_helpers as helpers
from workspace_paths import local_path

DEFAULT = shared.ROOT / 'results/supplementary/fullinput_benchmarks'


def selections(sensitivity):
    records = []
    for tissue in shared.TISSUES:
        source = sensitivity / tissue / 'alpha_summary.csv'
        summary = pd.read_csv(source).sort_values('alpha')
        if len(summary) != 5 or not summary.n_repeats.eq(20).all():
            raise ValueError(f'Incomplete sensitivity: {tissue}')
        best = summary.loc[summary.mean_test_macro_f1.idxmax()]
        records.append(dict(analysis='tissues', tissue=tissue, strain='B6J',
            selected_alpha=float(best.alpha), selection_metric='highest mean held-out macro-F1',
            sensitivity_mean_macro_f1=float(best.mean_test_macro_f1),
            sensitivity_sd_macro_f1=float(best.sd_test_macro_f1)))
    records.extend(dict(analysis='strains', tissue='CortexHippocampus', strain=s,
        selected_alpha=.9, selection_metric='specified cortex strain alpha') for s in shared.STRAINS)
    return pd.DataFrame(records)


def link_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    # Mutable tables/manifests get independent copies; large evidence is immutable.
    if source.suffix.lower() in ('.csv', '.json', '.txt', '.log'):
        shutil.copy2(source, destination)
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def copy_tree(source, destination):
    shutil.copytree(source, destination, copy_function=link_file)


def publish(output, choices):
    tables = []
    for choice in choices.itertuples():
        dataset = f'{choice.tissue}_{choice.strain}'
        table = pd.read_csv(output / 'runs' / dataset / 'benchmark_summary.csv')
        expected = {(m,k) for m in shared.METHODS for k in shared.PANELS}
        if len(table) != 15 or not table.status.eq('ok').all() or set(zip(table.method,table.requested_panel_size)) != expected:
            raise ValueError(f'Incomplete cohort: {dataset}')
        table['selected_alpha'] = float('nan')
        table.loc[table.method.eq('cell_sieve'), 'selected_alpha'] = choice.selected_alpha
        table['analysis'] = choice.analysis
        table['cohort'] = choice.tissue if choice.analysis == 'tissues' else choice.strain
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / 'benchmark_results_both.csv', index=False)
    for analysis, group in combined.groupby('analysis'):
        group.to_csv(output / f'benchmark_results_{analysis}.csv', index=False)
    choices.to_csv(output / 'selected_alphas.csv', index=False)
    return combined


def render_saved(output):
    output = Path(output)
    table = pd.read_csv(output / 'benchmark_results_both.csv')
    if len(table) != 255 or not table.status.eq('ok').all():
        raise ValueError('Expected 255 successful rows')
    figures = shared.ROOT / 'figs'
    for analysis, group in table.groupby('analysis'):
        labels = list(shared.TISSUES if analysis == 'tissues' else shared.STRAINS)
        shared.render(group, figures, labels, 'Supp2' if analysis == 'tissues' else 'Supp3')
    shared.atomic_json(output / 'figure_provenance.json', {
        'benchmark_table_sha256': shared.digest(output / 'benchmark_results_both.csv'),
        'renderer_sha256': shared.digest(shared.__file__),
        'artifacts': {p.name: shared.digest(p) for p in sorted(figures.iterdir())
                      if p.stem in ('Supp2','Supp3') and p.suffix in ('.pdf','.png','.svg')},
        'scope': 'CV conditional on full-cohort selection; not nested validation'})


def arrange(args, choices):
    """Copy only evidence supporting the final rows; never load a fitting engine."""
    out = args.out_dir.resolve()
    if out.exists() and any(p.name != '.cache' for p in out.iterdir()):
        raise FileExistsError('Arrange requires a new output folder')
    source = args.import_dir.resolve()
    original = pd.read_csv(source / 'benchmark_results_both.csv')
    if len(original) != 255 or not original.status.eq('ok').all():
        raise ValueError('Incomplete import table')
    shared.atomic_json(out/'configuration.json', dict(
        mode='selected_alpha_fullinput', panels=list(shared.PANELS), methods=list(shared.METHODS),
        workers=4, time_limit=300, seed=42, cv_splits=3, all_namefiltered_genes=True,
        selections=choices.fillna('').to_dict('records')))
    manifest = []
    for choice in choices.drop_duplicates(['tissue','strain']).itertuples():
        dataset = f'{choice.tissue}_{choice.strain}'
        group = original[(original.analysis == choice.analysis) & (original.cohort == (choice.tissue if choice.analysis=='tissues' else choice.strain))].copy()
        cellsieve = group[group.method.eq('cell_sieve')]
        if not cellsieve.selected_alpha.eq(choice.selected_alpha).all():
            raise ValueError(f'Wrong alpha: {dataset}')
        cohort = out/'runs'/dataset
        cohort.mkdir(parents=True)
        input_source = local_path(group.iloc[0].adata_path)
        link_file(input_source, out/'h5ads'/input_source.name)
        metadata = json.loads(input_source.with_suffix('.json').read_text())
        metadata.update(benchmark_h5ad=str(out/'h5ads'/input_source.name), all_namefiltered_genes_h5ad=str(out/'h5ads'/input_source.name))
        shared.atomic_json((out/'h5ads'/input_source.name).with_suffix('.json'), metadata)
        replacements = {str(input_source):str(out/'h5ads'/input_source.name)}
        for method, rows in group.groupby('method'):
            old = local_path(rows.iloc[0].output_dir).parent
            copy_tree(old, cohort/method)
            replacements[str(Path(rows.iloc[0].output_dir).parent)] = str(cohort/method)
        old_native = local_path(cellsieve.iloc[0].output_dir).parent.parent
        for name in ['cell_sieve_package_runs','cell_sieve_package_train.h5ad']:
            original_path = old_native/name
            if original_path.is_dir():
                copy_tree(original_path, cohort/name)
            elif original_path.exists():
                link_file(original_path, cohort/name)
        native = cohort/'cell_sieve_package_runs'
        if any(p.name.startswith('panel_size_') and '_alpha_' in p.name and not p.name.endswith(f'_alpha_{choice.selected_alpha:.2f}') for p in native.rglob('*') if p.is_dir()):
            raise ValueError(f'Unexpected native alpha: {dataset}')
        manifest.extend(dict(dataset=dataset, method=r.method, panel_size=int(r.requested_panel_size),
            original_output=r.output_dir, original_input=r.adata_path,
            original_source_summary=r.source_summary if isinstance(r.source_summary,str) else '',
            selected_alpha=choice.selected_alpha if r.method=='cell_sieve' else None) for r in group.itertuples())
        group['adata_path'] = str(out/'h5ads'/input_source.name)
        group['source_summary'] = str(cohort/'benchmark_summary.csv')
        for old,new in replacements.items():
            group['output_dir'] = group.output_dir.str.replace(old,new,regex=False)
        group = group.drop(columns=['analysis','cohort'])
        group.to_csv(cohort/'benchmark_summary.csv',index=False)
        group.to_csv(cohort/'benchmark_summary_partial.csv',index=False)
    shared.atomic_json(out/'import_provenance.json', {'operation':'arrangement only; no fits rerun', 'source_table_sha256':shared.digest(source/'benchmark_results_both.csv'), 'rows':manifest})
    combined = publish(out,choices)
    # Every measurement and runtime is unchanged, including the duplicated B6J display rows.
    keys = ['analysis','cohort','method','requested_panel_size']
    excluded = set(keys+['adata_path','output_dir','source_summary'])
    pd.testing.assert_frame_equal(original.set_index(keys).sort_index().drop(columns=list(excluded-set(keys)), errors='ignore'),
        combined.set_index(keys).sort_index().drop(columns=list(excluded-set(keys)), errors='ignore'), check_exact=False,rtol=1e-12,atol=1e-12)
    print('Arranged 16 unique cohorts / 255 display rows without fitting.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['arrange','plot','run'],default='plot')
    p.add_argument('--out-dir',type=Path,default=DEFAULT)
    p.add_argument('--sensitivity-dir',type=Path,default=shared.ROOT/'results/supplementary/alpha_subsampling')
    p.add_argument('--import-dir',type=Path,default=DEFAULT)
    p.add_argument('--data-dir',type=Path,default=Path('/home/igvf-analysis/projects/8cube/processed_data/uci_submitted/annotated'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--time-limit',type=float,default=300)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    out=args.out_dir.resolve()
    if not out.is_relative_to(shared.ROOT):
        p.error('Output must remain inside SP_2026')
    if args.stage=='plot':
        render_saved(out)
        return
    choices=selections(args.sensitivity_dir)
    if args.dry_run:
        print(choices.to_string(index=False))
        return
    if args.stage=='arrange':
        arrange(args,choices)
        return
    engine=helpers.engine_module()
    helpers.inside_workspace(out)
    configuration=dict(mode='selected_alpha_fullinput',panels=list(shared.PANELS), methods=list(shared.METHODS),
        workers=args.workers,time_limit=args.time_limit,seed=42,cv_splits=3,all_namefiltered_genes=True,
        selections=choices.fillna('').to_dict('records'))
    helpers.locked_config(out,configuration)
    for choice in choices.drop_duplicates(['tissue','strain']).itertuples():
        subset=out/'h5ads'/f'{choice.tissue}_{choice.strain}.h5ad'
        helpers.prepare_full_input(out,args.data_dir,choice.tissue,choice.strain,subset)
        settings=helpers.settings(engine,subset,out/'runs',choice.selected_alpha,shared.PANELS,args.workers,args.time_limit)
        settings.methods=','.join(shared.METHODS)
        engine.run_dataset(subset,settings)
    publish(out,choices)
    render_saved(out)


if __name__=='__main__':
    main()
