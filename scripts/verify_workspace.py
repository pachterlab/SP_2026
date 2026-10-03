"""Integrity and saved-output consistency checks only; stdlib, no model imports."""
from pathlib import Path
import ast,csv,json,hashlib,math,tokenize,sys,subprocess
from workspace_paths import local_path
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checks={};manifest=rows(ROOT/'archive_notes/copy_manifest.csv')
    retired=set(json.loads((ROOT/'archive_notes/retired_figure1_panels.json').read_text())['paths'])
    assert len(retired)==5 and all(not (ROOT/path).exists() for path in retired), 'Retired Figure 1 panels unexpectedly present'
    copied_retired={r['destination_path'] for r in manifest if r['destination_path'] in retired}
    assert len(copied_retired)==2, 'Unexpected retired copied-file inventory'
    cleanup=json.loads((ROOT/'figure3_annotation/results/figure3_cleanup_manifest.json').read_text())
    retired_figure3=set(cleanup['removed'])
    retired_figure3.update('figure3_annotation/'+p for p in cleanup.get('removed_unused_analyses',[]))
    def figure3_removed(path):
        return any(path==p or path.startswith(p+'/') for p in retired_figure3)
    active_manifest=[r for r in manifest if r['destination_path'] not in retired
                     and not figure3_removed(r['destination_path'])]
    checks['retired_figure3_copied_files']=sum(figure3_removed(r['destination_path']) for r in manifest)
    checks['retired_figure1_panel_exports']=len(retired)
    for r in active_manifest:
        p=ROOT/r['destination_path'];assert p.is_file(),str(p)
        assert sha(p)==r['sha256'],'Copied file changed: '+str(p)
    checks['copied_files_verified']=len(active_manifest);checks['copied_bytes']=sum(int(r['bytes']) for r in active_manifest)
    preserved=json.loads((ROOT/'archive_notes/self_contained_inputs.json').read_text())['files']
    for item in preserved:
        p=ROOT/item['local']
        assert p.is_file() and p.stat().st_size==item['bytes'],str(p)
        assert sha(p)==item['sha256'],'Preserved input changed: '+str(p)
    checks['self_contained_inputs_verified']=len(preserved)
    relocated=json.loads((ROOT/'h5ads/data_locations.json').read_text())['files']
    for item in relocated:
        p=ROOT/item['local']; link=ROOT/item['original']
        assert link.is_symlink() and link.resolve()==p.resolve(),str(link)
        assert p.is_file() and p.stat().st_size==item['bytes'],str(p)
        assert sha(p)==item['sha256'],'Relocated dataset changed: '+str(p)
    checks['centralized_datasets_verified']=len(relocated)
    checks['source_inventory']='Historical inventory retained; verification reads local copies only.'
    for r in rows(ROOT/'archive_notes/preexisting_target_inventory.csv'):
        p=ROOT/r['path'];st=p.stat()
        assert (st.st_size,st.st_mtime_ns)==(int(r['bytes']),int(r['mtime_ns'])),'Pre-existing target changed: '+str(p)
    checks['preexisting_target_metadata_unchanged']=True
    parsed=[];vendor_encoding_issues=[]
    for p in (ROOT/'scripts').glob('*.py'):
        with tokenize.open(p) as f:ast.parse(f.read(),filename=str(p))
        parsed.append(str(p.relative_to(ROOT)))
    for r in active_manifest:
        p=ROOT/r['destination_path']
        if p.suffix=='.py':
            try:
                with tokenize.open(p) as f:ast.parse(f.read(),filename=str(p))
            except UnicodeDecodeError as exc:
                if '/Benchmark/external/' not in str(p):raise
                vendor_encoding_issues.append({'path':str(p.relative_to(ROOT)),'issue':str(exc)})
                continue
            parsed.append(str(p.relative_to(ROOT)))
    checks['scripts_parsed_without_execution']=len(parsed)
    checks['preserved_vendor_encoding_issues']=vendor_encoding_issues
    br=rows(ROOT/'archive_notes/benchmark_row_checks.csv')
    for r in br:
        d=ROOT/r['copied_output_dir']
        summary=rows(local_path(r['source_summary']))[int(r['row'])-2]
        if r['status']!='ok':
            assert r['check']=='RECORDED FAILURE';continue
        cv=json.loads((d/'cv_metrics.json').read_text());report=json.loads((d/'classification_report.json').read_text())
        for key in ['cv_accuracy','cv_macro_f1']:assert math.isclose(float(summary[key]),cv[key],abs_tol=1e-10)
        assert math.isclose(float(summary['accuracy']),report['accuracy'],abs_tol=1e-10)
        assert math.isclose(float(summary['macro_f1']),report['macro avg']['f1-score'],abs_tol=1e-10)
        assert len((d/'selected_genes.txt').read_text().splitlines())==int(summary['actual_panel_size'])
    checks['benchmark_successful_rows_verified']=sum(r['status']=='ok' for r in br)
    checks['benchmark_failure_rows_preserved']=sum(r['status']!='ok' for r in br)
    base=ROOT/'figure2_cortex_merfish/results';m=base/'merfish_panel50_cv'
    metrics=json.loads((m/'class_min10_metrics.json').read_text());pred=rows(m/'class_min10_spatial_predictions.csv')
    acc=sum(r['true_class']==r['predicted_class'] for r in pred)/len(pred)
    labels=sorted({r['true_class'] for r in pred}|{r['predicted_class'] for r in pred});fs=[]
    for l in labels:
        tp=sum(r['true_class']==l and r['predicted_class']==l for r in pred)
        den=sum(r['true_class']==l for r in pred)+sum(r['predicted_class']==l for r in pred)
        fs.append(2*tp/den)
    f1=sum(fs)/len(fs)
    assert math.isclose(acc,metrics['accuracy'],abs_tol=1e-12)
    assert math.isclose(f1,metrics['macro_f1'],abs_tol=1e-12)
    assert len(pred)==metrics['n_cells_evaluated']==8891
    assert len((base/'panel50/selected_genes.txt').read_text().splitlines())==50
    checks['merfish_saved_prediction_verification']={'accuracy':acc,'macro_f1':f1,'cells':len(pred),'classes':len(labels),'gene_count':50,'producer_missing':True}
    n=ROOT/'figure3_annotation/noise_benchmark/results/cortex_reference'
    truth={r['cell_barcode']:r for r in rows(n/'truth_key.csv')};flag=rows(n/'panel80_alpha070/flagged_misannotations.csv')
    assert len(truth)==len(rows(n/'truth_key.csv'))
    assert len({r['cell_barcode'] for r in flag})==len(flag)
    poisoned=lambda r:r['is_poisoned'].lower()=='true'
    total=sum(poisoned(r) for r in truth.values());tp=[r for r in flag if poisoned(truth[r['cell_barcode']])]
    recovered=sum(r['predicted_label']==truth[r['cell_barcode']]['true_label'] for r in tp)
    counts={'total_cells':len(truth),'poisoned':total,'clean':len(truth)-total,'poison_flagged':len(tp),'poison_missed':total-len(tp),'clean_flagged':len(flag)-len(tp),'clean_unflagged':len(truth)-total-len(flag)+len(tp),'recovered':recovered,'wrong_fix':len(tp)-recovered}
    saved=rows(n/'targeted_alluvial_counts.csv')[0]
    for k,v in counts.items():assert v==int(saved[k]),(k,v,saved[k])
    centroid=rows(n/'clean_false_positive_centroid_check.csv');checked=[r for r in centroid if r['checked'].lower()=='true'];closer=sum(r['predicted_is_closer'].lower()=='true' for r in checked)
    assert len(checked)==counts['clean_flagged']==1251 and closer==620
    checks['cortex_noise_saved_prediction_verification']={**counts,'catch_pct':100*len(tp)/total,'conditional_recovery_pct':100*recovered/len(tp),'all_injected_recovery_pct':100*recovered/total,'centroid_checked':len(checked),'closer_to_predicted_centroid':closer,'kidney_result':False}
    provenance=json.loads((ROOT/'figure3_annotation/results/figure3_provenance.json').read_text())
    for path,digest in provenance['inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    checks['figure3_final_input_integrity']='All final figure inputs match recorded hashes.'
    checks['verification_scope']='Existing saved values and file integrity only. No models, enrichment, PCA or scientific analyses executed.'
    (ROOT/'archive_notes/verification_report.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
