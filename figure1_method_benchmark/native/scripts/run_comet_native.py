"""Run actual COMET CLI; validate native singleton rankings used by the adapter."""
from pathlib import Path
import argparse,csv,json,hashlib,subprocess
from native_io import BASE,environment,run_command

def main():
    p=argparse.ArgumentParser()
    for name in ['markers','clusters','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve()
    if not a.out.is_relative_to(BASE/'runs') or a.out.exists():raise ValueError('Use a fresh Figure 1 native output directory')
    with a.clusters.open() as f:records=list(csv.reader(f,delimiter='\t'))
    if not records or len(records)>65536 or any(len(r)!=2 for r in records):raise ValueError('Invalid native COMET cluster input or cell-count overflow')
    classes=sorted({r[1] for r in records})
    if any('/' in c or '\\' in c for c in classes):raise ValueError('Unsafe class path')
    py=BASE/'.venv_comet/bin/python';env=environment(py);env['PYTHONPATH']=str(BASE/'vendor/COMETSC')
    subprocess.run([py,'-B','-c','import xlmhg.mhg_cython'],env=env,check=True)
    cmd=[py,'-B','-m','Comet',a.markers.resolve(),'UNUSED_VIS',a.clusters.resolve(),str(a.out/'native')+'/', '-C','1','-K','2','-skipvis','True','-Trim','2000','-Abbrev','2']
    run_command(cmd,a.out,env)
    log=(a.out/'stdout.log').read_text()+'\n'+(a.out/'stderr.log').read_text()
    for bad in ['XLMHG error','q-val error','t-test error','fold-change error','error in sliding values','discrete matrix construction failed','Traceback (most recent call last)']:
        if bad in log:raise RuntimeError('Native singleton statistics failed: '+bad)
    outputs=[]
    for cls in classes:
        files=list(a.out.rglob(f'cluster_{cls}_singleton_all_ranked.csv'))
        if len(files)!=1:raise RuntimeError('Missing native ranking '+cls)
        with files[0].open() as f:rows=list(csv.DictReader(f))
        if not rows:raise RuntimeError('Empty native ranking '+cls)
        for r in rows:
            if not 0<=float(r['mHG_stat'])<=1 or not float(r['rank'])>=1:raise RuntimeError('Invalid native rank/statistic')
        outputs.append({'class':cls,'path':str(files[0]),'rows':len(rows),'sha256':hashlib.sha256(files[0].read_bytes()).hexdigest()})
    (a.out/'native_validation.json').write_text(json.dumps({'status':'complete','n_cells':len(records),'classes':classes,'outputs':outputs,'native_pair_warning_count':log.count('pair file not generated'),'comparison_input':'Native singleton_all_ranked; missing abbreviated pairs are not used as scores'},indent=2)+'\n')
    print('Native COMET rankings validated for',len(classes),'classes')
if __name__=='__main__':main()
