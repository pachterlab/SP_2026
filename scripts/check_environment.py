"""Check the paper runtime and local dependencies without fitting any model."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'metadata/environment_check.json')
    args=parser.parse_args()
    failures=[];packages={}
    for line in (ROOT/'environments/paper-requirements.lock.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,expected=line.split('==')
        try:
            actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual=None
        packages[name]=dict(expected=expected,installed=actual)
        if actual!=expected:failures.append(f'{name}: expected {expected}, got {actual}')
    imports=['numpy','pandas','scipy','sklearn','scanpy','anndata','matplotlib','gurobipy','igraph','leidenalg','h5py','yaml']
    for name in imports:
        try:importlib.import_module(name)
        except Exception as exc:failures.append(f'Import {name}: {exc}')
    from figure_style import set_style
    try:font=set_style()
    except Exception as exc:font=None;failures.append(str(exc))
    gs=shutil.which('gs')
    if gs is None:failures.append('Ghostscript (gs) missing')
    paths=[ROOT/'h5ads/B6J_CortexHippocampus_subset.h5ad',ROOT/'h5ads/Merfish_cortex.h5ad',
        ROOT/'h5ads/B6J_CortexHippocampus_subset_POISONED_010.h5ad',
        ROOT/'external/cell_sieve_package/cellsieve.py',
        ROOT/'assets/methods.pdf',
        ROOT/'results/figure2/resources/go/go-basic.obo',
        ROOT/'results/figure2/resources/go/mgi.gaf.gz',
        ROOT/'results/supplementary/fullinput_benchmarks/benchmark_results_both.csv',
        ROOT/'results/supplementary/alpha_subsampling/summary/selected_alphas.csv']
    missing=[str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    failures.extend(f'Missing bundle file: {p}' for p in missing)
    comet=ROOT/'scripts/native/.venv_comet/bin/python'
    native_check=None
    if comet.exists():
        result=subprocess.run([str(comet),'-B','-c','import xlmhg.mhg_cython; import numpy; print(numpy.__version__)'],capture_output=True,text=True)
        native_check=dict(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
        if result.returncode:failures.append('Compiled COMET import failed')
    else:failures.append('COMET environment missing: run setup_benchmark_runtime.sh')
    report=dict(status='passed' if not failures else 'failed',python=sys.executable,python_version=platform.python_version(),
        packages=packages,helvetica=font,ghostscript=gs,comet=native_check,failures=failures,
        scope='Package/import/font/native-extension/input checks only; no Gurobi solve or scientific fit.')
    path=args.report.resolve()
    if not path.is_relative_to(ROOT):parser.error('Report must stay inside the workspace')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{report["status"]}: {len(packages)} pinned packages, core imports, fonts, COMET and local input bundle.')
    if failures:raise SystemExit('\n'.join(failures))


if __name__=='__main__':main()
