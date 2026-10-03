"""Rebuild the native execution engine from the preserved historical runner."""
from pathlib import Path
import ast,hashlib,json
BASE=Path(__file__).resolve().parents[1]
FIGURE=BASE.parent
source=FIGURE/'scripts/source/Benchmark/scripts/benchmark_gene_panels_v2_modes.py'
s=source.read_text();original=ast.parse(s)
replacements={
'select_scgenefit':'''def select_scgenefit(X_train, y_train, k):
    sys.path.insert(0,str(EXTERNAL_ROOT / "scGeneFit-python"))
    from scGeneFit.functions import get_markers
    indices=list(get_markers(StandardScaler().fit_transform(X_train), y_train.tolist(), k, method="centers", verbose=False))
    if len(indices)!=k or len(set(indices))!=k:raise RuntimeError("Native scGeneFit did not return the requested panel")
    return indices
''',
'select_comet_wilcoxon_fallback':'',
'select_comet':'''def select_comet(adata_train, y_train, k, output_dir):
    from panel_adapters import comet_ranking, take_panel
    return take_panel(comet_ranking(adata_train,y_train,output_dir.parent / "native_rankings"),k)
''',
'select_nsforest':'''def select_nsforest(adata_train, celltype_column, k, n_workers, output_dir):
    from panel_adapters import nsforest_ranking, take_panel
    return take_panel(nsforest_ranking(adata_train,celltype_column,n_workers,output_dir.parent / "native_rankings"),k)
'''}
lines=s.splitlines(keepends=True)
for node in sorted([n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in replacements],key=lambda n:n.lineno,reverse=True):lines[node.lineno-1:node.end_lineno]=[replacements[node.name]]
s=''.join(lines)
s=s.replace('REPO_ROOT = Path(__file__).resolve().parents[2]\nBENCHMARK_ROOT = Path(__file__).resolve().parents[1]\nEXTERNAL_ROOT = BENCHMARK_ROOT / "external"','BENCHMARK_ROOT = Path(__file__).resolve().parents[1]\nREPO_ROOT = BENCHMARK_ROOT.parent / "scripts/source"\nEXTERNAL_ROOT = BENCHMARK_ROOT / "vendor"')
a=s.index('                    elif method == "comet":');b=s.index('                    elif method == "nsforest":',a)
s=s[:a]+'''                    elif method == "comet":
                        genes = select_comet(adata_train,y_train,k,method_dir)
'''+s[b:]
for call in ['rank_by_linear_regression(X_train, y_train)','rank_by_logistic_regression(X_train, y_train, args.n_workers)','rank_by_l1_logistic(X_train, y_train, args.n_workers, args.l1_c)']:
 s=s.replace('rank_cache.setdefault(method, '+call+')','cached_ranking(rank_cache,method,lambda: '+call+')')
s=s.replace('def run_dataset(','''def cached_ranking(cache,key,factory):
    if key not in cache:cache[key]=factory()
    return cache[key]


def run_dataset(''')
s=s.replace('default=str(BENCHMARK_ROOT / "results")','default=str(BENCHMARK_ROOT / "runs/default")')
s=s.replace('help="Use Wilcoxon one-vs-rest ranking if COMET/xlmhg is unavailable."','help="Rejected: native statistical replacements are forbidden."')
s=s.replace('    args = parse_args()\n    out_dir','''    args = parse_args()
    if args.comet_fallback:raise ValueError("Native COMET cannot be replaced by a fallback selector")
    args.resume=False
    target=Path(args.out_dir).resolve()
    if not target.is_relative_to(BENCHMARK_ROOT / "runs"):raise ValueError("Use a fresh directory inside Figure 1 native/runs")
    if target.exists():raise FileExistsError("Fresh native output required")
    np.random.seed(args.random_seed)
    import hashlib
    runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_dir''')
s=s.replace('    log(f"Wrote combined summary: {combined_path}")','''    log(f"Wrote combined summary: {combined_path}")
    from importlib.metadata import version,PackageNotFoundError
    packages={}
    for name in ["numpy","pandas","scipy","scanpy","scikit-learn","gurobipy"]:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]="not installed"
    provenance={"argv":sys.argv,"python":sys.executable,"packages":packages,"native_only":True,
        "evaluation_unchanged":True,"native_rank_budget_adapters":True,
        "cellsieve_script":str(Path(args.cell_sieve_script).resolve()),
        "cellsieve_script_sha256":hashlib.sha256(Path(args.cell_sieve_script).read_bytes()).hexdigest(),
        "runner_sha256":runner_sha}
    (out_dir / "execution_provenance.json").write_text(json.dumps(provenance,indent=2)+"\\n")
    if (combined["status"]!="ok").any():sys.exit(1)''')
preserved=['select_shared_gene_universe','preprocess_adata','make_train_eval_indices','cross_validate_panel','evaluate_panel','summarize_predictions','rank_by_l1_logistic','rank_by_logistic_regression','rank_by_linear_regression','run_cell_sieve','save_method_outputs']
def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
a=functions(original);b=functions(ast.parse(s));assert all(a[k]==b[k] for k in preserved)
(BASE/'scripts/benchmark_execution.py').write_text(s)
(BASE/'execution_patch_manifest.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'working_copy':'scripts/benchmark_execution.py','preserved_function_bodies':preserved},indent=2)+'\n')
print('Rebuilt native engine; all 11 numerical functions match original AST')
