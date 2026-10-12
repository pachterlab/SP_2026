#!/usr/bin/env python3
"""
Benchmark gene-panel selectors on AnnData cell-type classification.

This runner compares:
  - The checksum-preserved native CellSieve package in external/cell_sieve_package/cellsieve.py
  - scGeneFit
  - COMET
  - NSForest
  - linear-regression DE
  - logistic-regression DE
  - L1 logistic-regression stability/DE

All tools are evaluated in compression mode after panel selection:
  1. fit a balanced logistic regression on the full dataset and predict the full dataset
  2. run a stratified CV check on the same selected panel

The full-data refit metrics keep the usual column names (`accuracy`,
`macro_f1`, `worst_class_f1`, ...). The CV check is reported as `cv_accuracy`,
`cv_macro_f1`, `cv_weighted_f1`, `cv_worst_class`, and `cv_worst_class_f1`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is present in benchmark_env.
    psutil = None


BENCHMARK_ROOT = Path(__file__).resolve().parents[3] / 'results/figure1/native'
REPO_ROOT = BENCHMARK_ROOT
EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "vendor"
RANDOM_STATE = 42


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def as_dense(matrix, dtype=np.float32) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=dtype)


def parse_panel_sizes(text: str) -> list[int | str]:
    sizes: list[int | str] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if item.lower() == "full":
            sizes.append("full")
        else:
            value = int(item)
            if value < 1:
                raise ValueError(f"Panel sizes must be >= 1, got {value}")
            sizes.append(value)
    return sizes


def safe_name(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_")


def select_shared_gene_universe(adata, max_genes: int | None, normalize: bool):
    if not max_genes or max_genes <= 0 or adata.n_vars <= max_genes:
        return adata.copy(), None

    ranking_adata = adata.copy()
    ranking_adata.var_names_make_unique()
    if normalize:
        sc.pp.normalize_total(ranking_adata, target_sum=1e4)
        sc.pp.log1p(ranking_adata)

    X = as_dense(ranking_adata.X)
    variances = np.var(X, axis=0)
    keep = np.argsort(variances)[-max_genes:]
    keep.sort()
    selected_genes = ranking_adata.var_names[keep].tolist()
    return adata[:, selected_genes].copy(), selected_genes


def preprocess_adata(adata, celltype_column: str, normalize: bool):
    adata = adata.copy()
    adata.var_names_make_unique()
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    labels = adata.obs[celltype_column].astype(str).to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return adata, y, encoder


def make_split(y: np.ndarray, test_size: float, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))


def make_train_eval_indices(y: np.ndarray, args):
    if args.train_eval_full:
        idx = np.arange(len(y))
        return idx, idx
    return make_split(y, args.test_size, args.random_seed)


def panel_k(panel_size: int | str, n_genes: int) -> int:
    return n_genes if str(panel_size).lower() == "full" else min(int(panel_size), n_genes)


class ResourceMonitor:
    """Poll this process and child processes while one benchmark row runs."""

    def __init__(self, poll_seconds: float = 0.2):
        self.poll_seconds = poll_seconds
        self.proc = psutil.Process(os.getpid()) if psutil else None
        self.started = 0.0
        self.elapsed_seconds = 0.0
        self.cpu_seconds = np.nan
        self.peak_rss_mb = np.nan
        self.thread_count_peak = np.nan
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_start = 0.0

    def __enter__(self):
        self.started = time.time()
        if self.proc:
            self._cpu_start = self._cpu_total()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_seconds = time.time() - self.started
        if self.proc:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=max(1.0, self.poll_seconds * 2))
            self._sample()
            self.cpu_seconds = max(0.0, self._cpu_total() - self._cpu_start)

    def _related_processes(self):
        if not self.proc:
            return []
        procs = [self.proc]
        try:
            procs.extend(self.proc.children(recursive=True))
        except psutil.Error:
            pass
        return procs

    def _cpu_total(self) -> float:
        total = 0.0
        for proc in self._related_processes():
            try:
                times = proc.cpu_times()
                total += times.user + times.system
            except psutil.Error:
                continue
        return total

    def _sample(self) -> None:
        rss = 0
        threads = 0
        for proc in self._related_processes():
            try:
                rss += proc.memory_info().rss
                threads += proc.num_threads()
            except psutil.Error:
                continue
        self.peak_rss_mb = max(float(self.peak_rss_mb) if not np.isnan(self.peak_rss_mb) else 0.0, rss / (1024**2))
        self.thread_count_peak = max(
            float(self.thread_count_peak) if not np.isnan(self.thread_count_peak) else 0.0,
            float(threads),
        )

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self._sample()

    def summary(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "thread_count_peak": self.thread_count_peak,
            "cpu_count_logical": psutil.cpu_count(logical=True) if psutil else os.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else np.nan,
        }


def summarize_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    encoder: LabelEncoder,
):
    labels = list(range(len(encoder.classes_)))
    report_dict = classification_report(
        y_true,
        pred,
        labels=labels,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        pred,
        labels=labels,
        target_names=encoder.classes_,
        digits=4,
        zero_division=0,
    )
    class_f1 = {
        cls: float(report_dict[cls]["f1-score"])
        for cls in encoder.classes_
        if cls in report_dict
    }
    worst_class = min(class_f1, key=class_f1.get)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "worst_class": worst_class,
        "worst_class_f1": class_f1[worst_class],
        "class_f1": class_f1,
        "report_dict": report_dict,
        "report_text": report_text,
    }


def cross_validate_panel(
    X: np.ndarray,
    y: np.ndarray,
    encoder: LabelEncoder,
    n_workers: int,
    cv_splits: int,
    random_seed: int,
):
    class_counts = pd.Series(y).value_counts()
    # Match cell_sieve_v2.py: cap CV by the rarest class so stratified folds
    # can contain every class when possible.
    n_splits = min(int(cv_splits), int(class_counts.min()))
    if n_splits < 2:
        return {
            "cv_accuracy": np.nan,
            "cv_macro_f1": np.nan,
            "cv_weighted_f1": np.nan,
            "cv_worst_class": "",
            "cv_worst_class_f1": np.nan,
            "cv_class_f1": {},
            "cv_folds": [],
            "cv_n_splits": n_splits,
        }

    labels = list(range(len(encoder.classes_)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_rows = []
    class_f1_by_fold = {cls: [] for cls in encoder.classes_}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            n_jobs=n_workers,
            random_state=random_seed,
        )
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        report = classification_report(
            y[test_idx],
            pred,
            labels=labels,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        for cls in encoder.classes_:
            class_f1_by_fold[cls].append(float(report[cls]["f1-score"]))
        fold_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y[test_idx], pred),
                "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y[test_idx], pred, average="weighted", zero_division=0),
            }
        )

    cv_class_f1 = {cls: float(np.mean(values)) for cls, values in class_f1_by_fold.items()}
    cv_worst_class = min(cv_class_f1, key=cv_class_f1.get)
    return {
        "cv_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
        "cv_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
        "cv_weighted_f1": float(np.mean([row["weighted_f1"] for row in fold_rows])),
        "cv_worst_class": cv_worst_class,
        "cv_worst_class_f1": cv_class_f1[cv_worst_class],
        "cv_class_f1": cv_class_f1,
        "cv_folds": fold_rows,
        "cv_n_splits": n_splits,
    }


def evaluate_panel(
    adata,
    y: np.ndarray,
    encoder: LabelEncoder,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    genes: list[str],
    n_workers: int,
    cv_splits: int,
    random_seed: int,
):
    genes = [g for g in dict.fromkeys(genes) if g in set(adata.var_names)]
    if not genes:
        raise ValueError("No selected genes were present in adata.var_names")

    X = as_dense(adata[:, genes].X)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        n_jobs=n_workers,
        random_state=random_seed,
    )
    clf.fit(X_train, y)
    logistic = summarize_predictions(y, clf.predict(X_train), encoder)
    cv_result = cross_validate_panel(X, y, encoder, n_workers, cv_splits=cv_splits, random_seed=random_seed)

    result = {"actual_panel_size": len(genes)}
    result.update(logistic)
    result.update(cv_result)
    return result


def save_method_outputs(out_dir: Path, genes: list[str], eval_result: dict, error: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(genes, dtype=str).to_csv(out_dir / "selected_genes.txt", index=False, header=False)
    if eval_result:
        (out_dir / "classification_report.txt").write_text(eval_result["report_text"])
        with open(out_dir / "classification_report.json", "w") as f:
            json.dump(eval_result["report_dict"], f, indent=2)
        with open(out_dir / "cv_metrics.json", "w") as f:
            json.dump(
                {
                    "cv_accuracy": eval_result["cv_accuracy"],
                    "cv_macro_f1": eval_result["cv_macro_f1"],
                    "cv_weighted_f1": eval_result["cv_weighted_f1"],
                    "cv_worst_class": eval_result["cv_worst_class"],
                    "cv_worst_class_f1": eval_result["cv_worst_class_f1"],
                    "cv_n_splits": eval_result["cv_n_splits"],
                    "cv_folds": eval_result["cv_folds"],
                    "cv_class_f1": eval_result["cv_class_f1"],
                },
                f,
                indent=2,
            )
    if error:
        (out_dir / "error.txt").write_text(error)


def rank_by_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> list[int]:
    Y = pd.get_dummies(y_train).to_numpy(dtype=float)
    model = LinearRegression(n_jobs=None).fit(StandardScaler().fit_transform(X_train), Y)
    scores = np.max(np.abs(np.asarray(model.coef_)), axis=0)
    return np.argsort(scores)[::-1].tolist()


def rank_by_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, n_workers: int) -> list[int]:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        n_jobs=n_workers,
        random_state=RANDOM_STATE,
    ).fit(StandardScaler().fit_transform(X_train), y_train)
    scores = np.max(np.abs(model.coef_), axis=0)
    return np.argsort(scores)[::-1].tolist()


def rank_by_l1_logistic(X_train: np.ndarray, y_train: np.ndarray, n_workers: int, l1_c: float) -> list[int]:
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=l1_c,
        class_weight="balanced",
        max_iter=800,
        tol=0.01,
        n_jobs=n_workers,
        random_state=RANDOM_STATE,
    ).fit(StandardScaler().fit_transform(X_train), y_train)
    scores = np.max(np.abs(model.coef_), axis=0)
    return np.argsort(scores)[::-1].tolist()


def rank_by_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_workers: int,
    l1_ratio: float,
    random_seed: int,
) -> list[int]:
    scaler = StandardScaler().fit(X_train)
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=0.001,
        l1_ratio=l1_ratio,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-3,
        random_state=random_seed,
        n_jobs=n_workers,
    ).fit(scaler.transform(X_train), y_train)
    scores = np.max(np.abs(model.coef_), axis=0)
    return np.argsort(scores)[::-1].tolist()


def select_scgenefit(X_train, y_train, k):
    sys.path.insert(0,str(EXTERNAL_ROOT / "scGeneFit-python"))
    from scGeneFit.functions import get_markers
    indices=list(get_markers(StandardScaler().fit_transform(X_train), y_train.tolist(), k, method="centers", verbose=False))
    if len(indices)!=k or len(set(indices))!=k:raise RuntimeError("Native scGeneFit did not return the requested panel")
    return indices


def select_comet(adata_train, y_train, k, output_dir):
    from panel_adapters import comet_ranking, take_panel
    return take_panel(comet_ranking(adata_train,y_train,output_dir.parent / "native_rankings"),k)




def select_nsforest(adata_train, celltype_column, k, n_workers, output_dir):
    from panel_adapters import nsforest_ranking, take_panel
    return take_panel(nsforest_ranking(adata_train,celltype_column,n_workers,output_dir.parent / "native_rankings"),k)


def write_train_h5ad(raw_adata, train_idx: np.ndarray, out_path: Path):
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_adata[train_idx].copy().write_h5ad(out_path)


CELLSIEVE_PACKAGE_SCRIPT = Path(__file__).resolve().parents[3] / 'external/cell_sieve_package/cellsieve.py'


def cell_sieve_mode_for_dataset(
    adata_path: Path,
    script_path: Path,
    cell_sieve_alpha: float,
) -> tuple[str, list[str], str]:
    """Require the sole authorized native CellSieve package source."""
    if script_path.resolve() != CELLSIEVE_PACKAGE_SCRIPT.resolve():
        raise ValueError('CellSieve must use the preserved workspace package: '+str(CELLSIEVE_PACKAGE_SCRIPT))
    return 'Package_0.1.0', ['--alpha', f'{cell_sieve_alpha:g}'], 'package'


def run_cell_sieve(
    raw_adata,
    train_idx: np.ndarray,
    panel_size: int | str,
    args,
    dataset_dir: Path,
    adata_path: Path,
) -> list[str]:
    script_path = Path(args.cell_sieve_script).expanduser().resolve()
    mode_name, mode_flags, version_label = cell_sieve_mode_for_dataset(
        adata_path,
        script_path,
        args.cell_sieve_alpha,
    )
    train_path = dataset_dir / f"cell_sieve_{version_label}_train.h5ad"
    write_train_h5ad(raw_adata, train_idx, train_path)

    out_dir = dataset_dir / f"cell_sieve_{version_label}_runs" / mode_name / f"panel_size_{panel_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_tag = f"{args.cell_sieve_alpha:.2f}"
    selected_path = out_dir / f"panel_size_{panel_size}_alpha_{alpha_tag}" / "selected_genes.txt"
    if selected_path.exists():
        return pd.read_csv(selected_path, header=None)[0].astype(str).tolist()

    cmd = [
        sys.executable,
        str(script_path),
        "--adata-path",
        str(train_path),
        "--out-dir",
        str(out_dir),
        "--celltype-column",
        args.celltype_column,
        "--panel-size",
        str(panel_size),
        "--n-workers",
        str(args.n_workers),
        "--time-limit",
        str(args.cell_sieve_time_limit),
        "--mip-gap",
        str(args.cell_sieve_mip_gap),
    ]
    cmd.extend(mode_flags)
    log(f"Running Cell Sieve {version_label} [{mode_name}]: " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    (out_dir / "stdout.txt").write_text(completed.stdout)
    (out_dir / "stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Cell Sieve failed with code {completed.returncode}; see {out_dir}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Cell Sieve did not write {selected_path}")
    return pd.read_csv(selected_path, header=None)[0].astype(str).tolist()


def cached_ranking(cache,key,factory):
    if key not in cache:cache[key]=factory()
    return cache[key]


def run_dataset(adata_path: Path, args) -> pd.DataFrame:
    dataset_name = safe_name(adata_path)
    dataset_dir = Path(args.out_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log(f"Reading {adata_path}")
    raw_adata = sc.read_h5ad(adata_path)
    raw_adata, selected_gene_universe = select_shared_gene_universe(
        raw_adata,
        args.max_genes,
        normalize=not args.no_normalize,
    )
    if selected_gene_universe is not None:
        (dataset_dir / "shared_gene_universe.txt").write_text("\n".join(selected_gene_universe) + "\n")
        log(f"Using shared {len(selected_gene_universe)}-gene high-variance universe for all methods.")

    adata, y, encoder = preprocess_adata(
        raw_adata,
        args.celltype_column,
        normalize=not args.no_normalize,
    )
    train_idx = np.arange(len(y))
    test_idx = train_idx
    split_mode = f"full_train_full_eval_plus_{args.cv_splits}fold_cv"
    log(f"Evaluation mode: {split_mode}")
    adata_train = adata[train_idx].copy()
    X_train = as_dense(adata_train.X)
    y_train = y[train_idx]

    panel_sizes = parse_panel_sizes(args.panel_sizes)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    partial_path = dataset_dir / "benchmark_summary_partial.csv"
    if partial_path.exists() and args.resume:
        all_rows = pd.read_csv(partial_path).to_dict("records")
        completed = {
            (str(row["method"]), str(row["requested_panel_size"]))
            for row in all_rows
            if row.get("status") == "ok"
        }
        log(f"Resuming from {partial_path}: {len(completed)} completed rows.")
    else:
        all_rows = []
        completed = set()
    rank_cache: dict[str, list[int]] = {}

    for method in methods:
        for requested_size in panel_sizes:
            if (method, str(requested_size)) in completed:
                log(f"Skipping completed row: {method} panel_size={requested_size}")
                continue
            k = panel_k(requested_size, adata.n_vars)
            method_dir = dataset_dir / method / f"panel_size_{requested_size}"
            genes: list[str] = []
            eval_result = {}
            error = None
            resources = {}

            log(f"{dataset_name}: {method} panel_size={requested_size}")
            with ResourceMonitor() as monitor:
                try:
                    if method == "cell_sieve":
                        genes = run_cell_sieve(raw_adata, train_idx, requested_size, args, dataset_dir, adata_path)
                    elif method == "linear_regression_de":
                        cached_ranking(rank_cache,method,lambda: rank_by_linear_regression(X_train, y_train))
                        genes = adata.var_names[rank_cache[method][:k]].tolist()
                    elif method == "logistic_regression_de":
                        cached_ranking(rank_cache,method,lambda: rank_by_logistic_regression(X_train, y_train, args.n_workers))
                        genes = adata.var_names[rank_cache[method][:k]].tolist()
                    elif method == "l1_regularization":
                        cached_ranking(rank_cache,method,lambda: rank_by_l1_logistic(X_train, y_train, args.n_workers, args.l1_c))
                        genes = adata.var_names[rank_cache[method][:k]].tolist()
                    elif method == "elastic_net_de":
                        rank_cache.setdefault(
                            method,
                            rank_by_elastic_net(
                                X_train,
                                y_train,
                                args.n_workers,
                                args.elastic_net_l1_ratio,
                                args.random_seed,
                            ),
                        )
                        genes = adata.var_names[rank_cache[method][:k]].tolist()
                    elif method == "scgenefit":
                        genes = adata.var_names[select_scgenefit(X_train, y_train, k)].tolist()
                    elif method == "comet":
                        genes = select_comet(adata_train,y_train,k,method_dir)
                    elif method == "nsforest":
                        genes = select_nsforest(adata_train, args.celltype_column, k, args.n_workers, method_dir)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    eval_result = evaluate_panel(
                        adata,
                        y,
                        encoder,
                        train_idx,
                        test_idx,
                        genes,
                        args.n_workers,
                        args.cv_splits,
                        args.random_seed,
                    )
                    save_method_outputs(method_dir, genes, eval_result)

                except Exception:
                    error = traceback.format_exc()
                    save_method_outputs(method_dir, genes, eval_result, error=error)
                    log(f"ERROR in {method} panel_size={requested_size}: {error.splitlines()[-1]}")
            resources = monitor.summary()

            row = {
                "dataset": dataset_name,
                "adata_path": str(adata_path),
                "method": method,
                "requested_panel_size": requested_size,
                "actual_panel_size": eval_result.get("actual_panel_size", len(genes)),
                "accuracy": eval_result.get("accuracy", np.nan),
                "macro_f1": eval_result.get("macro_f1", np.nan),
                "worst_class_f1": eval_result.get("worst_class_f1", np.nan),
                "worst_class": eval_result.get("worst_class", ""),
                "weighted_f1": eval_result.get("weighted_f1", np.nan),
                "cv_accuracy": eval_result.get("cv_accuracy", np.nan),
                "cv_macro_f1": eval_result.get("cv_macro_f1", np.nan),
                "cv_worst_class_f1": eval_result.get("cv_worst_class_f1", np.nan),
                "cv_worst_class": eval_result.get("cv_worst_class", ""),
                "cv_weighted_f1": eval_result.get("cv_weighted_f1", np.nan),
                "cv_n_splits": eval_result.get("cv_n_splits", np.nan),
                "cell_sieve_v2_mode": cell_sieve_mode_for_dataset(
                    adata_path,
                    Path(args.cell_sieve_script).expanduser().resolve(),
                    args.cell_sieve_alpha,
                )[0] if method == "cell_sieve" else "",
                "elapsed_seconds": resources.get("elapsed_seconds", np.nan),
                "cpu_seconds": resources.get("cpu_seconds", np.nan),
                "peak_rss_mb": resources.get("peak_rss_mb", np.nan),
                "thread_count_peak": resources.get("thread_count_peak", np.nan),
                "n_workers_requested": args.n_workers,
                "cpu_count_logical": resources.get("cpu_count_logical", np.nan),
                "cpu_count_physical": resources.get("cpu_count_physical", np.nan),
                "evaluation_mode": split_mode,
                "status": "ok" if error is None else "error",
                "error": "" if error is None else error.splitlines()[-1],
                "output_dir": str(method_dir),
            }
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(partial_path, index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(dataset_dir / "benchmark_summary.csv", index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata-path", action="append", required=True, help="AnnData .h5ad path. Repeat for multiple datasets.")
    parser.add_argument("--celltype-column", default="celltype")
    parser.add_argument("--out-dir", default=str(BENCHMARK_ROOT / "runs/default"))
    parser.add_argument(
        "--methods",
        default="cell_sieve,scgenefit,comet,nsforest,linear_regression_de,logistic_regression_de,l1_regularization",
    )
    parser.add_argument("--panel-sizes", default="1,2,4,8,16,24,32,48,64,96,128,160,full")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--train-eval-full",
        action="store_true",
        help="Use the full dataset for method selection/training and evaluate on the full dataset.",
    )
    parser.add_argument("--max-genes", type=int, default=3000, help="Variance-filter genes before benchmarking external tools.")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalize_total/log1p preprocessing in the benchmark harness.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Maximum stratified CV folds for the selected-panel check. Actual folds are min(this value, rarest class count), matching cell_sieve_v2.py.",
    )
    parser.add_argument("--l1-c", type=float, default=0.05)
    parser.add_argument(
        "--elastic-net-l1-ratio",
        type=float,
        default=0.5,
        help="L1 ratio for the pure elastic-net baseline selector.",
    )
    parser.add_argument("--comet-fallback", action="store_true", help="Rejected: native statistical replacements are forbidden.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing partial summary and recompute all rows.")
    parser.set_defaults(resume=True)

    parser.add_argument("--cell-sieve-time-limit", type=float, default=900)
    parser.add_argument("--cell-sieve-mip-gap", type=float, default=0.10)
    parser.add_argument("--cell-sieve-l1-c", type=float, default=0.05)
    parser.add_argument("--cell-sieve-stability-threshold", type=float, default=0.80)
    parser.add_argument("--cell-sieve-alpha", type=float, default=0.7, help="Elastic-net L1 ratio for the native CellSieve package.")
    parser.add_argument(
        "--cell-sieve-script",
        default=str(CELLSIEVE_PACKAGE_SCRIPT),
        help="Sole authorized CellSieve source: the preserved workspace package.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.comet_fallback:raise ValueError("Native COMET cannot be replaced by a fallback selector")
    args.resume=False
    target=Path(args.out_dir).resolve()
    if not target.is_relative_to(BENCHMARK_ROOT / "runs"):raise ValueError("Use a fresh directory inside Figure 1 native/runs")
    if target.exists():raise FileExistsError("Fresh native output required")
    np.random.seed(args.random_seed)
    import hashlib
    runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for adata_path in args.adata_path:
        summaries.append(run_dataset(Path(adata_path), args))

    combined = pd.concat(summaries, ignore_index=True)
    combined_path = out_dir / "benchmark_summary_all.csv"
    combined.to_csv(combined_path, index=False)
    log(f"Wrote combined summary: {combined_path}")
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
    (out_dir / "execution_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    if (combined["status"]!="ok").any():sys.exit(1)


if __name__ == "__main__":
    main()
