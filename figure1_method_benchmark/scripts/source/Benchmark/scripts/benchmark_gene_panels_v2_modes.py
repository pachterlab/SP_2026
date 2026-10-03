#!/usr/bin/env python3
"""
Benchmark gene-panel selectors on AnnData cell-type classification.

This runner compares:
  - Cell Sieve v2, using dataset-specific modes:
    * B cell ground truth: FineSieve_Unconstrained
    * Cortex/Hippocampus: CoarseSieve_PosOnly
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


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = BENCHMARK_ROOT / "external"
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


def select_scgenefit(X_train: np.ndarray, y_train: np.ndarray, k: int) -> list[int]:
    sys.path.insert(0, str(EXTERNAL_ROOT / "scGeneFit-python"))
    from scGeneFit.functions import get_markers

    X_scaled = StandardScaler().fit_transform(X_train)
    return list(get_markers(X_scaled, y_train.tolist(), k, method="centers", verbose=False))


def select_comet(adata_train, y_train: np.ndarray, k: int) -> list[str]:
    sys.path.insert(0, str(EXTERNAL_ROOT / "COMETSC"))
    if not hasattr(np, "int"):
        np.int = int  # COMETSC still uses the pre-NumPy-2 alias.
    from Comet import hgmd

    X = as_dense(adata_train.X, dtype=np.float64)
    marker_exp = pd.DataFrame(X, columns=adata_train.var_names, index=adata_train.obs_names)
    c_list = pd.Series(y_train, index=adata_train.obs_names)

    rows = []
    for cls in sorted(np.unique(y_train)):
        try:
            ranked = hgmd.batch_xlmhg(marker_exp, c_list, cls)
            ranked = ranked.sort_values("mHG_pval", ascending=True)
            rows.append(ranked[["gene_1", "mHG_pval"]].rename(columns={"mHG_pval": f"p_{cls}"}))
        except Exception:
            # Keep other classes running; the outer caller records total failure if no rows survive.
            log(f"COMET failed for class {cls}; continuing with remaining classes.")

    if not rows:
        raise RuntimeError("COMET produced no class rankings")

    merged = rows[0]
    for row in rows[1:]:
        merged = merged.merge(row, on="gene_1", how="outer")
    p_cols = [c for c in merged.columns if c.startswith("p_")]
    merged["score"] = merged[p_cols].min(axis=1)
    return merged.sort_values("score", ascending=True)["gene_1"].head(k).tolist()


def select_comet_wilcoxon_fallback(adata_train, y_train: np.ndarray, k: int) -> list[str]:
    """Fast COMET-style one-vs-rest up-marker ranking when xlmhg is unavailable."""
    X = as_dense(adata_train.X, dtype=np.float64)
    scores = np.full(X.shape[1], np.inf)
    for cls in sorted(np.unique(y_train)):
        in_mask = y_train == cls
        out_mask = ~in_mask
        for j in range(X.shape[1]):
            in_vals = X[in_mask, j]
            out_vals = X[out_mask, j]
            if np.mean(in_vals) <= np.mean(out_vals):
                continue
            try:
                _, pval = stats.ranksums(in_vals, out_vals)
                scores[j] = min(scores[j], pval)
            except Exception:
                continue
    return adata_train.var_names[np.argsort(scores)[:k]].tolist()


def select_nsforest(adata_train, celltype_column: str, k: int, n_workers: int, output_dir: Path) -> list[str]:
    sys.path.insert(0, str(EXTERNAL_ROOT / "NSForest"))
    from nsforest.preprocessing import prep_binary_scores, prep_medians
    from nsforest.nsforesting import NSForest

    ns_adata = adata_train.copy()
    ns_adata.obs[celltype_column] = ns_adata.obs[celltype_column].astype("category")
    ns_adata = prep_medians(ns_adata, celltype_column, positive_genes_only=True)
    ns_adata = prep_binary_scores(ns_adata, celltype_column)
    results = NSForest(
        ns_adata,
        celltype_column,
        n_trees=200,
        n_jobs=n_workers,
        n_top_genes=max(15, min(k * 2, 100)),
        n_binary_genes=max(10, min(k * 2, 100)),
        n_genes_eval=min(6, max(1, k)),
        save=True,
        output_folder=str(output_dir) + os.sep,
        outputfilename_prefix="nsforest",
    )
    ranked: list[str] = []
    for _, row in results.sort_values("f_score", ascending=False).iterrows():
        markers = row.get("NSForest_markers", [])
        if isinstance(markers, str):
            markers = [markers]
        for gene in markers:
            if gene not in ranked:
                ranked.append(gene)

    for _, row in results.sort_values("f_score", ascending=False).iterrows():
        binary_genes = row.get("binary_genes", [])
        if isinstance(binary_genes, str):
            binary_genes = [binary_genes]
        for gene in binary_genes:
            if gene not in ranked:
                ranked.append(gene)

    if len(ranked) < k:
        X = as_dense(adata_train.X)
        for gene in adata_train.var_names[np.argsort(np.var(X, axis=0))[::-1]]:
            if gene not in ranked:
                ranked.append(gene)
            if len(ranked) >= k:
                break
    return ranked[:k]


def write_train_h5ad(raw_adata, train_idx: np.ndarray, out_path: Path):
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_adata[train_idx].copy().write_h5ad(out_path)


def cell_sieve_mode_for_dataset(
    adata_path: Path,
    script_path: Path,
    cell_sieve_alpha: float,
) -> tuple[str, list[str], str]:
    """Return mode name, extra CLI flags, and version label for the selected Cell Sieve script."""
    dataset_key = str(adata_path).lower()
    script_name = script_path.name.lower()
    if "v4" in script_name:
        return "ElasticNet_Unconstrained", ["--alpha", f"{cell_sieve_alpha:g}"], "v4"
    if "v3" in script_name:
        return "ElasticNet_Unconstrained", [], "v3"

    if "bcell" in dataset_key or "b_cell" in dataset_key or "b-cell" in dataset_key:
        return "FineSieve_Unconstrained", ["--fine-sieve"], "v2"
    if "cortexhippocampus" in dataset_key or "cortex" in dataset_key or "hippocampus" in dataset_key:
        return "CoarseSieve_PosOnly", ["--restrict-positive-only"], "v2"
    raise ValueError(
        "Could not infer Cell Sieve v2 mode from dataset path. "
        "Expected a B cell or Cortex/Hippocampus dataset path."
    )


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
    if version_label == "v4":
        alpha_tag = f"{args.cell_sieve_alpha:.2f}"
        selected_path = out_dir / f"panel_size_{panel_size}_{mode_name}_alpha_{alpha_tag}" / "selected_genes.txt"
    else:
        selected_path = out_dir / f"panel_size_{panel_size}_{mode_name}" / "selected_genes.txt"
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
    if version_label != "v4":
        cmd.extend(
            [
                "--cache-dir",
                str(dataset_dir / f"cell_sieve_{version_label}_cache" / mode_name),
            ]
        )
    if version_label == "v2":
        cmd.extend(
            [
                "--l1-c",
                str(args.cell_sieve_l1_c),
                "--stability-threshold",
                str(args.cell_sieve_stability_threshold),
            ]
        )
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
                        rank_cache.setdefault(method, rank_by_linear_regression(X_train, y_train))
                        genes = adata.var_names[rank_cache[method][:k]].tolist()
                    elif method == "logistic_regression_de":
                        rank_cache.setdefault(method, rank_by_logistic_regression(X_train, y_train, args.n_workers))
                        genes = adata.var_names[rank_cache[method][:k]].tolist()
                    elif method == "l1_regularization":
                        rank_cache.setdefault(method, rank_by_l1_logistic(X_train, y_train, args.n_workers, args.l1_c))
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
                        try:
                            genes = select_comet(adata_train, y_train, k)
                        except Exception as exc:
                            if args.comet_fallback:
                                log(f"COMET adapter failed ({exc}); using Wilcoxon fallback.")
                                genes = select_comet_wilcoxon_fallback(adata_train, y_train, k)
                            else:
                                raise
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
    parser.add_argument("--out-dir", default=str(BENCHMARK_ROOT / "results"))
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
    parser.add_argument("--comet-fallback", action="store_true", help="Use Wilcoxon one-vs-rest ranking if COMET/xlmhg is unavailable.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing partial summary and recompute all rows.")
    parser.set_defaults(resume=True)

    parser.add_argument("--cell-sieve-time-limit", type=float, default=900)
    parser.add_argument("--cell-sieve-mip-gap", type=float, default=0.10)
    parser.add_argument("--cell-sieve-l1-c", type=float, default=0.05)
    parser.add_argument("--cell-sieve-stability-threshold", type=float, default=0.80)
    parser.add_argument("--cell-sieve-alpha", type=float, default=0.9, help="Elastic-net alpha for Cell Sieve v4.")
    parser.add_argument(
        "--cell-sieve-script",
        default=str(REPO_ROOT / "cell_sieve_v4.py"),
        help="Path to the Cell Sieve v2 script to benchmark.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for adata_path in args.adata_path:
        summaries.append(run_dataset(Path(adata_path), args))

    combined = pd.concat(summaries, ignore_index=True)
    combined_path = out_dir / "benchmark_summary_all.csv"
    combined.to_csv(combined_path, index=False)
    log(f"Wrote combined summary: {combined_path}")


if __name__ == "__main__":
    main()
