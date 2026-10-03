#!/usr/bin/env python3
"""Fast L1 vs L2 marker-selection benchmark across a few tissues.

Workflow:
1. Load each AnnData file.
2. Normalize + log1p.
3. Keep the top HVGs.
4. Stratified train/test split for final evaluation.
5. Stratified subsample of the training split for fast marker ranking.
6. Fit L2 and L1 logistic regression rankers on the subsample.
7. Take the top-k genes from each ranker, train a fresh classifier on the
   training split, and evaluate on the holdout split.

Outputs a summary CSV plus a compact comparison plot in Benchmark/results/L1_L2_comp.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


RANDOM_STATE = 42
DEFAULT_DATASETS = [
    "/home/nikki/celltype_compression_2/CortexHippocampus/B6J_CortexHippocampus_subset.h5ad",
    "/home/nikki/celltype_compression_2/Liver/B6J_Liver_subset.h5ad",
    "/home/nikki/celltype_compression_2/Kidney/B6J_Kidney_subset.h5ad",
    "/home/nikki/celltype_compression_2/Gastrocnemius/B6J_Gastrocnemius_subset.h5ad",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def as_dense(matrix, dtype=float):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=dtype)


class ResourceMonitor:
    def __init__(self, poll_seconds: float = 0.2):
        self.poll_seconds = poll_seconds
        self.proc = psutil.Process(os.getpid()) if psutil else None
        self._stop = False
        self._thread = None
        self.started = 0.0
        self.elapsed_seconds = 0.0
        self.cpu_seconds = np.nan
        self.peak_rss_mb = np.nan
        self.thread_count_peak = np.nan
        self._cpu_start = 0.0

    def __enter__(self):
        self.started = time.time()
        if self.proc:
            self._cpu_start = self._cpu_total()
            import threading

            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_seconds = time.time() - self.started
        if self.proc:
            self._stop = True
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
        rss = 0.0
        threads = 0.0
        for proc in self._related_processes():
            try:
                rss += proc.memory_info().rss
                threads += proc.num_threads()
            except psutil.Error:
                continue
        current_rss = rss / (1024**2)
        self.peak_rss_mb = max(
            float(self.peak_rss_mb) if not np.isnan(self.peak_rss_mb) else 0.0,
            current_rss,
        )
        self.thread_count_peak = max(
            float(self.thread_count_peak) if not np.isnan(self.thread_count_peak) else 0.0,
            float(threads),
        )

    def _poll(self) -> None:
        while not self._stop:
            self._sample()
            time.sleep(self.poll_seconds)

    def summary(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "thread_count_peak": self.thread_count_peak,
            "cpu_count_logical": psutil.cpu_count(logical=True) if psutil else os.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else np.nan,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata-path", action="append", default=None, help="AnnData file. Repeat to override the defaults.")
    p.add_argument("--celltype-column", default="celltype")
    p.add_argument("--out-dir", default="/home/nikki/celltype_compression_2/Benchmark/results/L1_L2_comp")
    p.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--hvg-count", type=int, default=3000)
    p.add_argument("--ranking-subsample-size", type=int, default=2000)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--l1-c", type=float, default=0.05)
    p.add_argument("--l2-c", type=float, default=1.0)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--panel-sizes", default="10,20,30")
    p.add_argument("--full-eval", action="store_true", help="Use all cells for ranking and evaluation.")
    p.add_argument("--no-hvg", action="store_true", help="Skip HVG filtering and use all genes.")
    return p.parse_args()


def _rank_by_lr(X: np.ndarray, y: np.ndarray, penalty: str, C: float, random_seed: int, n_workers: int) -> tuple[np.ndarray, LogisticRegression]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if penalty == "l1":
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            class_weight="balanced",
            max_iter=1200,
            tol=0.01,
            random_state=random_seed,
            n_jobs=n_workers,
        )
    else:
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=C,
            class_weight="balanced",
            max_iter=1200,
            random_state=random_seed,
            n_jobs=n_workers,
        )
    clf.fit(Xs, y)
    scores = np.max(np.abs(clf.coef_), axis=0)
    return scores, clf


def _select_top_genes(scores: np.ndarray, var_names: pd.Index, top_k: int) -> list[str]:
    idx = np.argsort(scores)[::-1][:top_k]
    return var_names[idx].tolist()


def _evaluate_panel(adata, y, train_idx, test_idx, genes: list[str], n_workers: int) -> dict:
    genes = [g for g in dict.fromkeys(genes) if g in set(adata.var_names)]
    if not genes:
        raise ValueError("No selected genes survived filtering.")

    X = as_dense(adata[:, genes].X)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        n_jobs=n_workers,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y[train_idx])
    pred = clf.predict(X_test)
    report = classification_report(y[test_idx], pred, output_dict=True, zero_division=0)
    class_f1 = {
        cls: float(report[cls]["f1-score"])
        for cls in report
        if cls not in {"accuracy", "macro avg", "weighted avg"} and isinstance(report.get(cls), dict)
    }
    worst_class = min(class_f1, key=class_f1.get)
    return {
        "actual_panel_size": len(genes),
        "accuracy": accuracy_score(y[test_idx], pred),
        "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y[test_idx], pred, average="weighted", zero_division=0),
        "worst_class": worst_class,
        "worst_class_f1": class_f1[worst_class],
        "report_dict": report,
        "report_text": classification_report(y[test_idx], pred, digits=4, zero_division=0),
    }


def _pick_stratified_subset(y: np.ndarray, subset_size: int, random_seed: int) -> np.ndarray:
    subset_size = min(subset_size, len(y))
    if subset_size == len(y):
        return np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=random_seed)
    subset_idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return subset_idx


def _plot_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    labels = {
        "logistic_regression_de": "Logistic regression (L2)",
        "l1_regularization": "Logistic regression (L1)",
    }
    colors = {
        "logistic_regression_de": "#7f3c8d",
        "l1_regularization": "#8c564b",
    }
    methods = ["logistic_regression_de", "l1_regularization"]
    tissues = list(dict.fromkeys(summary["dataset_label"].tolist()))
    panel_sizes = sorted(summary["requested_panel_size"].unique())

    fig, axes = plt.subplots(len(tissues), 1, figsize=(9.5, 3.4 * len(tissues)), sharex=True)
    if len(tissues) == 1:
        axes = [axes]

    for ax, tissue in zip(axes, tissues):
        sub_tissue = summary[summary["dataset_label"].eq(tissue)]
        for method in methods:
            sub = sub_tissue[sub_tissue["method"].eq(method)].sort_values("requested_panel_size")
            ax.plot(
                sub["requested_panel_size"],
                sub["macro_f1"],
                marker="o",
                linewidth=2.2,
                color=colors[method],
                label=labels[method],
            )
        ax.set_title(tissue)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.set_ylabel("Macro F1")
        ax.axhline(0.9, color="black", linestyle=":", linewidth=1.0, alpha=0.45)
        ax.set_xticks(panel_sizes)

    axes[-1].set_xlabel("Panel size (genes)")
    axes[0].legend(frameon=False)
    fig.suptitle("L1 vs L2 full-eval macro F1")
    fig.tight_layout()
    fig.savefig(out_dir / "macro_f1_comparison.png", dpi=240)
    fig.savefig(out_dir / "macro_f1_comparison.pdf")
    plt.close(fig)


def run_dataset(path: Path, args) -> list[dict]:
    dataset_label = path.stem.replace("B6J_", "").replace("_subset", "")
    dataset_dir = Path(args.out_dir) / dataset_label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log(f"Reading {path}")
    adata = sc.read_h5ad(path)
    adata = adata.copy()
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if not args.no_hvg and args.hvg_count and args.hvg_count < adata.n_vars:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=args.hvg_count,
            flavor="seurat",
            subset=True,
        )
        log(f"{dataset_label}: kept {adata.n_vars} HVGs")

    celltype_series = pd.Series(adata.obs[args.celltype_column].astype(str).values)
    y = celltype_series.astype("category").cat.codes.to_numpy()
    classes = celltype_series.astype("category").cat.categories.tolist()

    if args.full_eval:
        train_idx = np.arange(len(y))
        test_idx = np.arange(len(y))
        rank_idx = np.arange(len(y))
        split_note = "full_train_full_eval"
    else:
        train_idx, test_idx = next(
            StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_seed).split(
                np.zeros(len(y)), y
            )
        )
        rank_idx = _pick_stratified_subset(y[train_idx], args.ranking_subsample_size, args.random_seed)
        rank_idx = train_idx[rank_idx]
        split_note = f"stratified_holdout_test_size_{args.test_size}"

    log(
        f"{dataset_label}: train={len(train_idx)} test={len(test_idx)} "
        f"rank_subsample={len(rank_idx)} mode={split_note}"
    )

    X_rank = as_dense(adata[rank_idx].X)
    y_rank = y[rank_idx]

    panel_sizes = [int(x) for x in args.panel_sizes.split(",") if x.strip()]
    rows: list[dict] = []
    for method, penalty, C in [
        ("logistic_regression_de", "l2", args.l2_c),
        ("l1_regularization", "l1", args.l1_c),
    ]:
        method_dir = dataset_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        log(f"{dataset_label}: fitting {method}")
        error = None
        with ResourceMonitor() as monitor:
            try:
                scores, model = _rank_by_lr(
                    X_rank,
                    y_rank,
                    penalty=penalty,
                    C=C,
                    random_seed=args.random_seed,
                    n_workers=args.n_workers,
                )
                for requested_size in panel_sizes:
                    panel_dir = method_dir / f"panel_size_{requested_size}"
                    panel_dir.mkdir(parents=True, exist_ok=True)
                    genes = _select_top_genes(scores, adata.var_names, requested_size)
                    eval_result = _evaluate_panel(
                        adata,
                        y,
                        train_idx,
                        test_idx,
                        genes,
                        args.n_workers,
                    )

                    pd.Series(genes, dtype=str).to_csv(panel_dir / "selected_genes.txt", index=False, header=False)
                    (panel_dir / "classification_report.txt").write_text(eval_result["report_text"])
                    with open(panel_dir / "classification_report.json", "w") as f:
                        json.dump(eval_result["report_dict"], f, indent=2)

                    rows.append(
                        {
                            "dataset": dataset_label,
                            "adata_path": str(path),
                            "dataset_label": dataset_label,
                            "method": method,
                            "method_label": "Logistic regression (L2)" if method == "logistic_regression_de" else "Logistic regression (L1)",
                            "requested_panel_size": requested_size,
                            "actual_panel_size": eval_result.get("actual_panel_size", len(genes)),
                            "accuracy": eval_result.get("accuracy", np.nan),
                            "macro_f1": eval_result.get("macro_f1", np.nan),
                            "worst_class_f1": eval_result.get("worst_class_f1", np.nan),
                            "worst_class": eval_result.get("worst_class", ""),
                            "weighted_f1": eval_result.get("weighted_f1", np.nan),
                            "elapsed_seconds": np.nan,
                            "cpu_seconds": np.nan,
                            "peak_rss_mb": np.nan,
                            "thread_count_peak": np.nan,
                            "n_workers_requested": args.n_workers,
                            "cpu_count_logical": psutil.cpu_count(logical=True) if psutil else os.cpu_count(),
                            "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else np.nan,
                            "hvg_count": adata.n_vars,
                            "ranking_subsample_size": len(rank_idx),
                            "train_size": len(train_idx),
                            "test_size": len(test_idx),
                            "celltype_column": args.celltype_column,
                            "evaluation_mode": split_note,
                            "status": "ok",
                            "error": "",
                            "output_dir": str(panel_dir),
                        }
                    )
            except Exception:
                error = traceback.format_exc()
                (method_dir / "error.txt").write_text(error)
                log(f"ERROR in {dataset_label}/{method}: {error.splitlines()[-1]}")

        resources = monitor.summary()
        if rows and rows[-1]["dataset"] == dataset_label and rows[-1]["method"] == method:
            for row in reversed(rows):
                if row["dataset"] == dataset_label and row["method"] == method and row["status"] == "ok":
                    row["elapsed_seconds"] = resources.get("elapsed_seconds", np.nan)
                    row["cpu_seconds"] = resources.get("cpu_seconds", np.nan)
                    row["peak_rss_mb"] = resources.get("peak_rss_mb", np.nan)
                    row["thread_count_peak"] = resources.get("thread_count_peak", np.nan)
                else:
                    break

    pd.DataFrame(rows).to_csv(dataset_dir / "benchmark_summary.csv", index=False)
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata_paths = args.adata_path if args.adata_path else DEFAULT_DATASETS
    all_rows: list[dict] = []
    for raw_path in adata_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path("/home/nikki/celltype_compression_2") / path
        all_rows.extend(run_dataset(path, args))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    summary.to_csv(out_dir / "benchmark_summary_all.csv", index=False)
    _plot_comparison(summary, out_dir)

    best = (
        summary.sort_values(["dataset_label", "macro_f1"], ascending=[True, False])
        .groupby("dataset_label", as_index=False)
        .head(1)
        .sort_values("macro_f1", ascending=False)
    )
    best.to_csv(out_dir / "best_rows_by_macro_f1.csv", index=False)

    log(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
