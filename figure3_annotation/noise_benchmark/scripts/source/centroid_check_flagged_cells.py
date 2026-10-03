#!/usr/bin/env python3
"""
Check whether Cell Sieve-corrected/flagged cells are closer to their corrected
celltype centroid than to their original annotation centroid.
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = (
    "/home/nikki/celltype_compression_2/cell_sieve_v6_output/CortexHippocampus/"
    "panel_size_80_ElasticNet_Unconstrained_alpha_0.70/"
    "CLEANED_B6J_CortexHippocampus_subset.h5ad"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centroid-distance validation for corrected/flagged cells."
    )
    parser.add_argument("--adata-path", default=DEFAULT_H5AD, help="Cleaned h5ad to inspect.")
    parser.add_argument(
        "--original-col",
        default="celltype",
        help="Original annotation column in adata.obs.",
    )
    parser.add_argument(
        "--corrected-col",
        default="celltype_corrected",
        help="Corrected annotation column in adata.obs.",
    )
    parser.add_argument(
        "--embedding-key",
        default="X_pca",
        help="Embedding in adata.obsm used for centroid distances.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to '<h5ad directory>/centroid_check'.",
    )
    parser.add_argument(
        "--exclude-flagged-from-centroids",
        action="store_true",
        default=True,
        help="Build label centroids from unflagged cells only. This is the default.",
    )
    parser.add_argument(
        "--include-flagged-in-centroids",
        dest="exclude_flagged_from_centroids",
        action="store_false",
        help="Build centroids from all cells, including corrected/flagged cells.",
    )
    parser.add_argument(
        "--centroid-method",
        choices=["mean", "median"],
        default="median",
        help="Centroid estimator. Median is more robust for rare/noisy cell types.",
    )
    parser.add_argument(
        "--min-centroid-cells",
        type=int,
        default=20,
        help="Minimum reference cells required to evaluate a label centroid.",
    )
    parser.add_argument(
        "--top-n-nearest",
        type=int,
        default=5,
        help="Store corrected-label rank among this many nearest centroids.",
    )
    parser.add_argument(
        "--transition-plot-top-n",
        type=int,
        default=25,
        help="Maximum original-to-corrected label transitions to draw in the flagged-only alluvial.",
    )
    parser.add_argument(
        "--transition-plot-min-cells",
        type=int,
        default=1,
        help="Minimum cells required for an original-to-corrected transition to be drawn.",
    )
    return parser.parse_args()


def validate_inputs(adata, original_col, corrected_col, embedding_key):
    missing_obs = [col for col in (original_col, corrected_col) if col not in adata.obs]
    if missing_obs:
        raise ValueError(f"Missing adata.obs columns: {missing_obs}")
    if embedding_key not in adata.obsm:
        raise ValueError(f"Missing adata.obsm embedding: {embedding_key}")


def clean_label_series(series):
    labels = series.astype("string").str.strip()
    return labels.mask(labels.isna() | labels.isin(["", "NA", "nan", "None"]))


def build_centroids(coords, labels, method, min_cells):
    centroids = {}
    counts = pd.Series(labels).value_counts(dropna=True).to_dict()
    for label in pd.Series(labels).dropna().unique():
        mask = labels == label
        if np.sum(mask) < min_cells:
            continue
        if method == "median":
            centroids[label] = np.median(coords[mask], axis=0)
        else:
            centroids[label] = coords[mask].mean(axis=0)
    return centroids, counts


def summarize_by_transition(results):
    grouped = (
        results.groupby(["original_label", "corrected_label"], observed=True)
        .agg(
            n_cells=("cell_barcode", "size"),
            checked_n=("checked", "sum"),
            corrected_closer_n=("corrected_is_closer", "sum"),
            corrected_nearest_n=("corrected_is_nearest_centroid", "sum"),
            corrected_in_top_n_n=("corrected_in_top_n", "sum"),
            median_original_distance=("original_centroid_distance", "median"),
            median_corrected_distance=("corrected_centroid_distance", "median"),
            median_distance_delta=("distance_delta_original_minus_corrected", "median"),
            mean_distance_delta=("distance_delta_original_minus_corrected", "mean"),
        )
        .reset_index()
        .sort_values(["corrected_closer_n", "n_cells"], ascending=False)
    )
    denom = grouped["checked_n"].replace(0, np.nan)
    grouped["corrected_closer_pct_checked"] = grouped["corrected_closer_n"] / denom * 100
    grouped["corrected_nearest_pct_checked"] = grouped["corrected_nearest_n"] / denom * 100
    grouped["corrected_in_top_n_pct_checked"] = grouped["corrected_in_top_n_n"] / denom * 100
    return grouped


def save_summary_plot(summary, out_path):
    values = [
        summary["total_cells"],
        summary["flagged_cells"],
        summary["corrected_closer_n"],
        summary["corrected_nearest_n"],
        summary["corrected_in_top_n_n"],
    ]
    labels = [
        "Total cells",
        "Flagged/corrected",
        "Corrected centroid closer",
        "Corrected centroid nearest",
        f"Corrected in top {summary['top_n_nearest']}",
    ]
    colors = ["#b7c9df", "#4c78a8", "#2ca25f", "#41ab5d", "#74c476"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Cells")
    ax.set_title("Centroid Check for Flagged Cells", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fmt_count(value):
    return f"{int(value):,}"


def _draw_flow(ax, source, target, value, scale, color, alpha=0.55):
    if value <= 0:
        return

    height = value * scale
    source_top = source["out_cursor"]
    source_bottom = source_top - height
    target_top = target["in_cursor"]
    target_bottom = target_top - height
    source["out_cursor"] = source_bottom
    target["in_cursor"] = target_bottom

    x0 = source["x"] + source["w"]
    x1 = target["x"]
    dx = (x1 - x0) * 0.55
    verts = [
        (x0, source_top),
        (x0 + dx, source_top),
        (x1 - dx, target_top),
        (x1, target_top),
        (x1, target_bottom),
        (x1 - dx, target_bottom),
        (x0 + dx, source_bottom),
        (x0, source_bottom),
        (x0, source_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def save_centroid_alluvial(summary, out_path, counts_csv, checked_results=None):
    checked_results = checked_results if checked_results is not None else pd.DataFrame()

    if checked_results.empty:
        closer_to_corrected = 0
        top_5_centroid = 0
        unsupported = 0
        unchecked = summary["flagged_cells"]
    else:
        checked_n = int(len(checked_results))
        unchecked = int(summary["flagged_cells"] - checked_n)

        closer_mask = checked_results["corrected_is_closer"].fillna(False)
        top_5_mask = checked_results["corrected_in_top_n"].fillna(False)

        closer_to_corrected = int(closer_mask.sum())
        top_5_centroid = int((~closer_mask & top_5_mask).sum())
        unsupported = int((~closer_mask & ~top_5_mask).sum())

    alluvial_counts = {
        "total_cells": summary["total_cells"],
        "flagged_cells": summary["flagged_cells"],
        "unflagged_cells": summary["total_cells"] - summary["flagged_cells"],
        "closer_to_corrected": closer_to_corrected,
        "top_5_centroid": top_5_centroid,
        "unsupported": unsupported,
        "unchecked": unchecked,
        "total_good": closer_to_corrected + top_5_centroid,
        "total_bad": unsupported + unchecked,
    }

    pd.DataFrame([alluvial_counts]).to_csv(counts_csv, index=False)

    node_specs = [
        ("Total", "total_cells", 0),
        ("Flagged", "flagged_cells", 1),
        ("Unflagged", "unflagged_cells", 1),
        ("Closer to\ncorrected", "closer_to_corrected", 2),
        (f"Top {summary['top_n_nearest']}\ncentroid", "top_5_centroid", 2),
        ("Unsupported", "unsupported", 2),
        ("Unchecked", "unchecked", 2),
        ("Good", "total_good", 3),
        ("Bad", "total_bad", 3),
    ]

    links = [
        (0, 1, "flagged_cells", "#4c78a8"),
        (0, 2, "unflagged_cells", "#b7c9df"),
        (1, 3, "closer_to_corrected", "#238b45"),
        (1, 4, "top_5_centroid", "#74c476"),
        (1, 5, "unsupported", "#f16913"),
        (1, 6, "unchecked", "#9e9e9e"),
        (3, 7, "closer_to_corrected", "#238b45"),
        (4, 7, "top_5_centroid", "#74c476"),
        (5, 8, "unsupported", "#f16913"),
        (6, 8, "unchecked", "#9e9e9e"),
    ]

    layer_x = {
        0: 0.04,
        1: 0.32,
        2: 0.62,
        3: 0.90,
    }

    node_w = 0.055
    total = alluvial_counts["total_cells"]
    scale = 0.74 / total if total else 0

    nodes = []
    for label, key, layer in node_specs:
        value = alluvial_counts[key]
        height = max(value * scale, 0.014 if value > 0 else 0.0)

        nodes.append({
            "label": label,
            "count_label": _fmt_count(value),
            "key": key,
            "layer": layer,
            "value": value,
            "h": height,
            "x": layer_x[layer],
            "w": node_w,
        })

    for layer in sorted(layer_x):
        layer_nodes = [node for node in nodes if node["layer"] == layer]
        gap = 0.060 if len(layer_nodes) > 1 else 0

        total_height = sum(node["h"] for node in layer_nodes) + gap * (len(layer_nodes) - 1)

        if total_height > 0.90 and len(layer_nodes) > 1:
            gap = max(
                (0.90 - sum(node["h"] for node in layer_nodes)) / (len(layer_nodes) - 1),
                0.025,
            )
            total_height = sum(node["h"] for node in layer_nodes) + gap * (len(layer_nodes) - 1)

        y = 0.5 + total_height / 2
        for node in layer_nodes:
            y -= node["h"]
            node["y"] = y
            node["in_cursor"] = y + node["h"]
            node["out_cursor"] = y + node["h"]
            y -= gap

    fig, ax = plt.subplots(figsize=(20, 9))
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for source_idx, target_idx, key, color in links:
        _draw_flow(ax, nodes[source_idx], nodes[target_idx], alluvial_counts[key], scale, color)

    max_layer = max(layer_x)

    for node in nodes:
        rect = Rectangle(
            (node["x"], node["y"]),
            node["w"],
            node["h"],
            facecolor="#f7f7f7",
            edgecolor="#404040",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(rect)

        if node["layer"] == max_layer:
            label_x = node["x"] - 0.014
            ha = "right"
        else:
            label_x = node["x"] + node["w"] + 0.014
            ha = "left"

        y_mid = node["y"] + node["h"] / 2

        ax.text(
            label_x,
            y_mid + 0.014,
            node["label"],
            ha=ha,
            va="center",
            fontsize=14,
            fontweight="bold" if node["layer"] in (0, max_layer) else "normal",
            linespacing=1.05,
            zorder=4,
        )

        ax.text(
            label_x,
            y_mid - 0.020,
            f"n={node['count_label']}",
            ha=ha,
            va="center",
            fontsize=12,
            color="#555555",
            zorder=4,
        )

    title = (
        "Identifying and Validating Misannotations "
        f"({summary.get('tissue', 'Cortex')}, gene panel size=80)"
    )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_centroid_support_columns(results):
    annotated = results.copy()

    checked = annotated["checked"].fillna(False)
    closer = annotated["corrected_is_closer"].fillna(False)
    top_n = annotated["corrected_in_top_n"].fillna(False)

    annotated["closer_to_corrected"] = checked & closer
    annotated["top_n_centroid_support"] = checked & ~closer & top_n
    annotated["centroid_supported"] = (
        annotated["closer_to_corrected"] |
        annotated["top_n_centroid_support"]
    )
    annotated["centroid_unsupported"] = checked & ~annotated["centroid_supported"]
    annotated["centroid_unchecked"] = ~checked

    return annotated


def summarize_label_transitions(results):
    annotated = add_centroid_support_columns(results)
    grouped = (
        annotated.groupby(["original_label", "corrected_label"], observed=True)
        .agg(
            n_cells=("cell_barcode", "size"),
            checked_n=("checked", "sum"),
            supported_n=("centroid_supported", "sum"),
            unsupported_n=("centroid_unsupported", "sum"),
            unchecked_n=("centroid_unchecked", "sum"),
            top_n_centroid_support_n=("top_n_centroid_support", "sum"),
            corrected_nearest_n=("corrected_is_nearest_centroid", "sum"),
            corrected_in_top_n_n=("corrected_in_top_n", "sum"),
            corrected_closer_n=("corrected_is_closer", "sum"),
        )
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    grouped["support_pct_checked"] = (
        grouped["supported_n"] / grouped["checked_n"].replace(0, np.nan) * 100
    )
    grouped["unsupported_pct_checked"] = (
        grouped["unsupported_n"] / grouped["checked_n"].replace(0, np.nan) * 100
    )
    return grouped


def save_flagged_transition_alluvial(
    transition_counts,
    out_path,
    top_n=50,
    min_cells=1,
):
    if transition_counts.empty:
        return

    plot_counts = transition_counts[transition_counts["n_cells"] >= min_cells].head(top_n).copy()
    omitted_n = int(transition_counts["n_cells"].sum() - plot_counts["n_cells"].sum())
    if omitted_n > 0:
        plot_counts = pd.concat(
            [
                plot_counts,
                pd.DataFrame([{
                    "original_label": "Other original labels",
                    "corrected_label": "Other corrected labels",
                    "n_cells": omitted_n,
                }]),
            ],
            ignore_index=True,
        )

    original_totals = (
        plot_counts.groupby("original_label", observed=True)["n_cells"]
        .sum()
        .sort_values(ascending=False)
    )
    corrected_totals = (
        plot_counts.groupby("corrected_label", observed=True)["n_cells"]
        .sum()
        .sort_values(ascending=False)
    )
    total = int(plot_counts["n_cells"].sum())
    node_w = 0.035
    left_x = 0.07
    right_x = 0.78
    nodes = []
    gap = 0.012
    min_height = 0.006
    available_height = 0.84

    def layer_scale(totals):
        if total == 0:
            return 0
        usable_height = available_height - gap * max(len(totals) - 1, 0)
        return max(usable_height, 0.1) / total

    scale = min(layer_scale(original_totals), layer_scale(corrected_totals))

    def add_nodes(totals, layer, x):
        node_gap = gap if len(totals) > 1 else 0
        heights = [max(value * scale, min_height if value > 0 else 0.0) for value in totals]
        total_height = sum(heights) + node_gap * (len(heights) - 1)
        y = 0.5 + total_height / 2
        for (label, value), height in zip(totals.items(), heights):
            y -= height
            nodes.append({
                "id": (layer, label),
                "label": f"{label}\n({_fmt_count(value)})",
                "layer": layer,
                "x": x,
                "y": y,
                "w": node_w,
                "h": height,
                "in_cursor": y + height,
                "out_cursor": y + height,
            })
            y -= node_gap

    add_nodes(original_totals, "original", left_x)
    add_nodes(corrected_totals, "corrected", right_x)
    node_lookup = {node["id"]: node for node in nodes}

    source_labels = list(original_totals.index)
    color_map = {
        label: plt.cm.tab20(i % 20)
        for i, label in enumerate(source_labels)
    }

    fig_height = max(8, 0.28 * max(len(original_totals), len(corrected_totals)))
    fig, ax = plt.subplots(figsize=(17, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for _, row in plot_counts.iterrows():
        source = node_lookup[("original", row["original_label"])]
        target = node_lookup[("corrected", row["corrected_label"])]
        _draw_flow(
            ax,
            source,
            target,
            row["n_cells"],
            scale,
            color_map[row["original_label"]],
            alpha=0.42,
        )

    for node in nodes:
        rect = Rectangle(
            (node["x"], node["y"]),
            node["w"],
            node["h"],
            facecolor="#f7f7f7",
            edgecolor="#404040",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(rect)
        is_right = node["layer"] == "corrected"
        label_x = node["x"] - 0.008 if is_right else node["x"] + node["w"] + 0.008
        ha = "right" if is_right else "left"
        ax.text(
            label_x,
            node["y"] + node["h"] / 2,
            node["label"],
            ha=ha,
            va="center",
            fontsize=8.5,
            zorder=4,
        )

    title = "Flagged Cells: Original Label to Corrected Label"
    if omitted_n > 0:
        title += f" (top {len(plot_counts) - 1} flows + other)"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    ax.text(left_x, 0.98, "Original", ha="left", va="top", fontsize=12, fontweight="bold")
    ax.text(right_x + node_w, 0.98, "Corrected", ha="right", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_flagged_transition_heatmap(transition_counts, out_path):
    if transition_counts.empty:
        return

    matrix = transition_counts.pivot(
        index="original_label",
        columns="corrected_label",
        values="n_cells",
    ).fillna(0)
    row_order = matrix.sum(axis=1).sort_values(ascending=False).index
    col_order = matrix.sum(axis=0).sort_values(ascending=False).index
    matrix = matrix.loc[row_order, col_order]

    plot_matrix = np.log1p(matrix.to_numpy())
    fig_w = max(10, 0.42 * len(matrix.columns))
    fig_h = max(8, 0.38 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(plot_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Corrected label")
    ax.set_ylabel("Original label")
    ax.set_title("Flagged Cells: Original-to-Corrected Transition Counts", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log1p(cell count)")

    max_count = int(matrix.to_numpy().max())
    annotation_threshold = max(1, max_count // 8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iat[i, j])
            if value >= annotation_threshold:
                ax.text(j, i, str(value), ha="center", va="center", color="white", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top_n_centroid_transition_heatmap(transition_counts, out_path, top_n_label=5):
    top_n_counts = transition_counts[
        transition_counts["top_n_centroid_support_n"] > 0
    ].copy()
    if top_n_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No cells in the Top {top_n_label} centroid category",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    matrix = top_n_counts.pivot(
        index="original_label",
        columns="corrected_label",
        values="top_n_centroid_support_n",
    ).fillna(0)
    row_order = matrix.sum(axis=1).sort_values(ascending=False).index
    col_order = matrix.sum(axis=0).sort_values(ascending=False).index
    matrix = matrix.loc[row_order, col_order]

    plot_matrix = np.log1p(matrix.to_numpy())
    fig_w = max(10, 0.42 * len(matrix.columns))
    fig_h = max(8, 0.38 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(plot_matrix, cmap="Greens", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Corrected label")
    ax.set_ylabel("Original label")
    ax.set_title(
        f"Flagged Cells in Top {top_n_label} Centroid Category by Label Transition",
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"log1p(Top {top_n_label} category cell count)")

    max_count = int(matrix.to_numpy().max())
    annotation_threshold = max(1, max_count // 8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iat[i, j])
            if value >= annotation_threshold:
                ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top_n_centroid_distance_scatter(results, out_path, top_n_label=5):
    annotated = add_centroid_support_columns(results)
    plot_df = annotated[annotated["top_n_centroid_support"]].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No cells in the Top {top_n_label} centroid category",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    x = plot_df["original_centroid_distance"].to_numpy(dtype=float)
    y = plot_df["corrected_centroid_distance"].to_numpy(dtype=float)
    ranks = plot_df["corrected_centroid_rank"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(ranks)
    x = x[finite]
    y = y[finite]
    ranks = ranks[finite]

    axis_max = float(max(np.max(x), np.max(y))) if len(x) else 1.0
    axis_pad = axis_max * 0.04
    axis_limit = axis_max + axis_pad

    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        x,
        y,
        c=ranks,
        cmap="viridis_r",
        vmin=1,
        vmax=top_n_label,
        s=18,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    ax.plot([0, axis_limit], [0, axis_limit], color="#202020", linewidth=1.5, linestyle="--")
    ax.set_xlim(0, axis_limit)
    ax.set_ylim(0, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance to annotated centroid (human)")
    ax.set_ylabel("Distance to predicted centroid (model)")
    ax.set_title(
        f"Top {top_n_label} Centroid Category: Cell-Level Distances",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(alpha=0.2)
    ax.text(
        axis_limit * 0.97,
        axis_limit * 0.03,
        "Predicted centroid closer",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#238b45",
    )
    ax.text(
        axis_limit * 0.03,
        axis_limit * 0.97,
        "Annotated centroid closer",
        ha="left",
        va="top",
        fontsize=9,
        color="#b24a1b",
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted centroid rank")
    cbar.set_ticks(np.arange(1, top_n_label + 1))

    above_n = int(np.sum(y > x))
    below_n = int(np.sum(y < x))
    ax.text(
        0.02,
        0.02,
        f"n={len(x):,} | above line={above_n:,} | below line={below_n:,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_unsupported_transition_plot(transition_counts, out_path, top_n=35):
    if transition_counts.empty:
        return

    unsupported_counts = transition_counts[transition_counts["unsupported_n"] > 0].copy()
    if unsupported_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No unsupported flagged label transitions",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    unsupported_counts = unsupported_counts.sort_values(
        ["unsupported_n", "n_cells"],
        ascending=False,
    )
    plot_df = unsupported_counts.head(top_n).copy()
    omitted = unsupported_counts.iloc[top_n:]
    if not omitted.empty:
        other = {
            "original_label": "Other original labels",
            "corrected_label": "Other corrected labels",
            "n_cells": int(omitted["n_cells"].sum()),
            "supported_n": int(omitted["supported_n"].sum()),
            "unsupported_n": int(omitted["unsupported_n"].sum()),
            "unchecked_n": int(omitted["unchecked_n"].sum()),
        }
        other["checked_n"] = other["supported_n"] + other["unsupported_n"]
        other["unsupported_pct_checked"] = (
            other["unsupported_n"] / other["checked_n"] * 100
            if other["checked_n"] else np.nan
        )
        plot_df = pd.concat([plot_df, pd.DataFrame([other])], ignore_index=True)

    plot_df["transition"] = plot_df["original_label"] + " -> " + plot_df["corrected_label"]
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(plot_df))
    fig_h = max(8, 0.34 * len(plot_df))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    unsupported = plot_df["unsupported_n"].to_numpy()
    ax.barh(y, unsupported, color="#f16913")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["transition"], fontsize=8)
    ax.set_xlabel("Unsupported flagged cells")
    ax.set_title(
        "Unsupported Flagged Label Transitions",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)

    max_unsupported = int(plot_df["unsupported_n"].max()) if not plot_df.empty else 0
    ax.set_xlim(0, max_unsupported * 1.35 if max_unsupported else 1)
    for i, row in plot_df.iterrows():
        total = int(row["n_cells"])
        unsupported_n = int(row["unsupported_n"])
        pct = row["unsupported_pct_checked"]
        pct_label = "NA" if pd.isna(pct) else f"{pct:.0f}%"
        ax.text(
            unsupported_n + max_unsupported * 0.015,
            i,
            f"{unsupported_n}/{total} ({pct_label})",
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top_selected_genes_heatmap(adata, out_path, top_n=5):
    gene_cols = [
        col for col in adata.obs.columns
        if col.startswith("selected_gene_") or col.startswith("top_gene_")
    ]

    if not gene_cols:
        print("No selected gene columns found in adata.obs.")
        print("Available obs columns containing 'gene':")
        print([col for col in adata.obs.columns if "gene" in col.lower()])
        return

    gene_cols = gene_cols[:top_n]

    gene_counts = []
    for col in gene_cols:
        counts = adata.obs[col].dropna().astype(str).value_counts()
        for gene, count in counts.items():
            gene_counts.append({
                "rank": col,
                "gene": gene,
                "count": int(count),
            })

    gene_df = pd.DataFrame(gene_counts)
    if gene_df.empty:
        print("Selected gene columns were found, but they were empty.")
        return

    top_genes = (
        gene_df.groupby("gene")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )

    matrix = (
        gene_df[gene_df["gene"].isin(top_genes)]
        .pivot_table(
            index="gene",
            columns="rank",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
    )

    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]

    fig_h = max(8, 0.35 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(8, fig_h))

    im = ax.imshow(np.log1p(matrix.to_numpy()), cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(matrix.index, fontsize=9)

    ax.set_xlabel("Selected gene rank")
    ax.set_ylabel("Gene")
    ax.set_title(f"Top {top_n} Selected Genes", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log1p(selection count)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iat[i, j])
            if value > 0:
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved top selected genes heatmap: {out_path}")

def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.adata_path), "centroid_check")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.adata_path}")
    adata = sc.read_h5ad(args.adata_path)
    validate_inputs(adata, args.original_col, args.corrected_col, args.embedding_key)

    # Recompute PCA exactly like the poison experiment.
    if "raw_counts" not in adata.layers:
        raise ValueError("Expected adata.layers['raw_counts'], but it was not found.")

    adata.X = adata.layers["raw_counts"].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")

    coords = np.asarray(adata.obsm["X_pca"])
    obs = adata.obs[[args.original_col, args.corrected_col]].copy()
    obs[args.original_col] = clean_label_series(obs[args.original_col])
    obs[args.corrected_col] = clean_label_series(obs[args.corrected_col])

    valid_label_mask = obs[args.original_col].notna() & obs[args.corrected_col].notna()
    flagged_mask = valid_label_mask & (obs[args.original_col] != obs[args.corrected_col])
    centroid_mask = ~flagged_mask if args.exclude_flagged_from_centroids else np.ones(len(obs), dtype=bool)
    centroid_labels = obs.loc[centroid_mask, args.original_col].to_numpy()
    centroids, reference_counts = build_centroids(
        coords[centroid_mask],
        centroid_labels,
        args.centroid_method,
        args.min_centroid_cells,
    )
    centroid_labels_ordered = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[label] for label in centroid_labels_ordered]) if centroids else None

    rows = []
    flagged_indices = np.flatnonzero(flagged_mask.to_numpy())
    for idx in flagged_indices:
        cell_barcode = obs.index[idx]
        original_label = obs.iloc[idx][args.original_col]
        corrected_label = obs.iloc[idx][args.corrected_col]
        cell_coords = coords[idx]

        original_centroid = centroids.get(original_label)
        corrected_centroid = centroids.get(corrected_label)
        if original_centroid is None or corrected_centroid is None:
            missing = []
            if original_centroid is None:
                missing.append("original_label_centroid")
            if corrected_centroid is None:
                missing.append("corrected_label_centroid")
            rows.append({
                "cell_barcode": cell_barcode,
                "original_label": original_label,
                "corrected_label": corrected_label,
                "original_reference_n": reference_counts.get(original_label, 0),
                "corrected_reference_n": reference_counts.get(corrected_label, 0),
                "checked": False,
                "unchecked_reason": ";".join(missing),
                "original_centroid_distance": np.nan,
                "corrected_centroid_distance": np.nan,
                "distance_delta_original_minus_corrected": np.nan,
                "corrected_is_closer": False,
                "nearest_centroid_label": pd.NA,
                "nearest_centroid_distance": np.nan,
                "corrected_centroid_rank": np.nan,
                "corrected_is_nearest_centroid": False,
                "corrected_in_top_n": False,
            })
            continue

        original_dist = float(np.linalg.norm(cell_coords - original_centroid))
        corrected_dist = float(np.linalg.norm(cell_coords - corrected_centroid))
        all_distances = np.linalg.norm(centroid_matrix - cell_coords, axis=1)
        nearest_order = np.argsort(all_distances)
        nearest_label = centroid_labels_ordered[nearest_order[0]]
        corrected_rank = int(np.where(np.array(centroid_labels_ordered)[nearest_order] == corrected_label)[0][0] + 1)

        rows.append({
            "cell_barcode": cell_barcode,
            "original_label": original_label,
            "corrected_label": corrected_label,
            "original_reference_n": reference_counts.get(original_label, 0),
            "corrected_reference_n": reference_counts.get(corrected_label, 0),
            "checked": True,
            "unchecked_reason": "",
            "original_centroid_distance": original_dist,
            "corrected_centroid_distance": corrected_dist,
            "distance_delta_original_minus_corrected": original_dist - corrected_dist,
            "corrected_is_closer": corrected_dist < original_dist,
            "nearest_centroid_label": nearest_label,
            "nearest_centroid_distance": float(all_distances[nearest_order[0]]),
            "corrected_centroid_rank": corrected_rank,
            "corrected_is_nearest_centroid": nearest_label == corrected_label,
            "corrected_in_top_n": corrected_rank <= args.top_n_nearest,
        })

    results = pd.DataFrame(rows)
    per_cell_csv = os.path.join(out_dir, "flagged_cell_centroid_distances.csv")
    transition_csv = os.path.join(out_dir, "flagged_cell_centroid_summary_by_transition.csv")
    summary_csv = os.path.join(out_dir, "flagged_cell_centroid_summary.csv")
    plot_png = os.path.join(out_dir, "flagged_cell_centroid_summary.png")
    alluvial_png = os.path.join(out_dir, "flagged_cell_centroid_alluvial.png")
    alluvial_counts_csv = os.path.join(out_dir, "flagged_cell_centroid_alluvial_counts.csv")
    label_transition_csv = os.path.join(out_dir, "flagged_cell_label_transition_counts.csv")
    label_transition_alluvial_png = os.path.join(out_dir, "flagged_cell_label_transition_alluvial.png")
    label_transition_heatmap_png = os.path.join(out_dir, "flagged_cell_label_transition_heatmap.png")
    top_n_centroid_transition_heatmap_png = os.path.join(
        out_dir,
        f"flagged_cell_top_{args.top_n_nearest}_centroid_transition_heatmap.png",
    )
    top_n_centroid_distance_scatter_png = os.path.join(
        out_dir,
        f"flagged_cell_top_{args.top_n_nearest}_centroid_distance_scatter.png",
    )
    top_selected_genes_heatmap_png = os.path.join(
        out_dir,
        "top_5_selected_genes_heatmap.png",
    )

    if results.empty:
        summary = {
            "adata_path": args.adata_path,
            "total_cells": adata.n_obs,
            "flagged_cells": int(flagged_mask.sum()),
            "checked_flagged_cells": 0,
            "unchecked_flagged_cells": int(flagged_mask.sum()),
            "corrected_closer_n": 0,
            "corrected_closer_pct": 0.0,
            "corrected_nearest_n": 0,
            "corrected_nearest_pct": 0.0,
            "corrected_in_top_n_n": 0,
            "corrected_in_top_n_pct": 0.0,
            "centroids_exclude_flagged": args.exclude_flagged_from_centroids,
            "centroid_method": args.centroid_method,
            "min_centroid_cells": args.min_centroid_cells,
            "top_n_nearest": args.top_n_nearest,
            "embedding_key": args.embedding_key,
        }
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        save_centroid_alluvial(summary, alluvial_png, alluvial_counts_csv, checked_results)
        print("No flagged cells with valid original and corrected centroids were found.")
        print(f"Saved summary: {summary_csv}")
        print(f"Saved alluvial plot: {alluvial_png}")
        return

    transition_summary = summarize_by_transition(results)
    label_transition_counts = summarize_label_transitions(results)
    checked_results = results[results["checked"]].copy()
    unchecked_n = int((~results["checked"]).sum())
    if checked_results.empty:
        closer_pct = nearest_pct = top_n_pct = 0.0
        median_original = median_corrected = median_delta = mean_delta = np.nan
    else:
        closer_pct = float(checked_results["corrected_is_closer"].mean() * 100)
        nearest_pct = float(checked_results["corrected_is_nearest_centroid"].mean() * 100)
        top_n_pct = float(checked_results["corrected_in_top_n"].mean() * 100)
        median_original = float(checked_results["original_centroid_distance"].median())
        median_corrected = float(checked_results["corrected_centroid_distance"].median())
        median_delta = float(checked_results["distance_delta_original_minus_corrected"].median())
        mean_delta = float(checked_results["distance_delta_original_minus_corrected"].mean())

    summary = {
        "adata_path": args.adata_path,
        "total_cells": adata.n_obs,
        "flagged_cells": int(flagged_mask.sum()),
        "checked_flagged_cells": len(checked_results),
        "unchecked_flagged_cells": unchecked_n,
        "corrected_closer_n": int(checked_results["corrected_is_closer"].sum()),
        "corrected_closer_pct": closer_pct,
        "corrected_nearest_n": int(checked_results["corrected_is_nearest_centroid"].sum()),
        "corrected_nearest_pct": nearest_pct,
        "corrected_in_top_n_n": int(checked_results["corrected_in_top_n"].sum()),
        "corrected_in_top_n_pct": top_n_pct,
        "median_original_distance": median_original,
        "median_corrected_distance": median_corrected,
        "median_distance_delta": median_delta,
        "mean_distance_delta": mean_delta,
        "centroids_exclude_flagged": args.exclude_flagged_from_centroids,
        "centroid_method": args.centroid_method,
        "min_centroid_cells": args.min_centroid_cells,
        "top_n_nearest": args.top_n_nearest,
        "embedding_key": args.embedding_key,
    }

    results.to_csv(per_cell_csv, index=False)
    transition_summary.to_csv(transition_csv, index=False)
    label_transition_counts.to_csv(label_transition_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    save_summary_plot(summary, plot_png)
    save_centroid_alluvial(summary, alluvial_png, alluvial_counts_csv, checked_results)
    save_flagged_transition_alluvial(
        label_transition_counts,
        label_transition_alluvial_png,
        top_n=args.transition_plot_top_n,
        min_cells=args.transition_plot_min_cells,
    )
    save_flagged_transition_heatmap(label_transition_counts, label_transition_heatmap_png)
    save_top_n_centroid_transition_heatmap(
        label_transition_counts,
        top_n_centroid_transition_heatmap_png,
        top_n_label=args.top_n_nearest,
    )
    save_top_n_centroid_distance_scatter(
        results,
        top_n_centroid_distance_scatter_png,
        top_n_label=args.top_n_nearest,
    )
    save_top_selected_genes_heatmap(
        adata,
        top_selected_genes_heatmap_png,
        top_n=5,
    )

    print(f"Flagged cells: {summary['flagged_cells']:,}")
    print(f"Checked flagged cells: {summary['checked_flagged_cells']:,}")
    if summary["unchecked_flagged_cells"]:
        print(f"Unchecked flagged cells: {summary['unchecked_flagged_cells']:,}")
    print(
        "Corrected centroid closer: "
        f"{summary['corrected_closer_n']:,}/{summary['checked_flagged_cells']:,} "
        f"({summary['corrected_closer_pct']:.1f}%)"
    )
    print(
        "Corrected centroid nearest: "
        f"{summary['corrected_nearest_n']:,}/{summary['checked_flagged_cells']:,} "
        f"({summary['corrected_nearest_pct']:.1f}%)"
    )
    print(
        f"Corrected centroid in top {summary['top_n_nearest']}: "
        f"{summary['corrected_in_top_n_n']:,}/{summary['checked_flagged_cells']:,} "
        f"({summary['corrected_in_top_n_pct']:.1f}%)"
    )
    print(f"Saved per-cell distances: {per_cell_csv}")
    print(f"Saved transition summary: {transition_csv}")
    print(f"Saved label transition counts: {label_transition_csv}")
    print(f"Saved overall summary: {summary_csv}")
    print(f"Saved plot: {plot_png}")
    print(f"Saved alluvial counts: {alluvial_counts_csv}")
    print(f"Saved alluvial plot: {alluvial_png}")
    print(f"Saved label transition alluvial: {label_transition_alluvial_png}")
    print(f"Saved label transition heatmap: {label_transition_heatmap_png}")
    print(f"Saved Top {args.top_n_nearest} centroid transition heatmap: {top_n_centroid_transition_heatmap_png}")
    print(f"Saved Top {args.top_n_nearest} centroid distance scatter: {top_n_centroid_distance_scatter_png}")
    print(f"Saved top selected genes heatmap: {top_selected_genes_heatmap_png}")


if __name__ == "__main__":
    main()
