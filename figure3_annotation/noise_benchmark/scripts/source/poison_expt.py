#!/usr/bin/env python3
"""
Automated Poison Sweep Orchestrator with Centroid Validation
Runs the poisoner, the sieve, the grader, and the spatial centroid analysis.
"""
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import scanpy as sc

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
DATASETS = {
    #"B_cell": "/home/nikki/celltype_compression_2/B_cell_ground_truth/BCellGroundTruth_B_cell_ground_truth_subset.h5ad",
    "Cortex": "/home/nikki/celltype_compression_2/CortexHippocampus/B6J_CortexHippocampus_subset.h5ad",
    #"Gastrocnemius": "/home/nikki/celltype_compression_2/Gastrocnemius/B6J_Gastrocnemius_subset.h5ad",
    #"Heart": "/home/nikki/celltype_compression_2/Heart/B6J_Heart_subset.h5ad"
}

OUT_BASE_DIR = "/home/nikki/celltype_compression_2/poison_expt_alphas_output/"

# The ONE configuration you care about right now
TARGET_POISON = 0.10
TARGET_SIZE = 80
TARGET_ALPHA = 0.7
N_WORKERS = 16
CENTROID_METHOD = "median"
MIN_CENTROID_CELLS = 20
TOP_N_NEAREST = 5


def _fmt_count(value):
    return f"{int(value):,}"


def build_label_centroids(coords, labels, method="median", min_cells=20):
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


def centroid_check_flagged_cells(flagged_rows, adata, coords, centroids, reference_counts):
    labels_ordered = list(centroids.keys())
    if not labels_ordered:
        return pd.DataFrame()

    centroid_matrix = np.vstack([centroids[label] for label in labels_ordered])
    label_array = np.array(labels_ordered)
    rows = []

    for _, row in flagged_rows.iterrows():
        cell_barcode = row["cell_barcode"]
        annotated_label = row["annotated_label"]
        predicted_label = row["predicted_label"]
        try:
            cell_idx = adata.obs_names.get_loc(cell_barcode)
        except KeyError:
            rows.append({
                "cell_barcode": cell_barcode,
                "annotated_label": annotated_label,
                "predicted_label": predicted_label,
                "annotated_reference_n": reference_counts.get(annotated_label, 0),
                "predicted_reference_n": reference_counts.get(predicted_label, 0),
                "checked": False,
                "unchecked_reason": "missing_cell_barcode",
                "annotated_centroid_distance": np.nan,
                "predicted_centroid_distance": np.nan,
                "distance_delta_annotated_minus_predicted": np.nan,
                "predicted_is_closer": False,
                "nearest_centroid_label": pd.NA,
                "nearest_centroid_distance": np.nan,
                "predicted_centroid_rank": np.nan,
                "predicted_in_top_n": False,
            })
            continue

        missing = []
        annotated_centroid = centroids.get(annotated_label)
        predicted_centroid = centroids.get(predicted_label)
        if annotated_centroid is None:
            missing.append("annotated_label_centroid")
        if predicted_centroid is None:
            missing.append("predicted_label_centroid")
        if missing:
            rows.append({
                "cell_barcode": cell_barcode,
                "annotated_label": annotated_label,
                "predicted_label": predicted_label,
                "annotated_reference_n": reference_counts.get(annotated_label, 0),
                "predicted_reference_n": reference_counts.get(predicted_label, 0),
                "checked": False,
                "unchecked_reason": ";".join(missing),
                "annotated_centroid_distance": np.nan,
                "predicted_centroid_distance": np.nan,
                "distance_delta_annotated_minus_predicted": np.nan,
                "predicted_is_closer": False,
                "nearest_centroid_label": pd.NA,
                "nearest_centroid_distance": np.nan,
                "predicted_centroid_rank": np.nan,
                "predicted_in_top_n": False,
            })
            continue

        cell_coords = coords[cell_idx]
        annotated_dist = float(np.linalg.norm(cell_coords - annotated_centroid))
        predicted_dist = float(np.linalg.norm(cell_coords - predicted_centroid))
        all_distances = np.linalg.norm(centroid_matrix - cell_coords, axis=1)
        nearest_order = np.argsort(all_distances)
        predicted_rank = int(np.where(label_array[nearest_order] == predicted_label)[0][0] + 1)
        nearest_label = labels_ordered[nearest_order[0]]

        rows.append({
            "cell_barcode": cell_barcode,
            "annotated_label": annotated_label,
            "predicted_label": predicted_label,
            "annotated_reference_n": reference_counts.get(annotated_label, 0),
            "predicted_reference_n": reference_counts.get(predicted_label, 0),
            "checked": True,
            "unchecked_reason": "",
            "annotated_centroid_distance": annotated_dist,
            "predicted_centroid_distance": predicted_dist,
            "distance_delta_annotated_minus_predicted": annotated_dist - predicted_dist,
            "predicted_is_closer": predicted_dist < annotated_dist,
            "nearest_centroid_label": nearest_label,
            "nearest_centroid_distance": float(all_distances[nearest_order[0]]),
            "predicted_centroid_rank": predicted_rank,
            "predicted_in_top_n": predicted_rank <= TOP_N_NEAREST,
        })

    return pd.DataFrame(rows)


def _draw_flow(ax, source, target, value, scale, color, alpha=0.45):
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
    dx = (x1 - x0) * 0.60

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

    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
    )


def plot_poison_alluvial(flow_metrics, out_dir):
    """Render one cleaner alluvial/Sankey-style plot per tissue."""
    if not flow_metrics:
        return []

    node_specs = [
        ("Total", "total_cells", 0),
        ("Poisoned", "poisoned", 1),
        ("Clean", "clean", 1),
        ("Poison\nflagged", "poison_flagged", 2),
        ("Missed", "poison_missed", 2),
        ("Clean\nflagged", "clean_flagged", 2),
        ("Unflagged", "clean_unflagged", 2),
        ("Recovered", "recovered", 3),
        ("Wrong\nfix", "wrong_fix", 3),
        ("Human\nerror", "human_error_caught", 3),
        ("Collateral", "collateral", 3),
        ("Good", "total_good", 4),
        ("Bad", "total_bad", 4),
    ]

    links = [
        (0, 1, "poisoned", "#d95f02"),
        (0, 2, "clean", "#4c78a8"),
        (1, 3, "poison_flagged", "#d95f02"),
        (1, 4, "poison_missed", "#9e9e9e"),
        (2, 5, "clean_flagged", "#4c78a8"),
        (2, 6, "clean_unflagged", "#b7c9df"),
        (3, 7, "recovered", "#2ca25f"),
        (3, 8, "wrong_fix", "#de2d26"),
        (5, 9, "human_error_caught", "#41ab5d"),
        (5, 10, "collateral", "#f16913"),
        (7, 11, "recovered", "#2ca25f"),
        (9, 11, "human_error_caught", "#41ab5d"),
        (8, 12, "wrong_fix", "#de2d26"),
        (10, 12, "collateral", "#f16913"),
    ]

    layer_x = {
        0: 0.03,
        1: 0.23,
        2: 0.50,
        3: 0.76,
        4: 0.95,
    }

    node_w = 0.055
    saved_paths = []

    for row in flow_metrics:
        row = row.copy()
        row["total_good"] = row["recovered"] + row["human_error_caught"]
        row["total_bad"] = row["wrong_fix"] + row["collateral"]

        total = row["total_cells"]
        scale = 0.74 / total if total else 0

        nodes = []
        for label, key, layer in node_specs:
            value = row[key]
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
            gap = 0.065 if len(layer_nodes) > 1 else 0

            total_height = sum(node["h"] for node in layer_nodes) + gap * (len(layer_nodes) - 1)

            # If a layer is too tall, shrink only the gaps so labels do not overflow.
            if total_height > 0.90 and len(layer_nodes) > 1:
                gap = max((0.90 - sum(node["h"] for node in layer_nodes)) / (len(layer_nodes) - 1), 0.025)
                total_height = sum(node["h"] for node in layer_nodes) + gap * (len(layer_nodes) - 1)

            y = 0.5 + total_height / 2
            for node in layer_nodes:
                y -= node["h"]
                node["y"] = y
                node["in_cursor"] = y + node["h"]
                node["out_cursor"] = y + node["h"]
                y -= gap

        fig, ax = plt.subplots(figsize=(22, 10))
        ax.set_xlim(0, 1.04)
        ax.set_ylim(0, 1)
        ax.axis("off")

        for source_idx, target_idx, key, color in links:
            _draw_flow(ax, nodes[source_idx], nodes[target_idx], row[key], scale, color)

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
                fontsize=15,
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
                fontsize=13,
                color="#555555",
                zorder=4,
            )

        title = (
            f"{row['Tissue']} Corruption Experiment "
            f"(Gene panel size={TARGET_SIZE}, Poison={TARGET_POISON*100:.0f}%)"
        )

        ax.set_title(title, fontsize=20, fontweight="bold", pad=22)
        fig.tight_layout()

        base = os.path.join(out_dir, f"{row['Tissue']}_poison_alluvial")
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base}.pdf", bbox_inches="tight")
        plt.close(fig)

        saved_paths.extend([f"{base}.png", f"{base}.pdf"])

    if len(flow_metrics) > 1:
        print("Note: combined alluvial panels are not generated; per-tissue alluvials were saved.")

    return saved_paths

    
def main():
    os.makedirs(OUT_BASE_DIR, exist_ok=True)
    all_metrics = []
    flow_metrics = []

    for tissue_name, clean_h5ad in DATASETS.items():
        print(f"\n{'='*60}\nANALYZING TISSUE: {tissue_name}\n{'='*60}")
        
        # --- 1. PRECOMPUTE PCA CENTROIDS ---
        print(f"Loading {tissue_name} to calculate spatial centroids...")
        adata = sc.read_h5ad(clean_h5ad)
        adata.X = adata.layers['raw_counts']
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack')
            
        pca_coords = adata.obsm['X_pca']
        
        # Paths
        frac_str = f"{TARGET_POISON:.2f}".replace('.', '')
        poisoned_h5ad = clean_h5ad.replace(".h5ad", f"_POISONED_{frac_str}.h5ad")
        truth_key_csv = poisoned_h5ad.replace(".h5ad", "_truth_key.csv")
        sieve_out_dir = os.path.join(OUT_BASE_DIR, f"{tissue_name}_POISONED_{frac_str}")
        
        # --- 2. RUN POISONER ---
        print(f">>> Injecting {TARGET_POISON*100}% Poison...")
        subprocess.run(["python3", "poison_injector.py", "--adata-in", clean_h5ad, "--adata-out", poisoned_h5ad, 
                        "--celltype-col", "celltype", "--poison-fraction", str(TARGET_POISON)], check=True)
        
        # --- 3. RUN SIEVE ---
        print(f">>> Running Sieve (Size: {TARGET_SIZE}, Alpha: {TARGET_ALPHA})...")
        subprocess.run(["python3", "cell_sieve_v4.py", "--adata-path", poisoned_h5ad, "--out-dir", sieve_out_dir, 
                        "--panel-size", str(TARGET_SIZE), "--alpha", str(TARGET_ALPHA), "--n-workers", str(N_WORKERS), 
                        "--detect-misannotations", "--celltype-column", "celltype_poisoned"], check=True)
        
        alpha_str = f"{TARGET_ALPHA:.2f}"
        flagged_csv = os.path.join(sieve_out_dir, f"panel_size_{TARGET_SIZE}_ElasticNet_Unconstrained_alpha_{alpha_str}", "flagged_misannotations.csv")
        
        # --- 4. INTERNAL METRIC CALCULATION ---
        print(">>> Extracting Exact Metrics...")
        truth_df = pd.read_csv(truth_key_csv)
        flagged_df = pd.read_csv(flagged_csv)
        merged_df = flagged_df.merge(truth_df, on="cell_barcode")
        flagged_barcodes = set(merged_df["cell_barcode"])
        
        total_cells = len(truth_df)
        total_poison_injected = int(truth_df["is_poisoned"].sum())
        total_clean = total_cells - total_poison_injected
        
        # Split data
        true_positives = merged_df[merged_df["is_poisoned"] == True]
        false_positives = merged_df[merged_df["is_poisoned"] == False]
        
        # 1-5: The Count Metrics
        total_flagged = len(merged_df)
        flagged_poisonous = len(true_positives)
        clean_flagged = len(false_positives)
        perfect_recovery = len(true_positives[true_positives["predicted_label"] == true_positives["true_label"]])
        wrong_recovery = len(true_positives[true_positives["predicted_label"] != true_positives["true_label"]])
        poison_missed = total_poison_injected - flagged_poisonous
        clean_unflagged = total_clean - clean_flagged
        
        # 6: Robust centroid distance metric for clean flagged cells.
        reference_df = truth_df[
            (~truth_df["is_poisoned"]) &
            (~truth_df["cell_barcode"].isin(flagged_barcodes))
        ].copy()
        reference_positions = adata.obs_names.get_indexer(reference_df["cell_barcode"])
        valid_reference = reference_positions >= 0
        reference_coords = pca_coords[reference_positions[valid_reference]]
        reference_labels = reference_df.loc[valid_reference, "true_label"].to_numpy()
        centroids, reference_counts = build_label_centroids(
            reference_coords,
            reference_labels,
            method=CENTROID_METHOD,
            min_cells=MIN_CENTROID_CELLS,
        )
        centroid_results = centroid_check_flagged_cells(
            false_positives,
            adata,
            pca_coords,
            centroids,
            reference_counts,
        )
        checked_centroid_results = centroid_results[centroid_results["checked"]] if not centroid_results.empty else centroid_results
        closer_count = int(checked_centroid_results["predicted_is_closer"].sum()) if not checked_centroid_results.empty else 0
        centroid_checked = len(checked_centroid_results)
        centroid_unchecked = clean_flagged - centroid_checked
        centroid_top_n_count = int(checked_centroid_results["predicted_in_top_n"].sum()) if not checked_centroid_results.empty else 0
        if not centroid_results.empty:
            centroid_results.insert(0, "Tissue", tissue_name)
            centroid_results.to_csv(
                os.path.join(sieve_out_dir, "clean_false_positive_centroid_check.csv"),
                index=False,
            )
                
        pct_closer = (closer_count / centroid_checked * 100) if centroid_checked > 0 else 0
        pct_top_n = (centroid_top_n_count / centroid_checked * 100) if centroid_checked > 0 else 0
        catch_rate = (flagged_poisonous / total_poison_injected * 100) if total_poison_injected > 0 else 0
        recovery_rate = (perfect_recovery / flagged_poisonous * 100) if flagged_poisonous > 0 else 0
        
        all_metrics.append({
            "Tissue": tissue_name,
            "Total Flagged": total_flagged,
            "Poison Caught": flagged_poisonous,
            "Clean Flagged\n(False Pos)": clean_flagged,
            "Perfect\nRecovery": perfect_recovery,
            "Wrong\nRecovery": wrong_recovery,
            "Catch Rate %": catch_rate,
            "Recovery Rate %": recovery_rate,
            "Centroid Checked": centroid_checked,
            "Centroid Unchecked": centroid_unchecked,
            "Centroid Valid %": pct_closer,
            f"Centroid Top {TOP_N_NEAREST} %": pct_top_n,
            "Centroid Method": CENTROID_METHOD,
            "Min Centroid Cells": MIN_CENTROID_CELLS
        })
        flow_metrics.append({
            "Tissue": tissue_name,
            "total_cells": total_cells,
            "poisoned": total_poison_injected,
            "clean": total_clean,
            "poison_flagged": flagged_poisonous,
            "poison_missed": poison_missed,
            "clean_flagged": clean_flagged,
            "clean_unflagged": clean_unflagged,
            "recovered": perfect_recovery,
            "wrong_fix": wrong_recovery,
            "human_error_caught": closer_count,
            "collateral": clean_flagged - closer_count,
        })

    # ==========================================
    # 5. GENERATE ALLUVIAL PLOTS
    # ==========================================
    print("\n>>> Generating Alluvial Plot...")
    df = pd.DataFrame(all_metrics)
    flow_df = pd.DataFrame(flow_metrics)
    saved_plots = plot_poison_alluvial(flow_metrics, OUT_BASE_DIR)
    for plot_path in saved_plots:
        print(f"Saved alluvial plot: {plot_path}")
    
    # Save the numbers to CSV just in case
    df.to_csv(os.path.join(OUT_BASE_DIR, "targeted_metrics_summary.csv"), index=False)
    flow_df.to_csv(os.path.join(OUT_BASE_DIR, "targeted_alluvial_counts.csv"), index=False)

if __name__ == "__main__":
    main()
