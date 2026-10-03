#!/usr/bin/env python3
"""
Poison Injector: Artificially misannotates single-cell data for ground-truth testing.

python3 poison_injector.py --adata-in /home/nikki/celltype_compression_2/B_cell_ground_truth/BCellGroundTruth_B_cell_ground_truth_subset.h5ad --adata-out /home/nikki/celltype_compression_2/B_cell_ground_truth/BCellGroundTruth_B_cell_ground_truth_subset_POISONED.h5ad --celltype-col celltype --poison-fraction 0.10
"""
import argparse
import pandas as pd
import scanpy as sc
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata-in", required=True, help="Clean h5ad file")
    parser.add_argument("--adata-out", required=True, help="Where to save poisoned h5ad")
    parser.add_argument("--celltype-col", default="celltype")
    parser.add_argument("--poison-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    print(f"Loading {args.adata_in}...")
    adata = sc.read_h5ad(args.adata_in)
    
    original_labels = adata.obs[args.celltype_col].astype(str).values
    unique_labels = np.unique(original_labels)
    n_cells = len(original_labels)
    
    # Determine which cells get poisoned
    n_poison = int(n_cells * args.poison_fraction)
    poison_indices = np.random.choice(n_cells, n_poison, replace=False)
    
    new_labels = original_labels.copy()
    
    print(f"Injecting poison into {n_poison} cells ({args.poison_fraction*100}%)...")
    for idx in poison_indices:
        true_label = original_labels[idx]
        # Pick a random label that is NOT the true label
        wrong_labels = unique_labels[unique_labels != true_label]
        new_labels[idx] = np.random.choice(wrong_labels)
        
    # Create the poisoned column
    poisoned_col_name = f"{args.celltype_col}_poisoned"
    adata.obs[poisoned_col_name] = new_labels
    
    # Save the Ground Truth Key so we can grade the sieve later
    truth_key = pd.DataFrame({
        "cell_barcode": adata.obs_names,
        "true_label": original_labels,
        "poisoned_label": new_labels,
        "is_poisoned": np.isin(np.arange(n_cells), poison_indices)
    })
    
    truth_csv = args.adata_out.replace('.h5ad', '_truth_key.csv')
    truth_key.to_csv(truth_csv, index=False)
    
    print(f"Saving poisoned dataset to {args.adata_out}")
    adata.write_h5ad(args.adata_out)
    print(f"Saved Ground Truth Key to {truth_csv}")

if __name__ == "__main__":
    main()