"""
Find genes that are highly correlated with the 
37 MILP-selected genes, which could be used as substitutes
in future MILP runs to find alternative optimal solutions.

python3 redundant_solutions.py \
  --adata-path /home/nikki/celltype_compression_2/CortexHippocampus/celltype_B6_subset.h5ad \
  --selected-genes /home/nikki/celltype_compression_2/cell_sieve_v6_output/CortexHippocampus_HCR/panel_size_4_ElasticNet_Unconstrained_alpha_0.70/selected_genes.txt \
  --corr-threshold 0.75 \
  --out-csv /home/nikki/celltype_compression_2/cell_sieve_v6_output/CortexHippocampus_HCR/panel_size_4_ElasticNet_Unconstrained_alpha_0.70/redundant_gene_swaps.csv

"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc

def as_dense_array(matrix, dtype=float):
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=dtype)

def parse_args():
    parser = argparse.ArgumentParser(description="Find highly correlated redundant genes for MILP swaps.")
    parser.add_argument("--adata-path", required=True, help="Path to your .h5ad file")
    parser.add_argument("--selected-genes", required=True, help="Path to the selected_genes.txt output from cell sieve")
    parser.add_argument("--corr-threshold", type=float, default=0.75, help="Minimum Pearson R to consider a valid swap")
    parser.add_argument("--out-csv", default="./auto_milp_output/CortexHippocampus/redundant_gene_swaps.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path)
    adata.layers['raw_counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Load the 37 MILP genes
    selected_genes = pd.read_csv(args.selected_genes, header=None)[0].tolist()
    all_genes = adata.var_names.tolist()
    
    print(f"Extracting expression matrices (Cells: {adata.n_obs})...")
    X_all = as_dense_array(adata.X)
    
    # Get column indices for the selected genes to avoid duplicating memory
    sel_idx = [all_genes.index(g) for g in selected_genes if g in all_genes]
    X_sel = X_all[:, sel_idx]
    
    print("Standardizing matrices for fast correlation computation...")
    # (X - mean) / std
    X_all_std = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-8)
    X_sel_std = (X_sel - X_sel.mean(axis=0)) / (X_sel.std(axis=0) + 1e-8)
    
    print("Calculating full Pearson correlation matrix...")
    # Vectorized Pearson correlation: (X_all^T * X_sel) / N
    # Resulting matrix shape: (N_all_genes, N_selected_genes)
    corr_matrix = np.dot(X_all_std.T, X_sel_std) / float(X_all.shape[0])
    
    print(f"Filtering for correlations >= {args.corr_threshold}...")
    rows = []
    
    for j, sel_gene in enumerate(selected_genes):
        if sel_gene not in all_genes:
            continue
            
        # Get correlations for this specific selected gene against ALL genes
        correlations = corr_matrix[:, j]
        
        # Find genes that pass the threshold
        high_corr_idx = np.where(correlations >= args.corr_threshold)[0]
        
        for i in high_corr_idx:
            cand_gene = all_genes[i]
            
            # Don't list the gene as a swap for itself
            if cand_gene != sel_gene:
                rows.append({
                    "MILP_Selected_Gene": sel_gene,
                    "Substitute_Gene": cand_gene,
                    "Pearson_Correlation": round(float(correlations[i]), 4)
                })
                
    if not rows:
        print(f"No redundant genes found above correlation {args.corr_threshold}.")
        return

    # Sort and save
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["MILP_Selected_Gene", "Pearson_Correlation"], ascending=[True, False])
    df.to_csv(args.out_csv, index=False)
    
    print(f"\nDone! Found {len(df)} possible gene swaps.")
    print(f"Saved mapping to: {args.out_csv}")
    
    # Print a quick preview
    print("\nTop 10 Strongest Swaps:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()