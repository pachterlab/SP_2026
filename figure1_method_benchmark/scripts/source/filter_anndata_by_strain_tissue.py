import argparse
import re
import time
from pathlib import Path

import pandas as pd
import scanpy as sc


DEFAULT_METRICS = "~/8cubeDB/final_table_1.csv"
DEFAULT_INPUT_TEMPLATE = (
    "/home/igvf-analysis/projects/8cube/processed_data/uci_submitted/"
    "annotated/{tissue}_annotated.h5ad"
)
DEFAULT_OUTPUT_ROOT = "/home/nikki/celltype_compression_2"
DEFAULT_STRAIN = "B6J"
DEFAULT_TISSUE = "Kidney"
DEFAULT_STRAIN_COLUMN = "Genotype"
DEFAULT_TISSUE_COLUMN = "Tissue"
DEFAULT_CELLTYPE_COLUMN = "celltype"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def expand_path(path):
    return Path(path).expanduser()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Subset an AnnData file by strain/tissue, drop low-quality cells, "
            "keep PSI-filtered celltype genes, and write a filtered h5ad."
        )
    )

    parser.add_argument(
        "-o",
        "--output-h5ad",
        default=None,
        help=(
            "Output h5ad path. Default: "
            "<tissue>/<strain>_<tissue>_subset.h5ad"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help=f"CSV containing PSI metrics. Default: {DEFAULT_METRICS}",
    )
    parser.add_argument(
        "--strain",
        default=DEFAULT_STRAIN,
        help=f"Strain value to keep from obs. Default: {DEFAULT_STRAIN}",
    )
    parser.add_argument(
        "--tissue",
        default=DEFAULT_TISSUE,
        help=f"Tissue / Analysis_level value to use. Default: {DEFAULT_TISSUE}",
    )
    parser.add_argument(
        "--strain-column",
        default=DEFAULT_STRAIN_COLUMN,
        help=f"obs column containing strain. Default: {DEFAULT_STRAIN_COLUMN}",
    )
    parser.add_argument(
        "--tissue-column",
        default=DEFAULT_TISSUE_COLUMN,
        help=f"obs column containing tissue. Default: {DEFAULT_TISSUE_COLUMN}",
    )
    parser.add_argument(
        "--celltype-column",
        default=DEFAULT_CELLTYPE_COLUMN,
        help=f"obs column containing celltype labels. Default: {DEFAULT_CELLTYPE_COLUMN}",
    )
    parser.add_argument(
        "--low-quality-label",
        default="low quality",
        help="Celltype label to remove. Default: 'low quality'",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.8,
        help="Minimum Psi_mean to keep a gene. Default: 0.8",
    )
    parser.add_argument(
        "--analysis-type",
        default="Celltype",
        help="Analysis_type value to keep from the metrics table. Default: Celltype",
    )
    parser.add_argument(
        "--no-tissue-subset",
        action="store_true",
        help="Do not subset obs by tissue; tissue is still used for metrics Analysis_level.",
    )
    return parser.parse_args()


def require_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


def default_output_path(input_h5ad, strain, tissue, output_root=DEFAULT_OUTPUT_ROOT):
    strain_name = safe_name(strain)
    tissue_name = safe_name(tissue)
    return (
        expand_path(output_root)
        / tissue_name
        / f"{strain_name}_{tissue_name}_subset.h5ad"
    )


def default_input_path(tissue):
    return DEFAULT_INPUT_TEMPLATE.format(tissue=safe_name(tissue))


def main():
    args = parse_args()

    input_h5ad = expand_path(default_input_path(args.tissue))
    metrics_path = expand_path(args.metrics)
    output_h5ad = (
        expand_path(args.output_h5ad)
        if args.output_h5ad
        else default_output_path(
            input_h5ad,
            args.strain,
            args.tissue,
            args.output_root,
        )
    )

    log(f"Loading AnnData in backed mode: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad, backed="r")
    log(f"Input shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    obs_columns = [args.strain_column, args.celltype_column]
    if not args.no_tissue_subset:
        obs_columns.append(args.tissue_column)
    require_columns(adata.obs, obs_columns, "adata.obs")

    if args.strain == "All":
        log("Keeping all strains (no strain filtering).")
        keep_cells = pd.Series(True, index=adata.obs.index)
    else:    
        keep_cells = adata.obs[args.strain_column].astype(str).eq(args.strain)
        log(f"Cells matching {args.strain_column} == {args.strain}: {keep_cells.sum():,}")

    if not args.no_tissue_subset:
        tissue_mask = adata.obs[args.tissue_column].astype(str).eq(args.tissue)
        log(f"Cells matching {args.tissue_column} == {args.tissue}: {tissue_mask.sum():,}")
        keep_cells &= tissue_mask

    low_quality_mask = (
        adata.obs['celltype']
        .astype(str)
        .str.casefold()
        .eq(args.low_quality_label.casefold())
    )
    log(f"Cells labelled {args.low_quality_label!r}: {low_quality_mask.sum():,}")
    keep_cells &= ~low_quality_mask

    n_cells = int(keep_cells.sum())
    if n_cells == 0:
        raise ValueError("No cells remain after strain/tissue/low-quality filtering.")

    log(f"Loading metrics table: {metrics_path}")
    metrics = pd.read_csv(metrics_path, index_col=0)
    require_columns(
        metrics,
        ["Analysis_level", "Analysis_type", "Psi_mean", "gene_name"],
        "metrics table",
    )

    psi_filtered = metrics[
        (metrics["Analysis_level"] == args.tissue)
        & (metrics["Analysis_type"] == args.analysis_type)
        & (metrics["Psi_mean"] > args.psi_threshold)
    ]["gene_name"].dropna().unique()

    valid_genes = [
        gene
        for gene in psi_filtered
        if gene in adata.var_names
        and not gene.startswith(("Gm", "mt-", "Rps", "Rpl", "ENSMUSG"))
        and not gene.endswith("Rik")
    ]

    log(f"PSI-filtered genes before name cleanup: {len(psi_filtered):,}")
    log(f"Genes retained in AnnData after cleanup: {len(valid_genes):,}")
    if not valid_genes:
        raise ValueError("No genes remain after PSI and gene-name filtering.")

    log("Subsetting cells/genes and moving result into memory...")
    filtered = adata[keep_cells.values, valid_genes].to_memory()
    log(f"Output shape: {filtered.n_obs:,} cells x {filtered.n_vars:,} genes")

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing: {output_h5ad}")
    filtered.write_h5ad(output_h5ad)
    log("Done.")


if __name__ == "__main__":
    main()
