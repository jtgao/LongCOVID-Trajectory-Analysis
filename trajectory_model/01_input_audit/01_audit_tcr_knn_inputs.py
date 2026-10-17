#!/usr/bin/env python3
"""
STEP 01 — T-cell input audit.

Purpose:
    Identify the exact T-cell metadata and representations needed
    for the reconstructed kNN trajectory pipeline.

Reads:
    Existing annotated T-cell AnnData.
    Existing T-cell AnnData with protein.
    Checks whether the existing Seurat RDS is present.

Writes:
    01_candidate_obs_columns.csv
    01_obsm_representations.csv
    01_obsp_graphs.csv
    01_input_summary.csv
    01_cell_id_overlap.csv, when both H5AD files exist.

Does NOT construct a graph or modify source data.
"""

from pathlib import Path
import json

import pandas as pd
import scanpy as sc


import argparse

parser = argparse.ArgumentParser(description="Trajectory stage 01. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
parser.add_argument("--annotated-t", type=Path, required=True, help="Cleaned T-cell H5AD.")
parser.add_argument("--protein-h5ad", type=Path, required=True, help="T-cell H5AD with ADT counts.")
parser.add_argument("--seurat-rds", type=Path, required=True, help="Native Seurat RDS location to audit.")
args = parser.parse_args()
PROJECT = args.project_dir
OUT = PROJECT / "01_input_audit" / "outputs"
INPUTS = {"annotated_T": args.annotated_t, "T_with_protein": args.protein_h5ad}
SEURAT_RDS = args.seurat_rds

OBS_KEYWORDS = (
    "patient", "subject", "donor", "sample",
    "tcr", "cdr3", "clonotype", "clone",
    "time", "month", "visit",
    "seurat", "cluster", "manual",
)

REP_KEYWORDS = (
    "pca", "harmony", "integrat", "latent", "umap"
)


def inspect_h5ad(source, path):
    """Inspect metadata without loading the full RNA matrix."""

    print(f"\n{'=' * 70}")
    print(f"SOURCE: {source}")
    print(f"PATH: {path}")

    if not path.is_file():
        print("[MISSING] File not found.")
        return None, [], [], [], {
            "source": source,
            "path": str(path),
            "exists": False,
            "n_cells": None,
            "n_genes": None,
            "unique_cell_ids": None,
        }

    adata = sc.read_h5ad(path, backed="r")

    try:
        print(f"Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"Cell IDs unique: {adata.obs_names.is_unique}")

        obs_rows = []
        rep_rows = []
        graph_rows = []

        print("\nOBS COLUMNS:")
        for column in adata.obs.columns:
            print(f"  {column}")

            if not any(
                word in str(column).lower()
                for word in OBS_KEYWORDS
            ):
                continue

            values = adata.obs[column]

            top_values = (
                values.astype("string")
                .value_counts(dropna=False)
                .head(5)
            )

            row = {
                "source": source,
                "column": str(column),
                "dtype": str(values.dtype),
                "nonmissing_cells": int(values.notna().sum()),
                "n_unique_nonmissing": int(
                    values.nunique(dropna=True)
                ),
                "top_5_values_and_counts": json.dumps(
                    {
                        str(key): int(count)
                        for key, count in top_values.items()
                    },
                    ensure_ascii=False,
                ),
            }

            obs_rows.append(row)

            print(
                f"    [CANDIDATE] {column}: "
                f"{row['nonmissing_cells']:,} nonmissing; "
                f"{row['n_unique_nonmissing']:,} unique"
            )
            print(
                f"       Top values: "
                f"{row['top_5_values_and_counts']}"
            )

        print("\nOBSM REPRESENTATIONS:")
        for key in adata.obsm.keys():
            value = adata.obsm[key]

            row = {
                "source": source,
                "key": str(key),
                "shape": str(value.shape),
                "type": type(value).__name__,
                "candidate_representation": any(
                    word in str(key).lower()
                    for word in REP_KEYWORDS
                ),
            }

            rep_rows.append(row)

            print(f"  {key}: {value.shape}")

        print("\nOBSP GRAPHS:")
        for key in adata.obsp.keys():
            value = adata.obsp[key]

            graph_rows.append({
                "source": source,
                "key": str(key),
                "shape": str(value.shape),
                "type": type(value).__name__,
            })

            print(f"  {key}: {value.shape}")

        print("\nUNS KEYS:")
        for key in adata.uns.keys():
            print(f"  {key}")

        cell_ids = pd.Index(
            adata.obs_names.astype(str),
            name="cell_id",
        )

        summary = {
            "source": source,
            "path": str(path),
            "exists": True,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "unique_cell_ids": bool(
                adata.obs_names.is_unique
            ),
        }

        return (
            cell_ids,
            obs_rows,
            rep_rows,
            graph_rows,
            summary,
        )

    finally:
        if adata.isbacked:
            adata.file.close()


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_reps = []
    all_graphs = []
    summaries = []
    cell_ids = {}

    print("STEP 01: TCR kNN reconstruction input audit")
    print(f"Seurat RDS exists: {SEURAT_RDS.is_file()}")
    print(f"Seurat RDS path: {SEURAT_RDS}")

    for source, path in INPUTS.items():

        (
            ids,
            obs_rows,
            rep_rows,
            graph_rows,
            summary,
        ) = inspect_h5ad(source, path)

        summaries.append(summary)
        all_obs.extend(obs_rows)
        all_reps.extend(rep_rows)
        all_graphs.extend(graph_rows)

        if ids is not None:
            cell_ids[source] = ids

    pd.DataFrame(
        all_obs,
        columns=[
            "source",
            "column",
            "dtype",
            "nonmissing_cells",
            "n_unique_nonmissing",
            "top_5_values_and_counts",
        ],
    ).to_csv(
        OUT / "01_candidate_obs_columns.csv",
        index=False,
    )

    pd.DataFrame(
        all_reps,
        columns=[
            "source",
            "key",
            "shape",
            "type",
            "candidate_representation",
        ],
    ).to_csv(
        OUT / "01_obsm_representations.csv",
        index=False,
    )

    pd.DataFrame(
        all_graphs,
        columns=[
            "source",
            "key",
            "shape",
            "type",
        ],
    ).to_csv(
        OUT / "01_obsp_graphs.csv",
        index=False,
    )

    pd.DataFrame(summaries).to_csv(
        OUT / "01_input_summary.csv",
        index=False,
    )

    if len(cell_ids) == 2:

        annotated = cell_ids["annotated_T"]
        protein = cell_ids["T_with_protein"]

        overlap = annotated.intersection(protein)

        overlap_summary = pd.DataFrame([{
            "annotated_T_cells": len(annotated),
            "T_with_protein_cells": len(protein),
            "shared_cell_ids": len(overlap),
            "annotated_only": len(
                annotated.difference(protein)
            ),
            "protein_only": len(
                protein.difference(annotated)
            ),
            "annotated_ids_unique": annotated.is_unique,
            "protein_ids_unique": protein.is_unique,
        }])

        overlap_summary.to_csv(
            OUT / "01_cell_id_overlap.csv",
            index=False,
        )

        print("\nCELL-ID OVERLAP:")
        print(overlap_summary.to_string(index=False))

    if "annotated_T" not in cell_ids:
        raise FileNotFoundError(
            "The primary annotated T-cell H5AD is missing."
        )

    print(f"\n[DONE] Audit outputs saved to: {OUT}")
    print("No graph constructed. No source files modified.")


if __name__ == "__main__":
    main()
