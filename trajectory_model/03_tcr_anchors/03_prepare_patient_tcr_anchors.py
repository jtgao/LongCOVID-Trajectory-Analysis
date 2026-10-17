#!/usr/bin/env python3
"""
STEP 03 — Match patient-specific TCR anchors to the validated graph.

Purpose
-------
Transfer original TCR metadata onto the NEW 155,293-cell graph using
cell_id, never the old Seurat node_idx.

Every cell with a valid patient and cleaned TRB CDR3 is an anchor
candidate. No minimum anchor count or state-count filter is applied.

Inputs
------
Step 02:
    02_T_cell_index.csv.gz

Original Step 21:
    T_seurat_graph_nodes_patient_specific.csv.gz

Outputs
-------
03_T_all_cells_with_tcr.csv.gz
03_T_anchor_cells.csv.gz
03_patient_tcr_group_summary.csv
03_anchor_mapping_qc.json
03_unmatched_old_tcr_cells.csv.gz

No source files are modified.
"""

print("[START] Step 03 Python process started.", flush=True)

import json
import sys
import traceback
from pathlib import Path

import pandas as pd


import argparse

parser = argparse.ArgumentParser(description="Trajectory stage 03. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
parser.add_argument("--tcr-metadata", type=Path, required=True, help="External per-cell TCR metadata CSV/CSV.GZ; not generated end-to-end here.")
args = parser.parse_args()
PROJECT = args.project_dir

GRAPH_DIR = PROJECT / "02_knn_graph" / "outputs"
OUT = PROJECT / "03_tcr_anchors" / "outputs"

CELL_INDEX = GRAPH_DIR / "02_T_cell_index.csv.gz"

# External metadata must provide OLD_COLUMNS below; keep values private.
OLD_NODES = args.tcr_metadata

OLD_COLUMNS = [
    "cell_id",
    "patient_from_sample",
    "cdr3_aa_TRB_clean",
    "has_TRB",
    "patient_clonotype_key",
]


def clean_text(series):
    """Standardize missing strings without changing valid identifiers."""

    values = series.astype("string").str.strip()

    invalid = (
        values.isna()
        | values.str.lower().isin(
            ["", "nan", "none", "null", "na", "<na>"]
        )
    )

    return values.mask(invalid, pd.NA)


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    for path in [CELL_INDEX, OLD_NODES]:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("[01] Reading the validated Step 02 cell index...", flush=True)

    cells = pd.read_csv(
        CELL_INDEX,
        dtype={
            "node_index": "int64",
            "cell_id": "string",
            "patient": "string",
            "manual_cell_type": "string",
        },
    )

    required_cells = {
        "node_index",
        "cell_id",
        "patient",
        "time_month",
        "manual_cell_type",
        "umap_1",
        "umap_2",
    }

    if not required_cells.issubset(cells.columns):
        raise KeyError(
            f"Missing Step 02 columns: "
            f"{sorted(required_cells - set(cells.columns))}"
        )

    if cells["cell_id"].isna().any():
        raise ValueError("Missing cell IDs in Step 02.")

    if cells["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell IDs in Step 02.")

    if cells["node_index"].duplicated().any():
        raise ValueError("Duplicate node indices in Step 02.")

    if sorted(cells["node_index"].tolist()) != list(range(len(cells))):
        raise ValueError(
            "Step 02 node indices are not exactly 0 through n-1."
        )

    cells["patient"] = clean_text(cells["patient"])

    print(
        f"[01] Loaded {len(cells):,} graph cells.",
        flush=True,
    )

    print("[02] Reading original Seurat TCR metadata...", flush=True)

    old = pd.read_csv(
        OLD_NODES,
        usecols=OLD_COLUMNS,
        dtype="string",
    )

    if old["cell_id"].isna().any():
        raise ValueError("Missing cell IDs in original metadata.")

    if old["cell_id"].duplicated().any():
        raise ValueError(
            "Duplicate cell IDs in original metadata. "
            "Review before performing a one-to-one merge."
        )

    old["patient_from_sample"] = clean_text(
        old["patient_from_sample"]
    )

    old["cdr3_aa_TRB_clean"] = clean_text(
        old["cdr3_aa_TRB_clean"]
    )

    old["patient_clonotype_key"] = clean_text(
        old["patient_clonotype_key"]
    )

    print(
        f"[02] Loaded {len(old):,} original Seurat cells.",
        flush=True,
    )

    print("[03] Checking exact cell-ID overlap...", flush=True)

    graph_ids = pd.Index(cells["cell_id"])
    old_ids = pd.Index(old["cell_id"])

    shared_ids = graph_ids.intersection(old_ids)
    graph_only_ids = graph_ids.difference(old_ids)
    old_only_ids = old_ids.difference(graph_ids)

    old_only = old[
        old["cell_id"].isin(old_only_ids)
    ].copy()

    old_only.to_csv(
        OUT / "03_unmatched_old_tcr_cells.csv.gz",
        index=False,
        compression="gzip",
    )

    print(
        f"[03] Shared cells: {len(shared_ids):,}; "
        f"graph-only: {len(graph_only_ids):,}; "
        f"old-only: {len(old_only_ids):,}",
        flush=True,
    )

    if len(shared_ids) == 0:
        raise ValueError(
            "No shared cells between Step 02 and original TCR metadata."
        )

    print("[04] Joining metadata by cell_id...", flush=True)

    merged = cells.merge(
        old,
        on="cell_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(merged) != len(cells):
        raise RuntimeError("The join changed the graph cell count.")

    if not merged["node_index"].is_monotonic_increasing:
        merged = merged.sort_values("node_index").reset_index(drop=True)

    if not (
        merged["node_index"].to_numpy()
        == cells["node_index"].to_numpy()
    ).all():
        raise RuntimeError(
            "Node ordering changed during the metadata join."
        )

    shared = merged["_merge"].eq("both")

    patient_mismatch = (
        shared
        & merged["patient"].notna()
        & merged["patient_from_sample"].notna()
        & merged["patient"].ne(merged["patient_from_sample"])
    )

    n_patient_mismatch = int(patient_mismatch.sum())

    if n_patient_mismatch:
        mismatch_path = OUT / "03_patient_mismatch_review.csv"

        merged.loc[
            patient_mismatch,
            [
                "node_index",
                "cell_id",
                "patient",
                "patient_from_sample",
            ],
        ].to_csv(mismatch_path, index=False)

        raise ValueError(
            f"{n_patient_mismatch:,} matched cells have conflicting "
            f"patient IDs. Review {mismatch_path} before continuing."
        )

    print("[04] Patient IDs agree for matched cells.", flush=True)

    print("[05] Defining patient-specific TRB anchors...", flush=True)

    valid = (
        shared
        & merged["patient"].notna()
        & merged["cdr3_aa_TRB_clean"].notna()
    )

    merged["is_anchor"] = valid

    merged["patient_clonotype_key_new"] = pd.Series(
        pd.NA,
        index=merged.index,
        dtype="string",
    )

    merged.loc[valid, "patient_clonotype_key_new"] = (
        merged.loc[valid, "patient"]
        + "__"
        + merged.loc[valid, "cdr3_aa_TRB_clean"]
    )

    # Validate against the existing key where one is available.
    old_key_present = valid & merged["patient_clonotype_key"].notna()

    old_key_mismatch = (
        old_key_present
        & merged["patient_clonotype_key"].ne(
            merged["patient_clonotype_key_new"]
        )
    )

    if old_key_mismatch.any():
        review_path = OUT / "03_clonotype_key_mismatch_review.csv"

        merged.loc[
            old_key_mismatch,
            [
                "cell_id",
                "patient",
                "cdr3_aa_TRB_clean",
                "patient_clonotype_key",
                "patient_clonotype_key_new",
            ],
        ].to_csv(review_path, index=False)

        raise ValueError(
            "New patient-specific keys disagree with original keys. "
            f"Review {review_path}."
        )

    # These columns document provenance but do not define anchors
    # using the old pipeline's filtered is_anchor flag.
    anchors = merged.loc[valid].copy()

    anchors = anchors.rename(
        columns={
            "patient_clonotype_key_new": "group_key",
        }
    )

    anchors = anchors.drop(columns=["_merge"])

    print(
        f"[05] Matched TCR anchor cells: {len(anchors):,}",
        flush=True,
    )

    print("[06] Summarizing ALL patient × clonotype groups...", flush=True)

    summary = (
        anchors.groupby(
            [
                "patient",
                "cdr3_aa_TRB_clean",
                "group_key",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_anchor_cells=("cell_id", "size"),
            n_timepoints=("time_month", "nunique"),
            n_manual_cell_types=("manual_cell_type", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "cdr3_aa_TRB_clean": "clonotype",
            }
        )
    )

    summary["eligible_for_multi_anchor_methods"] = (
        summary["n_anchor_cells"] >= 2
    )

    summary = summary.sort_values(
        ["n_anchor_cells", "patient", "clonotype"],
        ascending=[False, True, True],
    )

    if int(summary["n_anchor_cells"].sum()) != len(anchors):
        raise RuntimeError(
            "Group summary does not account for every anchor."
        )

    merged = merged.rename(
        columns={
            "patient_clonotype_key_new": "group_key",
        }
    ).drop(columns=["_merge"])

    merged.to_csv(
        OUT / "03_T_all_cells_with_tcr.csv.gz",
        index=False,
        compression="gzip",
    )

    anchors.to_csv(
        OUT / "03_T_anchor_cells.csv.gz",
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        OUT / "03_patient_tcr_group_summary.csv",
        index=False,
    )

    qc = {
        "graph_cells": int(len(cells)),
        "original_seurat_cells": int(len(old)),
        "shared_cell_ids": int(len(shared_ids)),
        "graph_only_cells": int(len(graph_only_ids)),
        "old_only_cells": int(len(old_only_ids)),
        "old_only_cells_with_clean_TRB": int(
            old_only["cdr3_aa_TRB_clean"].notna().sum()
        ),
        "patient_id_mismatches": n_patient_mismatch,
        "old_clonotype_key_mismatches": int(
            old_key_mismatch.sum()
        ),
        "matched_anchor_cells": int(len(anchors)),
        "patient_specific_tcr_groups": int(len(summary)),
        "groups_with_two_or_more_anchors": int(
            summary["eligible_for_multi_anchor_methods"].sum()
        ),
        "singleton_groups": int(
            (summary["n_anchor_cells"] == 1).sum()
        ),
        "minimum_anchors_per_group": (
            int(summary["n_anchor_cells"].min())
            if len(summary) else None
        ),
        "maximum_anchors_per_group": (
            int(summary["n_anchor_cells"].max())
            if len(summary) else None
        ),
        "old_filtered_anchor_flag_used": False,
        "old_node_indices_used": False,
        "trajectory_methods_run": False,
    }

    with open(
        OUT / "03_anchor_mapping_qc.json",
        "w",
    ) as handle:
        json.dump(qc, handle, indent=2)

    print("\n===== STEP 03 QC =====", flush=True)
    print(json.dumps(qc, indent=2), flush=True)

    print(
        "\n[SUCCESS] Step 03 completed. "
        "No trajectory methods were run.",
        flush=True,
    )


if __name__ == "__main__":

    try:
        main()

    except Exception:
        print("\n[ERROR] Step 03 failed.", flush=True)
        traceback.print_exc()
        sys.exit(1)
