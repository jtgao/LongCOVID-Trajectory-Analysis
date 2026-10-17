#!/usr/bin/env python3

"""
STEP 07 — Cross-method comparison and bridge-state summarization.

Methods:
    SP       = Step 04 rooted shortest-path tree
    MST      = Step 05 minimum spanning tree
    Steiner  = Step 06 Steiner approximation

For every reconstructed patient-specific clonotype group:
    - compare node counts
    - compare bridge-node counts
    - compare edge counts
    - compare total edge weights
    - calculate pairwise node Jaccard overlap
    - calculate pairwise bridge-node Jaccard overlap
    - calculate pairwise edge Jaccard overlap
    - calculate three-method intersections
    - summarize manual cell-state composition of bridge nodes

No new graph reconstruction is performed.
"""

print("[START] Step 07 Python started.", flush=True)

import json
import sys
import traceback
from pathlib import Path

print("[IMPORT] NumPy...", flush=True)
import numpy as np
print("[IMPORT] NumPy OK", flush=True)

print("[IMPORT] pandas...", flush=True)
import pandas as pd
print("[IMPORT] pandas OK", flush=True)


import argparse

parser = argparse.ArgumentParser(description="Trajectory stage 07. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
args = parser.parse_args()
PROJECT = args.project_dir

ANCHOR_DIR = PROJECT / "03_tcr_anchors" / "outputs"
SP_DIR = PROJECT / "04_root_shortest_paths" / "outputs"
MST_DIR = PROJECT / "05_mst_reconstruction" / "outputs"
STEINER_DIR = PROJECT / "06_steiner_reconstruction" / "outputs"

OUT = PROJECT / "07_cross_method_comparison" / "outputs"

GROUP_PATH = (
    ANCHOR_DIR
    / "03_patient_tcr_group_summary.csv"
)

METHODS = {
    "SP": {
        "nodes": SP_DIR / "04_method1_nodes.csv.gz",
        "edges": SP_DIR / "04_method1_edges.csv.gz",
    },
    "MST": {
        "nodes": MST_DIR / "05_method2_mst_nodes.csv.gz",
        "edges": MST_DIR / "05_method2_mst_edges.csv.gz",
    },
    "Steiner": {
        "nodes": STEINER_DIR / "06_method3_steiner_nodes.csv.gz",
        "edges": STEINER_DIR / "06_method3_steiner_edges.csv.gz",
    },
}


def jaccard(a, b):
    """
    Jaccard similarity for two sets.
    If both are empty, define similarity as 1.
    """
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def edge_set_from_group(df):
    """
    Return unordered graph edges as (min_node, max_node).
    """
    return {
        tuple(sorted((int(u), int(v))))
        for u, v in zip(
            df["node_u_index"],
            df["node_v_index"],
        )
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[01] Checking input files...", flush=True)

    required = [GROUP_PATH]

    for method in METHODS.values():
        required.append(method["nodes"])
        required.append(method["edges"])

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("[01] All inputs found.", flush=True)

    # ------------------------------------------------------
    # MASTER GROUP METADATA
    # ------------------------------------------------------

    print("[02] Loading clonotype group summary...", flush=True)

    groups = pd.read_csv(
        GROUP_PATH,
        dtype={
            "patient": "string",
            "clonotype": "string",
            "group_key": "string",
        },
    )

    multi_groups = groups[
        groups["n_anchor_cells"] >= 2
    ].copy()

    eligible_keys = set(
        multi_groups["group_key"].astype(str)
    )

    print(
        f"[02] {len(groups):,} total groups; "
        f"{len(multi_groups):,} multi-anchor groups.",
        flush=True,
    )

    # ------------------------------------------------------
    # LOAD METHOD OUTPUTS
    # ------------------------------------------------------

    node_tables = {}
    edge_tables = {}

    for method, paths in METHODS.items():

        print(
            f"[03] Loading {method} nodes...",
            flush=True,
        )

        nodes = pd.read_csv(
            paths["nodes"],
            dtype={
                "group_key": "string",
                "patient": "string",
                "cell_id": "string",
                "node_role": "string",
                "manual_cell_type": "string",
            },
        )

        print(
            f"[03] Loading {method} edges...",
            flush=True,
        )

        edges = pd.read_csv(
            paths["edges"],
            dtype={
                "group_key": "string",
                "patient": "string",
                "cell_u": "string",
                "cell_v": "string",
            },
        )

        method_keys_nodes = set(
            nodes["group_key"].astype(str)
        )

        method_keys_edges = set(
            edges["group_key"].astype(str)
        )

        if method_keys_nodes != eligible_keys:
            raise ValueError(
                f"{method} node groups do not exactly match "
                "the 3,629 multi-anchor groups."
            )

        if method_keys_edges != eligible_keys:
            raise ValueError(
                f"{method} edge groups do not exactly match "
                "the 3,629 multi-anchor groups."
            )

        if nodes.duplicated(
            subset=["group_key", "node_index"]
        ).any():
            raise ValueError(
                f"Duplicate group-node row in {method}."
            )

        node_tables[method] = nodes
        edge_tables[method] = edges

        print(
            f"[03] {method}: "
            f"{len(nodes):,} node rows; "
            f"{len(edges):,} edge rows.",
            flush=True,
        )

    # ------------------------------------------------------
    # PRE-GROUP TABLES
    # ------------------------------------------------------

    print("[04] Indexing outputs by clonotype group...", flush=True)

    node_grouped = {
        method: {
            str(key): group.copy()
            for key, group in table.groupby(
                "group_key",
                sort=False,
            )
        }
        for method, table in node_tables.items()
    }

    edge_grouped = {
        method: {
            str(key): group.copy()
            for key, group in table.groupby(
                "group_key",
                sort=False,
            )
        }
        for method, table in edge_tables.items()
    }

    # ------------------------------------------------------
    # GROUP-LEVEL COMPARISON
    # ------------------------------------------------------

    print("[05] Computing cross-method overlaps...", flush=True)

    rows = []

    ordered_groups = (
        multi_groups
        .sort_values(
            ["patient", "clonotype"]
        )
    )

    for number, meta in enumerate(
        ordered_groups.itertuples(index=False),
        start=1,
    ):

        key = str(meta.group_key)

        method_node_sets = {}
        method_bridge_sets = {}
        method_edge_sets = {}
        method_weights = {}
        method_counts = {}

        # Anchor sets should be the same under all methods.
        anchor_sets = {}

        for method in METHODS:

            ndf = node_grouped[method][key]
            edf = edge_grouped[method][key]

            nodes = set(
                ndf["node_index"]
                .astype(int)
                .tolist()
            )

            anchors = set(
                ndf.loc[
                    ndf["node_role"].eq("anchor"),
                    "node_index",
                ]
                .astype(int)
                .tolist()
            )

            bridges = set(
                ndf.loc[
                    ndf["node_role"].eq("bridge"),
                    "node_index",
                ]
                .astype(int)
                .tolist()
            )

            if nodes != anchors | bridges:
                raise RuntimeError(
                    f"Node-role partition failed for "
                    f"{method}, {key}"
                )

            edges = edge_set_from_group(
                edf
            )

            if len(edges) != len(edf):
                raise RuntimeError(
                    f"Duplicate edges in {method}, {key}"
                )

            # All three are trees.
            if len(edges) != len(nodes) - 1:
                raise RuntimeError(
                    f"{method} is not a tree for {key}: "
                    f"{len(nodes)} nodes, "
                    f"{len(edges)} edges."
                )

            total_weight = float(
                edf["weight"].sum()
            )

            method_node_sets[method] = nodes
            method_bridge_sets[method] = bridges
            method_edge_sets[method] = edges
            method_weights[method] = total_weight

            method_counts[method] = {
                "nodes": len(nodes),
                "bridges": len(bridges),
                "edges": len(edges),
            }

            anchor_sets[method] = anchors

        if not (
            anchor_sets["SP"]
            == anchor_sets["MST"]
            == anchor_sets["Steiner"]
        ):
            raise RuntimeError(
                f"Anchor sets differ across methods for {key}"
            )

        anchors = anchor_sets["SP"]

        if len(anchors) != int(meta.n_anchor_cells):
            raise RuntimeError(
                f"Anchor count mismatch for {key}"
            )

        # Pairwise overlaps.
        sp_mst_node_j = jaccard(
            method_node_sets["SP"],
            method_node_sets["MST"],
        )

        sp_steiner_node_j = jaccard(
            method_node_sets["SP"],
            method_node_sets["Steiner"],
        )

        mst_steiner_node_j = jaccard(
            method_node_sets["MST"],
            method_node_sets["Steiner"],
        )

        sp_mst_bridge_j = jaccard(
            method_bridge_sets["SP"],
            method_bridge_sets["MST"],
        )

        sp_steiner_bridge_j = jaccard(
            method_bridge_sets["SP"],
            method_bridge_sets["Steiner"],
        )

        mst_steiner_bridge_j = jaccard(
            method_bridge_sets["MST"],
            method_bridge_sets["Steiner"],
        )

        sp_mst_edge_j = jaccard(
            method_edge_sets["SP"],
            method_edge_sets["MST"],
        )

        sp_steiner_edge_j = jaccard(
            method_edge_sets["SP"],
            method_edge_sets["Steiner"],
        )

        mst_steiner_edge_j = jaccard(
            method_edge_sets["MST"],
            method_edge_sets["Steiner"],
        )

        # Three-method intersections.
        common_nodes = (
            method_node_sets["SP"]
            & method_node_sets["MST"]
            & method_node_sets["Steiner"]
        )

        common_bridges = (
            method_bridge_sets["SP"]
            & method_bridge_sets["MST"]
            & method_bridge_sets["Steiner"]
        )

        common_edges = (
            method_edge_sets["SP"]
            & method_edge_sets["MST"]
            & method_edge_sets["Steiner"]
        )

        rows.append({
            "group_key": key,
            "patient": str(meta.patient),
            "clonotype": str(meta.clonotype),
            "n_anchor_cells": int(meta.n_anchor_cells),
            "n_timepoints": int(meta.n_timepoints),

            "sp_nodes":
                method_counts["SP"]["nodes"],
            "sp_bridges":
                method_counts["SP"]["bridges"],
            "sp_edges":
                method_counts["SP"]["edges"],
            "sp_total_weight":
                method_weights["SP"],

            "mst_nodes":
                method_counts["MST"]["nodes"],
            "mst_bridges":
                method_counts["MST"]["bridges"],
            "mst_edges":
                method_counts["MST"]["edges"],
            "mst_total_weight":
                method_weights["MST"],

            "steiner_nodes":
                method_counts["Steiner"]["nodes"],
            "steiner_bridges":
                method_counts["Steiner"]["bridges"],
            "steiner_edges":
                method_counts["Steiner"]["edges"],
            "steiner_total_weight":
                method_weights["Steiner"],

            "sp_mst_node_jaccard":
                sp_mst_node_j,
            "sp_steiner_node_jaccard":
                sp_steiner_node_j,
            "mst_steiner_node_jaccard":
                mst_steiner_node_j,

            "sp_mst_bridge_jaccard":
                sp_mst_bridge_j,
            "sp_steiner_bridge_jaccard":
                sp_steiner_bridge_j,
            "mst_steiner_bridge_jaccard":
                mst_steiner_bridge_j,

            "sp_mst_edge_jaccard":
                sp_mst_edge_j,
            "sp_steiner_edge_jaccard":
                sp_steiner_edge_j,
            "mst_steiner_edge_jaccard":
                mst_steiner_edge_j,

            "three_method_common_nodes":
                len(common_nodes),
            "three_method_common_bridges":
                len(common_bridges),
            "three_method_common_edges":
                len(common_edges),
        })

        if number == 1 or number % 250 == 0:
            print(
                f"[05] Compared "
                f"{number:,}/{len(ordered_groups):,} groups.",
                flush=True,
            )

    comparison = pd.DataFrame(
        rows
    )

    comparison.to_csv(
        OUT
        / "07_cross_method_group_comparison.csv",
        index=False,
    )

    print(
        "[05] Group-level comparison saved.",
        flush=True,
    )

    # ------------------------------------------------------
    # BRIDGE STATE COUNTS — GROUP x METHOD x STATE
    # ------------------------------------------------------

    print(
        "[06] Summarizing bridge-cell states...",
        flush=True,
    )

    bridge_state_tables = []

    for method, nodes in node_tables.items():

        bridges = nodes[
            nodes["node_role"].eq("bridge")
        ].copy()

        state_counts = (
            bridges
            .groupby(
                [
                    "group_key",
                    "patient",
                    "manual_cell_type",
                ],
                observed=True,
            )
            .size()
            .rename("n_bridge_cells")
            .reset_index()
        )

        state_counts.insert(
            2,
            "method",
            method,
        )

        group_totals = (
            state_counts
            .groupby(
                [
                    "group_key",
                    "method",
                ]
            )["n_bridge_cells"]
            .transform("sum")
        )

        state_counts[
            "bridge_state_fraction"
        ] = (
            state_counts["n_bridge_cells"]
            / group_totals
        )

        bridge_state_tables.append(
            state_counts
        )

    bridge_state_by_group = pd.concat(
        bridge_state_tables,
        ignore_index=True,
    )

    bridge_state_by_group.to_csv(
        OUT
        / "07_bridge_state_counts_by_group_method.csv.gz",
        index=False,
        compression="gzip",
    )

    # ------------------------------------------------------
    # OVERALL BRIDGE STATE SUMMARY
    # ------------------------------------------------------

    overall_state = (
        bridge_state_by_group
        .groupby(
            [
                "method",
                "manual_cell_type",
            ],
            observed=True,
        )
        .agg(
            total_bridge_occurrences=(
                "n_bridge_cells",
                "sum",
            ),
            n_clonotype_groups=(
                "group_key",
                "nunique",
            ),
        )
        .reset_index()
    )

    method_totals = (
        overall_state
        .groupby("method")[
            "total_bridge_occurrences"
        ]
        .transform("sum")
    )

    overall_state[
        "fraction_of_method_bridge_occurrences"
    ] = (
        overall_state[
            "total_bridge_occurrences"
        ]
        / method_totals
    )

    overall_state = overall_state.sort_values(
        [
            "method",
            "total_bridge_occurrences",
        ],
        ascending=[
            True,
            False,
        ],
    )

    overall_state.to_csv(
        OUT
        / "07_bridge_state_overall_summary.csv",
        index=False,
    )

    # ------------------------------------------------------
    # GLOBAL METHOD SUMMARY
    # ------------------------------------------------------

    print(
        "[07] Writing global comparison summary...",
        flush=True,
    )

    global_rows = []

    for method in METHODS:

        prefix = {
            "SP": "sp",
            "MST": "mst",
            "Steiner": "steiner",
        }[method]

        global_rows.append({
            "method": method,
            "n_groups":
                len(comparison),
            "total_node_rows":
                int(
                    comparison[
                        f"{prefix}_nodes"
                    ].sum()
                ),
            "total_bridge_rows":
                int(
                    comparison[
                        f"{prefix}_bridges"
                    ].sum()
                ),
            "total_edge_rows":
                int(
                    comparison[
                        f"{prefix}_edges"
                    ].sum()
                ),
            "total_edge_weight":
                float(
                    comparison[
                        f"{prefix}_total_weight"
                    ].sum()
                ),
            "median_nodes_per_group":
                float(
                    comparison[
                        f"{prefix}_nodes"
                    ].median()
                ),
            "median_bridges_per_group":
                float(
                    comparison[
                        f"{prefix}_bridges"
                    ].median()
                ),
            "median_total_weight_per_group":
                float(
                    comparison[
                        f"{prefix}_total_weight"
                    ].median()
                ),
        })

    global_summary = pd.DataFrame(
        global_rows
    )

    global_summary.to_csv(
        OUT
        / "07_method_global_summary.csv",
        index=False,
    )

    # ------------------------------------------------------
    # PAIRWISE OVERLAP SUMMARY
    # ------------------------------------------------------

    pairwise_summary = pd.DataFrame([
        {
            "method_pair": "SP_vs_MST",
            "mean_node_jaccard":
                comparison[
                    "sp_mst_node_jaccard"
                ].mean(),
            "median_node_jaccard":
                comparison[
                    "sp_mst_node_jaccard"
                ].median(),
            "mean_bridge_jaccard":
                comparison[
                    "sp_mst_bridge_jaccard"
                ].mean(),
            "median_bridge_jaccard":
                comparison[
                    "sp_mst_bridge_jaccard"
                ].median(),
            "mean_edge_jaccard":
                comparison[
                    "sp_mst_edge_jaccard"
                ].mean(),
            "median_edge_jaccard":
                comparison[
                    "sp_mst_edge_jaccard"
                ].median(),
        },
        {
            "method_pair": "SP_vs_Steiner",
            "mean_node_jaccard":
                comparison[
                    "sp_steiner_node_jaccard"
                ].mean(),
            "median_node_jaccard":
                comparison[
                    "sp_steiner_node_jaccard"
                ].median(),
            "mean_bridge_jaccard":
                comparison[
                    "sp_steiner_bridge_jaccard"
                ].mean(),
            "median_bridge_jaccard":
                comparison[
                    "sp_steiner_bridge_jaccard"
                ].median(),
            "mean_edge_jaccard":
                comparison[
                    "sp_steiner_edge_jaccard"
                ].mean(),
            "median_edge_jaccard":
                comparison[
                    "sp_steiner_edge_jaccard"
                ].median(),
        },
        {
            "method_pair": "MST_vs_Steiner",
            "mean_node_jaccard":
                comparison[
                    "mst_steiner_node_jaccard"
                ].mean(),
            "median_node_jaccard":
                comparison[
                    "mst_steiner_node_jaccard"
                ].median(),
            "mean_bridge_jaccard":
                comparison[
                    "mst_steiner_bridge_jaccard"
                ].mean(),
            "median_bridge_jaccard":
                comparison[
                    "mst_steiner_bridge_jaccard"
                ].median(),
            "mean_edge_jaccard":
                comparison[
                    "mst_steiner_edge_jaccard"
                ].mean(),
            "median_edge_jaccard":
                comparison[
                    "mst_steiner_edge_jaccard"
                ].median(),
        },
    ])

    pairwise_summary.to_csv(
        OUT
        / "07_pairwise_overlap_summary.csv",
        index=False,
    )

    # ------------------------------------------------------
    # QC
    # ------------------------------------------------------

    qc = {
        "methods": [
            "SP",
            "MST",
            "Steiner",
        ],
        "multi_anchor_groups_compared":
            int(len(comparison)),
        "expected_multi_anchor_groups":
            int(len(multi_groups)),
        "all_methods_have_same_group_set":
            True,
        "anchors_verified_identical_across_methods":
            True,
        "tree_identity_checked_per_group":
            True,
        "outputs": {
            "group_comparison":
                "07_cross_method_group_comparison.csv",
            "bridge_state_by_group":
                "07_bridge_state_counts_by_group_method.csv.gz",
            "bridge_state_overall":
                "07_bridge_state_overall_summary.csv",
            "global_method_summary":
                "07_method_global_summary.csv",
            "pairwise_overlap_summary":
                "07_pairwise_overlap_summary.csv",
        },
        "stability_metrics": [
            "node_jaccard",
            "bridge_node_jaccard",
            "edge_jaccard",
            "three_method_intersection_counts",
        ],
        "single_composite_stability_score_created":
            False,
    }

    with open(
        OUT
        / "07_cross_method_qc.json",
        "w",
    ) as handle:
        json.dump(
            qc,
            handle,
            indent=2,
        )

    print(
        "\n===== STEP 07 QC =====",
        flush=True,
    )

    print(
        json.dumps(
            qc,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\n[SUCCESS] Step 07 completed.",
        flush=True,
    )


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print(
            "\n[ERROR] Step 07 failed.",
            flush=True,
        )

        traceback.print_exc()

        sys.exit(1)
