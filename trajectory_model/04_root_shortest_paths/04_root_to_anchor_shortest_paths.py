#!/usr/bin/env python3
"""
Step 04 — Root-to-anchor shortest-path union.

For each patient-specific TCR group with >=2 anchors:
  1. Choose the earliest observed anchor (month 1, then 3, then 6).
  2. Break same-month ties by the new Step 02 node_index.
  3. Run one graph Dijkstra search from that root.
  4. Trace paths to every other anchor and retain their edge union.

All shortest paths use the COMPLETE Step 02 T-cell kNN graph.
Intermediate cells need not share the patient's identity or TCR.
No biological directionality or ancestry is inferred.

Outputs:
  04_method1_nodes.csv.gz
  04_method1_edges.csv.gz
  04_method1_group_summary.csv
  04_method1_qc.json
"""

print("[START] Step 04 Python started.", flush=True)

import csv
import gzip
import json
import sys
import time
import traceback
from pathlib import Path

print("[IMPORT] NumPy / pandas / SciPy...", flush=True)
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
print("[IMPORT] Complete.", flush=True)


import argparse

parser = argparse.ArgumentParser(description="Trajectory stage 04. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
args = parser.parse_args()
PROJECT = args.project_dir

GRAPH_DIR = PROJECT / "02_knn_graph" / "outputs"
ANCHOR_DIR = PROJECT / "03_tcr_anchors" / "outputs"
OUT = PROJECT / "04_root_shortest_paths" / "outputs"

GRAPH_PATH = GRAPH_DIR / "02_T_knn_graph.npz"
CELL_PATH = GRAPH_DIR / "02_T_cell_index.csv.gz"
ANCHOR_PATH = ANCHOR_DIR / "03_T_anchor_cells.csv.gz"
GROUP_PATH = ANCHOR_DIR / "03_patient_tcr_group_summary.csv"

NO_PREDECESSOR = -9999


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for path in [GRAPH_PATH, CELL_PATH, ANCHOR_PATH, GROUP_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("[01] Reading validated graph...", flush=True)
    graph = sp.load_npz(GRAPH_PATH).tocsr()
    graph.sort_indices()

    n = graph.shape[0]
    if graph.shape != (155293, 155293):
        raise ValueError(f"Unexpected graph shape: {graph.shape}")
    if (graph != graph.T).nnz:
        raise ValueError("Graph is not symmetric.")
    if not np.isfinite(graph.data).all() or np.any(graph.data <= 0):
        raise ValueError("Graph must have finite, strictly positive edge weights.")

    print(
        f"[01] {n:,} cells; {graph.nnz // 2:,} undirected edges.",
        flush=True,
    )

    print("[02] Reading Step 02 cell identities...", flush=True)
    cells = pd.read_csv(
        CELL_PATH,
        usecols=[
            "node_index", "cell_id", "patient", "time_month",
            "manual_cell_type", "umap_1", "umap_2",
        ],
        dtype={"cell_id": "string", "patient": "string"},
    ).sort_values("node_index")

    if len(cells) != n:
        raise ValueError("Cell index and graph have different sizes.")
    if not np.array_equal(cells["node_index"].to_numpy(), np.arange(n)):
        raise ValueError("Step 02 node indices are not consecutive.")
    if cells["cell_id"].duplicated().any():
        raise ValueError("Duplicate Step 02 cell IDs.")

    cell_id = cells["cell_id"].astype(str).to_numpy()
    patient = cells["patient"].astype(str).to_numpy()
    time_month = cells["time_month"].to_numpy()
    cell_type = cells["manual_cell_type"].astype(str).to_numpy()
    umap = cells[["umap_1", "umap_2"]].to_numpy()

    print("[03] Reading Step 03 anchors and group summary...", flush=True)
    anchors = pd.read_csv(
        ANCHOR_PATH,
        usecols=["node_index", "cell_id", "patient", "time_month", "group_key"],
        dtype={"cell_id": "string", "patient": "string", "group_key": "string"},
    )

    all_groups = pd.read_csv(
        GROUP_PATH,
        dtype={"patient": "string", "clonotype": "string", "group_key": "string"},
    )

    if anchors["node_index"].duplicated().any():
        raise ValueError("An anchor node occurs more than once.")

    anchor_idx = anchors["node_index"].to_numpy(dtype=np.int64)
    if np.any(anchor_idx < 0) or np.any(anchor_idx >= n):
        raise ValueError("An anchor index is outside the graph.")

    if not np.array_equal(
        anchors["cell_id"].astype(str).to_numpy(),
        cell_id[anchor_idx],
    ):
        raise ValueError("Anchor cell IDs do not match their graph indices.")

    if not np.array_equal(
        anchors["patient"].astype(str).to_numpy(),
        patient[anchor_idx],
    ):
        raise ValueError("Anchor patient IDs do not match the cell index.")

    if anchors["group_key"].isna().any():
        raise ValueError("Missing patient-specific group keys.")

    # The supported dataset has observation months 1, 3, and 6.
    anchor_month = pd.to_numeric(anchors["time_month"], errors="raise")
    if anchor_month.isna().any() or not anchor_month.isin([1, 3, 6]).all():
        raise ValueError("Anchor time_month must be 1, 3, or 6.")
    anchors["time_month"] = anchor_month.astype(int)

    group_counts = anchors.groupby("group_key").size()
    expected_counts = all_groups.set_index("group_key")["n_anchor_cells"]

    if not group_counts.sort_index().equals(
        expected_counts.sort_index().astype(np.int64)
    ):
        raise ValueError("Step 03 anchor counts and group summary disagree.")

    multi = anchors[
        anchors["group_key"].map(group_counts).ge(2)
    ].copy()

    multi = multi.sort_values(
        ["group_key", "time_month", "node_index"]
    )

    grouped = multi.groupby("group_key", sort=True)
    n_groups = grouped.ngroups

    print(
        f"[03] {len(anchors):,} anchors; "
        f"{len(all_groups):,} total groups; "
        f"{n_groups:,} multi-anchor groups.",
        flush=True,
    )

    print("[04] Reconstructing root-to-anchor path unions...", flush=True)

    node_final = OUT / "04_method1_nodes.csv.gz"
    edge_final = OUT / "04_method1_edges.csv.gz"
    summary_final = OUT / "04_method1_group_summary.csv"

    # Write temporary outputs; rename only after the entire run succeeds.
    node_tmp = Path(str(node_final) + ".partial")
    edge_tmp = Path(str(edge_final) + ".partial")
    summary_tmp = Path(str(summary_final) + ".partial")

    node_columns = [
        "group_key", "patient", "node_index", "cell_id",
        "node_role", "is_root", "time_month", "manual_cell_type",
        "umap_1", "umap_2",
    ]
    edge_columns = [
        "group_key", "patient",
        "node_u_index", "node_v_index",
        "cell_u", "cell_v", "weight",
    ]
    summary_columns = [
        "group_key", "patient", "clonotype", "n_anchor_cells",
        "n_timepoints", "root_node_index", "root_cell_id",
        "root_time_month", "n_method_nodes", "n_bridge_cells",
        "n_method_edges", "max_root_to_anchor_distance", "status",
    ]

    # Set up one row for every original group, including singletons.
    summary_by_key = {}
    for row in all_groups.itertuples(index=False):
        key = str(row.group_key)
        summary_by_key[key] = {
            "group_key": key,
            "patient": str(row.patient),
            "clonotype": str(row.clonotype),
            "n_anchor_cells": int(row.n_anchor_cells),
            "n_timepoints": int(row.n_timepoints),
            "root_node_index": "",
            "root_cell_id": "",
            "root_time_month": "",
            "n_method_nodes": "",
            "n_bridge_cells": "",
            "n_method_edges": "",
            "max_root_to_anchor_distance": "",
            "status": "singleton_not_reconstructed",
        }

    start = time.monotonic()
    total_nodes_written = 0
    total_edges_written = 0

    with (
        gzip.open(node_tmp, "wt", newline="") as node_file,
        gzip.open(edge_tmp, "wt", newline="") as edge_file,
    ):
        node_writer = csv.writer(node_file)
        edge_writer = csv.writer(edge_file)

        node_writer.writerow(node_columns)
        edge_writer.writerow(edge_columns)

        for group_number, (key, group) in enumerate(grouped, start=1):
            group_indices = group["node_index"].to_numpy(dtype=np.int64)
            anchor_set = set(map(int, group_indices))

            # Sorted by month, then node_index: deterministic root.
            root_row = group.iloc[0]
            root = int(root_row["node_index"])
            root_month = int(root_row["time_month"])
            group_patient = str(root_row["patient"])

            if group["patient"].astype(str).nunique() != 1:
                raise ValueError(f"Mixed patient IDs in group {key}")

            # One Dijkstra search per group, NOT one per anchor.
            distances, predecessors = dijkstra(
                graph,
                directed=False,
                indices=root,
                return_predecessors=True,
            )

            if not np.isfinite(distances[group_indices]).all():
                raise RuntimeError(f"Unreachable anchor in group {key}")

            # Trace each anchor back to the root.
            # Stop at an already-visited node; its route to root
            # has already been added to this group's union.
            visited = {root}
            selected_edges = set()

            for target in group_indices:
                current = int(target)
                path_steps = 0

                while current not in visited:
                    previous = int(predecessors[current])

                    if previous == NO_PREDECESSOR or previous < 0:
                        raise RuntimeError(
                            f"Missing predecessor for {current} in {key}"
                        )

                    u, v = sorted((current, previous))
                    selected_edges.add((u, v))
                    visited.add(current)

                    current = previous
                    path_steps += 1

                    if path_steps > n:
                        raise RuntimeError(f"Predecessor loop in {key}")

            # A root-to-terminal path union should contain all anchors.
            if not anchor_set.issubset(visited):
                raise RuntimeError(f"Missing anchors in path union: {key}")

            # Verify every saved edge against the original graph.
            for u, v in sorted(selected_edges):
                weight = float(graph[u, v])
                if weight <= 0 or not np.isfinite(weight):
                    raise RuntimeError(
                        f"Non-kNN or invalid edge ({u}, {v}) in {key}"
                    )

                edge_writer.writerow([
                    key, group_patient, u, v,
                    cell_id[u], cell_id[v], weight,
                ])

            for node in sorted(visited):
                node_writer.writerow([
                    key,
                    group_patient,
                    node,
                    cell_id[node],
                    "anchor" if node in anchor_set else "bridge",
                    node == root,
                    time_month[node],
                    cell_type[node],
                    umap[node, 0],
                    umap[node, 1],
                ])

            n_bridges = len(visited - anchor_set)
            total_nodes_written += len(visited)
            total_edges_written += len(selected_edges)

            summary_by_key[key].update({
                "root_node_index": root,
                "root_cell_id": cell_id[root],
                "root_time_month": root_month,
                "n_method_nodes": len(visited),
                "n_bridge_cells": n_bridges,
                "n_method_edges": len(selected_edges),
                "max_root_to_anchor_distance": float(
                    np.max(distances[group_indices])
                ),
                "status": "complete",
            })

            if group_number == 1 or group_number % 25 == 0:
                elapsed = (time.monotonic() - start) / 60
                print(
                    f"[04] {group_number:,}/{n_groups:,} groups; "
                    f"{elapsed:.1f} min elapsed; "
                    f"{total_edges_written:,} saved group-edge rows.",
                    flush=True,
                )

    print("[05] Writing summary...", flush=True)
    with open(summary_tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for key in sorted(summary_by_key):
            writer.writerow(summary_by_key[key])

    # Publish outputs only once all groups have completed.
    node_tmp.replace(node_final)
    edge_tmp.replace(edge_final)
    summary_tmp.replace(summary_final)

    qc = {
        "method": "root_to_all_anchors_shortest_path_union",
        "graph_cells": int(n),
        "graph_undirected_edges": int(graph.nnz // 2),
        "all_patient_tcr_groups": int(len(all_groups)),
        "multi_anchor_groups_completed": int(n_groups),
        "singleton_groups_not_reconstructed": int(
            (all_groups["n_anchor_cells"] == 1).sum()
        ),
        "total_group_node_rows": int(total_nodes_written),
        "total_group_edge_rows": int(total_edges_written),
        "root_rule": "earliest available month 1, then 3, then 6",
        "root_tie_breaker": "smallest new Step 02 node_index",
        "graph_scope": "full T-cell graph; bridges may come from other patients",
        "edge_provenance": "all saved edges verified against Step 02 graph",
        "elapsed_minutes": round(
            (time.monotonic() - start) / 60, 2
        ),
    }

    with open(OUT / "04_method1_qc.json", "w") as handle:
        json.dump(qc, handle, indent=2)

    print("\n===== STEP 04 QC =====", flush=True)
    print(json.dumps(qc, indent=2), flush=True)
    print("[SUCCESS] Step 04 completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Step 04 failed.", flush=True)
        traceback.print_exc()
        sys.exit(1)
