#!/usr/bin/env python3

print("[START] Step 05 Python started.", flush=True)

import csv
import gzip
import json
import sys
import time
import traceback
from pathlib import Path

print("[IMPORT] NumPy...", flush=True)
import numpy as np
print("[IMPORT] NumPy OK", flush=True)

print("[IMPORT] pandas...", flush=True)
import pandas as pd
print("[IMPORT] pandas OK", flush=True)

print("[IMPORT] SciPy sparse...", flush=True)
import scipy.sparse as sp
print("[IMPORT] SciPy sparse OK", flush=True)

print("[IMPORT] SciPy graph algorithms...", flush=True)
from scipy.sparse.csgraph import (
    connected_components,
    dijkstra,
    minimum_spanning_tree,
)
print("[IMPORT] SciPy graph algorithms OK", flush=True)


import argparse

parser = argparse.ArgumentParser(description="Trajectory stage 05. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
args = parser.parse_args()
PROJECT = args.project_dir

GRAPH_DIR = PROJECT / "02_knn_graph" / "outputs"
ANCHOR_DIR = PROJECT / "03_tcr_anchors" / "outputs"
OUT = PROJECT / "05_mst_reconstruction" / "outputs"

GRAPH_PATH = GRAPH_DIR / "02_T_knn_graph.npz"
CELL_PATH = GRAPH_DIR / "02_T_cell_index.csv.gz"
ANCHOR_PATH = ANCHOR_DIR / "03_T_anchor_cells.csv.gz"
GROUP_PATH = ANCHOR_DIR / "03_patient_tcr_group_summary.csv"

NO_PREDECESSOR = -9999


def recover_path_nodes(predecessors, source, target, n_graph_nodes):

    source = int(source)
    current = int(target)

    path_nodes = {current}
    steps = 0

    while current != source:

        previous = int(predecessors[current])

        if previous == NO_PREDECESSOR or previous < 0:
            raise RuntimeError(
                f"No predecessor while tracing {source} -> {target}"
            )

        current = previous
        path_nodes.add(current)

        steps += 1

        if steps > n_graph_nodes:
            raise RuntimeError(
                f"Predecessor loop while tracing {source} -> {target}"
            )

    return path_nodes


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    print("[01] Checking input files...", flush=True)

    for path in [
        GRAPH_PATH,
        CELL_PATH,
        ANCHOR_PATH,
        GROUP_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("[01] Input files found.", flush=True)

    # ------------------------------------------------------
    # LOAD GRAPH
    # ------------------------------------------------------

    print("[02] Loading Step 02 graph NPZ...", flush=True)

    graph = sp.load_npz(GRAPH_PATH).tocsr()
    graph.sort_indices()

    print("[02] Graph NPZ loaded.", flush=True)

    n = graph.shape[0]

    if graph.shape != (155293, 155293):
        raise ValueError(
            f"Unexpected graph shape: {graph.shape}"
        )

    if (graph != graph.T).nnz:
        raise ValueError("Graph is not symmetric.")

    if not np.isfinite(graph.data).all():
        raise ValueError("Graph has non-finite weights.")

    if np.any(graph.data <= 0):
        raise ValueError("Graph has non-positive stored weights.")

    print("[03] Checking graph connectivity...", flush=True)

    n_components, _ = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    if n_components != 1:
        raise ValueError(
            f"Expected 1 graph component, found {n_components}"
        )

    print(
        f"[03] Graph validated: {n:,} cells; "
        f"{graph.nnz // 2:,} undirected edges.",
        flush=True,
    )

    # ------------------------------------------------------
    # LOAD CELL METADATA
    # ------------------------------------------------------

    print("[04] Loading Step 02 cell index...", flush=True)

    cells = pd.read_csv(
        CELL_PATH,
        usecols=[
            "node_index",
            "cell_id",
            "patient",
            "time_month",
            "manual_cell_type",
            "umap_1",
            "umap_2",
        ],
        dtype={
            "cell_id": "string",
            "patient": "string",
        },
    ).sort_values("node_index")

    print("[04] Cell index loaded.", flush=True)

    if len(cells) != n:
        raise ValueError(
            "Cell-index row count differs from graph dimension."
        )

    if not np.array_equal(
        cells["node_index"].to_numpy(),
        np.arange(n),
    ):
        raise ValueError(
            "Step 02 node indices are not 0,...,n-1."
        )

    cell_id = cells["cell_id"].astype(str).to_numpy()
    time_month = cells["time_month"].to_numpy()
    cell_type = (
        cells["manual_cell_type"].astype(str).to_numpy()
    )
    umap = cells[["umap_1", "umap_2"]].to_numpy()

    # ------------------------------------------------------
    # LOAD ANCHORS
    # ------------------------------------------------------

    print("[05] Loading Step 03 anchors...", flush=True)

    anchors = pd.read_csv(
        ANCHOR_PATH,
        usecols=[
            "node_index",
            "cell_id",
            "patient",
            "time_month",
            "group_key",
        ],
        dtype={
            "cell_id": "string",
            "patient": "string",
            "group_key": "string",
        },
    )

    print("[05] Anchor table loaded.", flush=True)

    print("[06] Loading Step 03 group summary...", flush=True)

    groups = pd.read_csv(
        GROUP_PATH,
        dtype={
            "patient": "string",
            "clonotype": "string",
            "group_key": "string",
        },
    )

    print("[06] Group summary loaded.", flush=True)

    anchor_indices = anchors["node_index"].to_numpy(
        dtype=np.int64
    )

    if anchors["node_index"].duplicated().any():
        raise ValueError(
            "An anchor graph node appears more than once."
        )

    if not np.array_equal(
        anchors["cell_id"].astype(str).to_numpy(),
        cell_id[anchor_indices],
    ):
        raise ValueError(
            "Anchor IDs do not match Step 02 node indices."
        )

    anchor_counts = anchors.groupby("group_key").size()

    expected_counts = (
        groups
        .set_index("group_key")["n_anchor_cells"]
        .astype(np.int64)
    )

    if not anchor_counts.sort_index().equals(
        expected_counts.sort_index()
    ):
        raise ValueError(
            "Step 03 anchor counts disagree with group summary."
        )

    multi = anchors[
        anchors["group_key"]
        .map(anchor_counts)
        .ge(2)
    ].copy()

    multi = multi.sort_values(
        ["group_key", "node_index"]
    )

    grouped = multi.groupby(
        "group_key",
        sort=True,
    )

    n_multi = grouped.ngroups

    total_multi_anchors = len(multi)

    expected_dijkstra_runs = (
        total_multi_anchors - n_multi
    )

    print(
        f"[06] {len(anchors):,} total anchors; "
        f"{n_multi:,} multi-anchor groups; "
        f"{expected_dijkstra_runs:,} expected Dijkstra searches.",
        flush=True,
    )

    # ------------------------------------------------------
    # OUTPUT PATHS
    # ------------------------------------------------------

    node_final = OUT / "05_method2_mst_nodes.csv.gz"
    edge_final = OUT / "05_method2_mst_edges.csv.gz"
    summary_final = OUT / "05_method2_mst_group_summary.csv"

    node_tmp = Path(str(node_final) + ".partial")
    edge_tmp = Path(str(edge_final) + ".partial")
    summary_tmp = Path(str(summary_final) + ".partial")

    node_columns = [
        "group_key",
        "patient",
        "node_index",
        "cell_id",
        "node_role",
        "time_month",
        "manual_cell_type",
        "umap_1",
        "umap_2",
    ]

    edge_columns = [
        "group_key",
        "patient",
        "node_u_index",
        "node_v_index",
        "cell_u",
        "cell_v",
        "weight",
    ]

    summary_columns = [
        "group_key",
        "patient",
        "clonotype",
        "n_anchor_cells",
        "n_timepoints",
        "n_anchor_pairs",
        "n_dijkstra_runs",
        "n_candidate_nodes",
        "n_bridge_nodes",
        "n_candidate_edges",
        "n_mst_edges",
        "mst_total_weight",
        "status",
    ]

    summary_by_key = {}

    for row in groups.itertuples(index=False):

        key = str(row.group_key)
        n_anchor = int(row.n_anchor_cells)

        summary_by_key[key] = {
            "group_key": key,
            "patient": str(row.patient),
            "clonotype": str(row.clonotype),
            "n_anchor_cells": n_anchor,
            "n_timepoints": int(row.n_timepoints),
            "n_anchor_pairs": n_anchor * (n_anchor - 1) // 2,
            "n_dijkstra_runs": "",
            "n_candidate_nodes": "",
            "n_bridge_nodes": "",
            "n_candidate_edges": "",
            "n_mst_edges": "",
            "mst_total_weight": "",
            "status": "singleton_not_reconstructed",
        }

    # ------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------

    print("[07] Starting MST reconstruction loop...", flush=True)

    started = time.monotonic()

    total_dijkstra = 0
    total_candidate_node_rows = 0
    total_mst_edge_rows = 0
    total_anchor_pairs = 0

    with (
        gzip.open(
            node_tmp,
            "wt",
            newline="",
        ) as node_handle,
        gzip.open(
            edge_tmp,
            "wt",
            newline="",
        ) as edge_handle,
    ):

        node_writer = csv.writer(node_handle)
        edge_writer = csv.writer(edge_handle)

        node_writer.writerow(node_columns)
        edge_writer.writerow(edge_columns)

        for group_number, (key, group) in enumerate(
            grouped,
            start=1,
        ):

            group = group.sort_values("node_index")

            anchor_nodes = group[
                "node_index"
            ].to_numpy(dtype=np.int64)

            anchor_set = set(map(int, anchor_nodes))

            group_patient = str(
                group.iloc[0]["patient"]
            )

            k = len(anchor_nodes)

            n_pairs = k * (k - 1) // 2
            total_anchor_pairs += n_pairs

            candidate_nodes = set(anchor_set)

            group_dijkstra_runs = 0

            # One Dijkstra search per source anchor,
            # reused for all later targets.
            for source_position in range(k - 1):

                source = int(
                    anchor_nodes[source_position]
                )

                distances, predecessors = dijkstra(
                    graph,
                    directed=False,
                    indices=source,
                    return_predecessors=True,
                )

                group_dijkstra_runs += 1
                total_dijkstra += 1

                targets = anchor_nodes[
                    source_position + 1:
                ]

                if not np.isfinite(
                    distances[targets]
                ).all():
                    raise RuntimeError(
                        f"Unreachable anchor pair in {key}"
                    )

                for target in targets:

                    path_nodes = recover_path_nodes(
                        predecessors=predecessors,
                        source=source,
                        target=int(target),
                        n_graph_nodes=n,
                    )

                    candidate_nodes.update(path_nodes)

            candidate_global = np.array(
                sorted(candidate_nodes),
                dtype=np.int64,
            )

            candidate_graph = graph[
                candidate_global
            ][:, candidate_global].tocsr()

            candidate_graph.sort_indices()

            candidate_components, _ = connected_components(
                candidate_graph,
                directed=False,
                return_labels=True,
            )

            if candidate_components != 1:
                raise RuntimeError(
                    f"Candidate graph for {key} has "
                    f"{candidate_components} components."
                )

            n_candidate_nodes = len(candidate_global)
            n_candidate_edges = int(
                candidate_graph.nnz // 2
            )

            mst_local = minimum_spanning_tree(
                candidate_graph
            ).tocoo()

            n_mst_edges = int(mst_local.nnz)

            if n_mst_edges != n_candidate_nodes - 1:
                raise RuntimeError(
                    f"MST edge count failed for {key}: "
                    f"{n_mst_edges} != "
                    f"{n_candidate_nodes - 1}"
                )

            mst_total_weight = float(
                mst_local.data.sum()
            )

            for node in candidate_global:

                node = int(node)

                node_writer.writerow([
                    key,
                    group_patient,
                    node,
                    cell_id[node],
                    "anchor" if node in anchor_set else "bridge",
                    time_month[node],
                    cell_type[node],
                    umap[node, 0],
                    umap[node, 1],
                ])

            for local_u, local_v, weight in zip(
                mst_local.row,
                mst_local.col,
                mst_local.data,
            ):

                global_u = int(
                    candidate_global[int(local_u)]
                )
                global_v = int(
                    candidate_global[int(local_v)]
                )

                u, v = sorted(
                    (global_u, global_v)
                )

                original_weight = float(
                    graph[u, v]
                )

                if (
                    original_weight <= 0
                    or not np.isfinite(original_weight)
                ):
                    raise RuntimeError(
                        f"MST introduced non-kNN edge "
                        f"({u}, {v}) in {key}"
                    )

                if not np.isclose(
                    original_weight,
                    float(weight),
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise RuntimeError(
                        f"MST edge-weight mismatch "
                        f"({u}, {v}) in {key}"
                    )

                edge_writer.writerow([
                    key,
                    group_patient,
                    u,
                    v,
                    cell_id[u],
                    cell_id[v],
                    original_weight,
                ])

            n_bridges = n_candidate_nodes - k

            total_candidate_node_rows += n_candidate_nodes
            total_mst_edge_rows += n_mst_edges

            summary_by_key[key].update({
                "n_dijkstra_runs": group_dijkstra_runs,
                "n_candidate_nodes": n_candidate_nodes,
                "n_bridge_nodes": n_bridges,
                "n_candidate_edges": n_candidate_edges,
                "n_mst_edges": n_mst_edges,
                "mst_total_weight": mst_total_weight,
                "status": "complete",
            })

            if (
                group_number == 1
                or group_number % 10 == 0
            ):
                elapsed_min = (
                    time.monotonic() - started
                ) / 60

                print(
                    f"[07] {group_number:,}/{n_multi:,} groups; "
                    f"{total_dijkstra:,}/"
                    f"{expected_dijkstra_runs:,} "
                    f"Dijkstra searches; "
                    f"{elapsed_min:.1f} min elapsed.",
                    flush=True,
                )

    print("[08] Writing summary...", flush=True)

    with open(
        summary_tmp,
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=summary_columns,
        )

        writer.writeheader()

        for key in sorted(summary_by_key):
            writer.writerow(summary_by_key[key])

    node_tmp.replace(node_final)
    edge_tmp.replace(edge_final)
    summary_tmp.replace(summary_final)

    elapsed_minutes = (
        time.monotonic() - started
    ) / 60

    qc = {
        "method":
            "all_anchor_pair_shortest_path_candidate_induced_graph_mst",
        "graph_cells":
            int(n),
        "graph_undirected_edges":
            int(graph.nnz // 2),
        "all_patient_tcr_groups":
            int(len(groups)),
        "multi_anchor_groups_completed":
            int(n_multi),
        "singleton_groups_not_reconstructed":
            int(
                (groups["n_anchor_cells"] == 1).sum()
            ),
        "multi_anchor_cells":
            int(total_multi_anchors),
        "anchor_pairs_processed":
            int(total_anchor_pairs),
        "dijkstra_searches":
            int(total_dijkstra),
        "expected_dijkstra_searches":
            int(expected_dijkstra_runs),
        "total_group_candidate_node_rows":
            int(total_candidate_node_rows),
        "total_group_mst_edge_rows":
            int(total_mst_edge_rows),
        "candidate_rule":
            "union of one predecessor-defined shortest path for every unordered anchor pair",
        "candidate_graph_rule":
            "induced subgraph of original Step 02 kNN graph",
        "mst_rule":
            "minimum-weight spanning tree of induced candidate graph",
        "bridge_rule":
            "all candidate non-anchor vertices are MST bridge nodes",
        "edge_provenance":
            "all saved MST edges verified against original Step 02 graph",
        "root_used":
            False,
        "elapsed_minutes":
            round(elapsed_minutes, 2),
    }

    with open(
        OUT / "05_method2_mst_qc.json",
        "w",
    ) as handle:
        json.dump(qc, handle, indent=2)

    print("\n===== STEP 05 QC =====", flush=True)
    print(json.dumps(qc, indent=2), flush=True)

    print("\n[SUCCESS] Step 05 completed.", flush=True)


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print("\n[ERROR] Step 05 failed.", flush=True)
        traceback.print_exc()
        sys.exit(1)
