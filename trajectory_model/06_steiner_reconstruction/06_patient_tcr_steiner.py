#!/usr/bin/env python3

"""
STEP 06 — Patient-specific graph Steiner approximation.

Input candidate graph:
    EXACTLY the same candidate vertex set used by Step 05 MST.

Approximation:
    Kou-Markowsky-Berman-style graph Steiner construction.

For each multi-anchor patient-specific clonotype:
    1. Induce H_c from the original Step 02 kNN graph using the
       Step 05 candidate vertices.
    2. Compute shortest-path distances among anchor terminals.
    3. Construct the terminal metric closure.
    4. Compute its MST.
    5. Expand metric-MST edges back to original H_c shortest paths.
    6. Compute an MST of the expanded path union.
    7. Iteratively prune nonterminal leaves.

Every final cell-cell edge is verified against the original
Step 02 kNN graph.
"""

print("[START] Step 06 Python started.", flush=True)

import csv
import gzip
import json
import sys
import time
import traceback
from collections import deque
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

parser = argparse.ArgumentParser(description="Trajectory stage 06. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
args = parser.parse_args()
PROJECT = args.project_dir

GRAPH_DIR = PROJECT / "02_knn_graph" / "outputs"
ANCHOR_DIR = PROJECT / "03_tcr_anchors" / "outputs"
MST_DIR = PROJECT / "05_mst_reconstruction" / "outputs"
OUT = PROJECT / "06_steiner_reconstruction" / "outputs"

GRAPH_PATH = GRAPH_DIR / "02_T_knn_graph.npz"
CELL_PATH = GRAPH_DIR / "02_T_cell_index.csv.gz"

ANCHOR_PATH = ANCHOR_DIR / "03_T_anchor_cells.csv.gz"
GROUP_PATH = ANCHOR_DIR / "03_patient_tcr_group_summary.csv"

MST_NODE_PATH = MST_DIR / "05_method2_mst_nodes.csv.gz"
MST_SUMMARY_PATH = MST_DIR / "05_method2_mst_group_summary.csv"

NO_PREDECESSOR = -9999


def recover_local_path(predecessor_row, source, target, n_nodes):
    """
    Recover one predecessor-defined shortest path in LOCAL
    candidate-graph indices.
    """
    source = int(source)
    current = int(target)

    path = [current]
    steps = 0

    while current != source:

        previous = int(predecessor_row[current])

        if previous == NO_PREDECESSOR or previous < 0:
            raise RuntimeError(
                f"Missing predecessor for local path "
                f"{source} -> {target}"
            )

        current = previous
        path.append(current)

        steps += 1

        if steps > n_nodes:
            raise RuntimeError(
                f"Predecessor loop for local path "
                f"{source} -> {target}"
            )

    path.reverse()
    return path


def prune_nonterminal_leaves(
    n_local,
    edge_pairs,
    terminal_local_set,
):
    """
    Iteratively remove degree-one NONTERMINAL vertices from a tree.

    Returns:
        active node set
        remaining undirected edge set
    """

    adjacency = {
        i: set()
        for i in range(n_local)
    }

    for u, v in edge_pairs:
        adjacency[u].add(v)
        adjacency[v].add(u)

    active = set()

    for u, v in edge_pairs:
        active.add(u)
        active.add(v)

    active.update(terminal_local_set)

    queue = deque(
        node
        for node in active
        if (
            node not in terminal_local_set
            and len(adjacency[node]) <= 1
        )
    )

    while queue:

        node = queue.popleft()

        if node not in active:
            continue

        if node in terminal_local_set:
            continue

        if len(adjacency[node]) > 1:
            continue

        neighbors = list(adjacency[node])

        if len(neighbors) == 1:

            neighbor = neighbors[0]

            adjacency[neighbor].remove(node)
            adjacency[node].clear()

            if (
                neighbor not in terminal_local_set
                and len(adjacency[neighbor]) <= 1
            ):
                queue.append(neighbor)

        active.remove(node)

    remaining_edges = set()

    for u in active:

        for v in adjacency[u]:

            if v in active and u < v:
                remaining_edges.add((u, v))

    return active, remaining_edges


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[01] Checking inputs...", flush=True)

    required = [
        GRAPH_PATH,
        CELL_PATH,
        ANCHOR_PATH,
        GROUP_PATH,
        MST_NODE_PATH,
        MST_SUMMARY_PATH,
    ]

    for path in required:

        if not path.is_file():
            raise FileNotFoundError(path)

    print("[01] Inputs found.", flush=True)

    # ----------------------------------------------------------
    # FULL ORIGINAL GRAPH
    # ----------------------------------------------------------

    print("[02] Loading Step 02 graph...", flush=True)

    graph = sp.load_npz(
        GRAPH_PATH
    ).tocsr()

    graph.sort_indices()

    n = graph.shape[0]

    if graph.shape != (155293, 155293):
        raise ValueError(
            f"Unexpected graph shape: {graph.shape}"
        )

    if (graph != graph.T).nnz:
        raise ValueError(
            "Step 02 graph is not symmetric."
        )

    if (
        not np.isfinite(graph.data).all()
        or np.any(graph.data <= 0)
    ):
        raise ValueError(
            "Invalid original graph weights."
        )

    print(
        f"[02] Graph loaded: {n:,} cells; "
        f"{graph.nnz // 2:,} undirected edges.",
        flush=True,
    )

    # ----------------------------------------------------------
    # CELL METADATA
    # ----------------------------------------------------------

    print("[03] Loading cell metadata...", flush=True)

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
    ).sort_values(
        "node_index"
    )

    if len(cells) != n:
        raise ValueError(
            "Cell metadata size differs from graph."
        )

    if not np.array_equal(
        cells["node_index"].to_numpy(),
        np.arange(n),
    ):
        raise ValueError(
            "Cell node indices are not 0,...,n-1."
        )

    cell_id = (
        cells["cell_id"]
        .astype(str)
        .to_numpy()
    )

    time_month = (
        cells["time_month"]
        .to_numpy()
    )

    cell_type = (
        cells["manual_cell_type"]
        .astype(str)
        .to_numpy()
    )

    umap = (
        cells[
            ["umap_1", "umap_2"]
        ]
        .to_numpy()
    )

    print("[03] Cell metadata loaded.", flush=True)

    # ----------------------------------------------------------
    # ORIGINAL ANCHORS / GROUPS
    # ----------------------------------------------------------

    print("[04] Loading anchor metadata...", flush=True)

    anchors = pd.read_csv(
        ANCHOR_PATH,
        usecols=[
            "node_index",
            "patient",
            "group_key",
        ],
        dtype={
            "patient": "string",
            "group_key": "string",
        },
    )

    groups = pd.read_csv(
        GROUP_PATH,
        dtype={
            "patient": "string",
            "clonotype": "string",
            "group_key": "string",
        },
    )

    anchor_grouped = {
        str(key):
        np.sort(
            group["node_index"]
            .to_numpy(dtype=np.int64)
        )
        for key, group
        in anchors.groupby(
            "group_key",
            sort=False,
        )
    }

    print(
        f"[04] Loaded {len(anchors):,} anchors and "
        f"{len(groups):,} groups.",
        flush=True,
    )

    # ----------------------------------------------------------
    # STEP 05 CANDIDATE VERTICES
    # ----------------------------------------------------------

    print(
        "[05] Loading Step 05 MST candidate vertices...",
        flush=True,
    )

    candidate_table = pd.read_csv(
        MST_NODE_PATH,
        usecols=[
            "group_key",
            "patient",
            "node_index",
            "node_role",
        ],
        dtype={
            "group_key": "string",
            "patient": "string",
            "node_role": "string",
        },
    )

    mst_summary = pd.read_csv(
        MST_SUMMARY_PATH,
        dtype={
            "group_key": "string",
            "patient": "string",
            "clonotype": "string",
            "status": "string",
        },
    )

    completed_keys = set(
        mst_summary.loc[
            mst_summary["status"].eq("complete"),
            "group_key",
        ].astype(str)
    )

    candidate_grouped = {
        str(key): group.copy()
        for key, group
        in candidate_table.groupby(
            "group_key",
            sort=False,
        )
    }

    if set(candidate_grouped) != completed_keys:
        raise ValueError(
            "Step 05 candidate-node groups do not match "
            "Step 05 completed groups."
        )

    n_multi = len(completed_keys)

    print(
        f"[05] Candidate vertices loaded for "
        f"{n_multi:,} reconstructed groups.",
        flush=True,
    )

    # ----------------------------------------------------------
    # OUTPUT SETUP
    # ----------------------------------------------------------

    node_final = (
        OUT
        / "06_method3_steiner_nodes.csv.gz"
    )

    edge_final = (
        OUT
        / "06_method3_steiner_edges.csv.gz"
    )

    summary_final = (
        OUT
        / "06_method3_steiner_group_summary.csv"
    )

    node_tmp = Path(
        str(node_final) + ".partial"
    )

    edge_tmp = Path(
        str(edge_final) + ".partial"
    )

    summary_tmp = Path(
        str(summary_final) + ".partial"
    )

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
        "n_candidate_nodes",
        "n_candidate_edges",
        "n_metric_edges",
        "n_metric_mst_edges",
        "n_expanded_union_nodes",
        "n_expanded_union_edges",
        "n_steiner_nodes",
        "n_steiner_bridge_nodes",
        "n_steiner_edges",
        "steiner_total_weight",
        "candidate_nodes_removed",
        "status",
    ]

    summary_by_key = {}

    group_info = (
        groups
        .set_index("group_key")
    )

    for row in groups.itertuples(
        index=False
    ):

        key = str(row.group_key)

        summary_by_key[key] = {
            "group_key": key,
            "patient": str(row.patient),
            "clonotype": str(row.clonotype),
            "n_anchor_cells":
                int(row.n_anchor_cells),
            "n_candidate_nodes": "",
            "n_candidate_edges": "",
            "n_metric_edges": "",
            "n_metric_mst_edges": "",
            "n_expanded_union_nodes": "",
            "n_expanded_union_edges": "",
            "n_steiner_nodes": "",
            "n_steiner_bridge_nodes": "",
            "n_steiner_edges": "",
            "steiner_total_weight": "",
            "candidate_nodes_removed": "",
            "status":
                "singleton_not_reconstructed",
        }

    # ----------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------

    print(
        "[06] Starting Steiner approximation loop...",
        flush=True,
    )

    started = time.monotonic()

    total_final_nodes = 0
    total_final_edges = 0
    total_bridges = 0
    total_candidate_removed = 0

    ordered_keys = sorted(
        completed_keys
    )

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

        node_writer = csv.writer(
            node_handle
        )

        edge_writer = csv.writer(
            edge_handle
        )

        node_writer.writerow(
            node_columns
        )

        edge_writer.writerow(
            edge_columns
        )

        for group_number, key in enumerate(
            ordered_keys,
            start=1,
        ):

            candidate_rows = (
                candidate_grouped[key]
                .sort_values("node_index")
            )

            candidate_global = (
                candidate_rows[
                    "node_index"
                ]
                .to_numpy(
                    dtype=np.int64
                )
            )

            if (
                len(candidate_global)
                != len(
                    np.unique(
                        candidate_global
                    )
                )
            ):
                raise ValueError(
                    f"Duplicate candidate node in {key}"
                )

            anchors_global = (
                anchor_grouped[key]
            )

            anchor_set_global = set(
                map(
                    int,
                    anchors_global,
                )
            )

            if not anchor_set_global.issubset(
                set(
                    map(
                        int,
                        candidate_global,
                    )
                )
            ):
                raise RuntimeError(
                    f"Candidate graph missing anchor in {key}"
                )

            k = len(
                anchors_global
            )

            if k < 2:
                raise RuntimeError(
                    f"Completed Step 05 group {key} "
                    f"has fewer than 2 anchors."
                )

            group_patient = str(
                group_info.loc[
                    key,
                    "patient",
                ]
            )

            # --------------------------------------------------
            # INDUCE EXACT SAME H_c AS MST
            # --------------------------------------------------

            H = graph[
                candidate_global
            ][
                :,
                candidate_global
            ].tocsr()

            H.sort_indices()

            n_candidate = len(
                candidate_global
            )

            n_candidate_edges = int(
                H.nnz // 2
            )

            components, _ = (
                connected_components(
                    H,
                    directed=False,
                    return_labels=True,
                )
            )

            if components != 1:
                raise RuntimeError(
                    f"Candidate graph H_c is disconnected "
                    f"for {key}"
                )

            global_to_local = {
                int(global_node):
                    int(local_node)
                for local_node, global_node
                in enumerate(
                    candidate_global
                )
            }

            terminal_local = np.array(
                [
                    global_to_local[
                        int(node)
                    ]
                    for node
                    in anchors_global
                ],
                dtype=np.int64,
            )

            terminal_set_local = set(
                map(
                    int,
                    terminal_local,
                )
            )

            # --------------------------------------------------
            # KMB STEP 1:
            # TERMINAL SHORTEST-PATH METRIC
            # --------------------------------------------------

            distances, predecessors = (
                dijkstra(
                    H,
                    directed=False,
                    indices=terminal_local,
                    return_predecessors=True,
                )
            )

            terminal_metric = (
                distances[
                    :,
                    terminal_local
                ]
            )

            if not np.isfinite(
                terminal_metric
            ).all():
                raise RuntimeError(
                    f"Non-finite terminal metric in {key}"
                )

            # numerical symmetry check
            if not np.allclose(
                terminal_metric,
                terminal_metric.T,
                rtol=1e-8,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"Terminal metric not symmetric in {key}"
                )

            # --------------------------------------------------
            # KMB STEP 2:
            # MST OF COMPLETE TERMINAL METRIC CLOSURE
            # --------------------------------------------------

            metric_sparse = sp.csr_matrix(
                terminal_metric
            )

            metric_mst = (
                minimum_spanning_tree(
                    metric_sparse
                ).tocoo()
            )

            if metric_mst.nnz != k - 1:
                raise RuntimeError(
                    f"Metric MST edge count failed "
                    f"for {key}"
                )

            # --------------------------------------------------
            # KMB STEP 3:
            # EXPAND METRIC MST EDGES TO H_c PATHS
            # --------------------------------------------------

            expanded_edges = set()
            expanded_nodes = set(
                terminal_set_local
            )

            for terminal_u, terminal_v in zip(
                metric_mst.row,
                metric_mst.col,
            ):

                terminal_u = int(
                    terminal_u
                )
                terminal_v = int(
                    terminal_v
                )

                source_local = int(
                    terminal_local[
                        terminal_u
                    ]
                )

                target_local = int(
                    terminal_local[
                        terminal_v
                    ]
                )

                path = recover_local_path(
                    predecessor_row=
                        predecessors[
                            terminal_u
                        ],
                    source=source_local,
                    target=target_local,
                    n_nodes=n_candidate,
                )

                expanded_nodes.update(
                    path
                )

                for left, right in zip(
                    path[:-1],
                    path[1:],
                ):

                    u, v = sorted(
                        (
                            int(left),
                            int(right),
                        )
                    )

                    if H[u, v] <= 0:
                        raise RuntimeError(
                            f"Expanded non-H_c edge "
                            f"in {key}: ({u}, {v})"
                        )

                    expanded_edges.add(
                        (u, v)
                    )

            # --------------------------------------------------
            # BUILD EXPANDED ORIGINAL-EDGE GRAPH Q_c
            # --------------------------------------------------

            q_rows = []
            q_cols = []
            q_data = []

            for u, v in sorted(
                expanded_edges
            ):

                weight = float(
                    H[u, v]
                )

                q_rows.extend(
                    [u, v]
                )

                q_cols.extend(
                    [v, u]
                )

                q_data.extend(
                    [weight, weight]
                )

            Q = sp.csr_matrix(
                (
                    q_data,
                    (
                        q_rows,
                        q_cols,
                    ),
                ),
                shape=(
                    n_candidate,
                    n_candidate,
                ),
            )

            q_components, q_labels = (
                connected_components(
                    Q,
                    directed=False,
                    return_labels=True,
                )
            )

            terminal_components = set(
                q_labels[
                    terminal_local
                ].tolist()
            )

            if len(
                terminal_components
            ) != 1:
                raise RuntimeError(
                    f"Expanded paths fail to connect terminals "
                    f"in {key}"
                )

            # --------------------------------------------------
            # KMB STEP 4:
            # MST OF EXPANDED ORIGINAL-EDGE GRAPH
            # --------------------------------------------------

            expanded_mst = (
                minimum_spanning_tree(
                    Q
                ).tocoo()
            )

            tree_edges = set()

            for u, v in zip(
                expanded_mst.row,
                expanded_mst.col,
            ):

                u, v = sorted(
                    (
                        int(u),
                        int(v),
                    )
                )

                tree_edges.add(
                    (u, v)
                )

            # Only nodes represented in the expanded tree,
            # plus terminals, are relevant to pruning.
            # --------------------------------------------------
            # KMB STEP 5:
            # PRUNE NONTERMINAL LEAVES
            # --------------------------------------------------

            final_local_nodes, final_local_edges = (
                prune_nonterminal_leaves(
                    n_local=n_candidate,
                    edge_pairs=tree_edges,
                    terminal_local_set=
                        terminal_set_local,
                )
            )

            if not terminal_set_local.issubset(
                final_local_nodes
            ):
                raise RuntimeError(
                    f"Steiner pruning removed terminal "
                    f"in {key}"
                )

            n_final_nodes = len(
                final_local_nodes
            )

            n_final_edges = len(
                final_local_edges
            )

            if n_final_edges != n_final_nodes - 1:
                raise RuntimeError(
                    f"Final Steiner structure is not a tree "
                    f"for {key}: "
                    f"{n_final_edges} edges, "
                    f"{n_final_nodes} nodes"
                )

            final_bridge_count = (
                n_final_nodes - k
            )

            final_weight = 0.0

            # --------------------------------------------------
            # SAVE FINAL NODES
            # --------------------------------------------------

            for local_node in sorted(
                final_local_nodes
            ):

                global_node = int(
                    candidate_global[
                        local_node
                    ]
                )

                role = (
                    "anchor"
                    if global_node
                    in anchor_set_global
                    else "bridge"
                )

                node_writer.writerow([
                    key,
                    group_patient,
                    global_node,
                    cell_id[
                        global_node
                    ],
                    role,
                    time_month[
                        global_node
                    ],
                    cell_type[
                        global_node
                    ],
                    umap[
                        global_node,
                        0
                    ],
                    umap[
                        global_node,
                        1
                    ],
                ])

            # --------------------------------------------------
            # SAVE FINAL ORIGINAL-kNN EDGES
            # --------------------------------------------------

            for local_u, local_v in sorted(
                final_local_edges
            ):

                global_u = int(
                    candidate_global[
                        local_u
                    ]
                )

                global_v = int(
                    candidate_global[
                        local_v
                    ]
                )

                u, v = sorted(
                    (
                        global_u,
                        global_v,
                    )
                )

                original_weight = float(
                    graph[u, v]
                )

                if (
                    original_weight <= 0
                    or not np.isfinite(
                        original_weight
                    )
                ):
                    raise RuntimeError(
                        f"Steiner introduced non-kNN edge "
                        f"({u}, {v}) in {key}"
                    )

                final_weight += (
                    original_weight
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

            candidate_removed = (
                n_candidate
                - n_final_nodes
            )

            total_final_nodes += (
                n_final_nodes
            )

            total_final_edges += (
                n_final_edges
            )

            total_bridges += (
                final_bridge_count
            )

            total_candidate_removed += (
                candidate_removed
            )

            summary_by_key[
                key
            ].update({
                "n_candidate_nodes":
                    n_candidate,
                "n_candidate_edges":
                    n_candidate_edges,
                "n_metric_edges":
                    k * (k - 1) // 2,
                "n_metric_mst_edges":
                    int(metric_mst.nnz),
                "n_expanded_union_nodes":
                    len(expanded_nodes),
                "n_expanded_union_edges":
                    len(expanded_edges),
                "n_steiner_nodes":
                    n_final_nodes,
                "n_steiner_bridge_nodes":
                    final_bridge_count,
                "n_steiner_edges":
                    n_final_edges,
                "steiner_total_weight":
                    final_weight,
                "candidate_nodes_removed":
                    candidate_removed,
                "status":
                    "complete",
            })

            if (
                group_number == 1
                or group_number % 10 == 0
            ):

                elapsed = (
                    time.monotonic()
                    - started
                ) / 60

                print(
                    f"[06] "
                    f"{group_number:,}/{n_multi:,} groups; "
                    f"{total_final_nodes:,} final node rows; "
                    f"{total_candidate_removed:,} "
                    f"candidate nodes removed; "
                    f"{elapsed:.1f} min elapsed.",
                    flush=True,
                )

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    print(
        "[07] Writing summary...",
        flush=True,
    )

    with open(
        summary_tmp,
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=
                summary_columns,
        )

        writer.writeheader()

        for key in sorted(
            summary_by_key
        ):
            writer.writerow(
                summary_by_key[key]
            )

    node_tmp.replace(
        node_final
    )

    edge_tmp.replace(
        edge_final
    )

    summary_tmp.replace(
        summary_final
    )

    elapsed_minutes = (
        time.monotonic()
        - started
    ) / 60

    qc = {
        "method":
            "kou_markowsky_berman_steiner_approximation",
        "candidate_source":
            "exact Step 05 MST candidate vertex sets",
        "candidate_graph":
            "induced subgraph of original Step 02 kNN graph",
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
                (
                    groups[
                        "n_anchor_cells"
                    ]
                    == 1
                ).sum()
            ),
        "total_steiner_node_rows":
            int(
                total_final_nodes
            ),
        "total_steiner_bridge_rows":
            int(
                total_bridges
            ),
        "total_steiner_edge_rows":
            int(
                total_final_edges
            ),
        "total_candidate_nodes_removed":
            int(
                total_candidate_removed
            ),
        "terminal_rule":
            "patient-specific clonotype anchors",
        "edge_provenance":
            "all final Steiner edges verified against original Step 02 graph",
        "exact_optimum_claimed":
            False,
        "root_used":
            False,
        "elapsed_minutes":
            round(
                elapsed_minutes,
                2,
            ),
    }

    with open(
        OUT
        / "06_method3_steiner_qc.json",
        "w",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    print(
        "\n===== STEP 06 QC =====",
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
        "\n[SUCCESS] Step 06 completed.",
        flush=True,
    )


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print(
            "\n[ERROR] Step 06 failed.",
            flush=True,
        )

        traceback.print_exc()

        sys.exit(1)
