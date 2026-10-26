#!/usr/bin/env python3
"""Independent 2D diagnostic. Production inputs are read-only."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import (
    connected_components, dijkstra, minimum_spanning_tree,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_path(pred, source, target):
    """Recover every original graph edge along a Dijkstra path."""
    source, target = int(source), int(target)
    path = [target]
    seen = {target}
    while path[-1] != source:
        previous = int(pred[path[-1]])
        require(previous >= 0, "Missing predecessor: unreachable anchor.")
        require(previous not in seen, "Cycle in predecessor chain.")
        seen.add(previous)
        path.append(previous)
    return path[::-1]


def path_edges(path):
    return {
        tuple(sorted((int(u), int(v))))
        for u, v in zip(path[:-1], path[1:])
    }


def mst_edges(graph, global_nodes):
    result = minimum_spanning_tree(graph).tocoo()
    return {
        tuple(sorted((int(global_nodes[u]), int(global_nodes[v]))))
        for u, v in zip(result.row, result.col)
    }


def edge_graph(edges, graph):
    """Compact weighted graph and its global node indices."""
    nodes = np.array(sorted({v for edge in edges for v in edge}), dtype=int)
    index = {int(v): i for i, v in enumerate(nodes)}
    rows, cols, values = [], [], []
    for u, v in sorted(edges):
        weight = float(graph[u, v])
        require(weight > 0, f"Non-input edge encountered: {u}, {v}")
        rows.extend([index[u], index[v]])
        cols.extend([index[v], index[u]])
        values.extend([weight, weight])
    matrix = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(nodes), len(nodes))
    )
    return matrix, nodes


def prune_leaves(edges, anchors):
    """Remove non-anchor leaves from the KMB tree."""
    neighbors = {}
    for u, v in edges:
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)
    queue = [
        v for v, adjacent in neighbors.items()
        if v not in anchors and len(adjacent) == 1
    ]
    result = set(edges)
    while queue:
        v = queue.pop()
        if v in anchors or len(neighbors[v]) != 1:
            continue
        u = next(iter(neighbors[v]))
        result.remove(tuple(sorted((u, v))))
        neighbors[v].remove(u)
        neighbors[u].remove(v)
        if u not in anchors and len(neighbors[u]) == 1:
            queue.append(u)
    return result


def reconstruct(graph, anchor_nodes, root):
    """Mirror stages 04–06, with the diagnostic graph as input."""
    anchors = np.asarray(sorted(anchor_nodes), dtype=int)

    # Stage 04: union of root-to-anchor shortest paths.
    distances, pred = dijkstra(
        graph, directed=False, indices=int(root), return_predecessors=True
    )
    require(np.isfinite(distances[anchors]).all(), "Disconnected anchors.")
    shortest = set()
    for target in anchors:
        shortest.update(path_edges(recover_path(pred, root, target)))

    # Stage 05: candidate vertices from every anchor-pair shortest path.
    candidate = set(map(int, anchors))
    for i, source in enumerate(anchors[:-1]):
        distances, pred = dijkstra(
            graph, directed=False, indices=int(source),
            return_predecessors=True,
        )
        require(np.isfinite(distances[anchors[i + 1:]]).all(),
                "Disconnected anchor pair.")
        for target in anchors[i + 1:]:
            candidate.update(recover_path(pred, source, target))

    candidate = np.asarray(sorted(candidate), dtype=int)
    H = graph[candidate][:, candidate].tocsr()
    H.sort_indices()
    mst = mst_edges(H, candidate)

    # Stage 06: KMB on the SAME induced candidate graph H.
    lookup = {int(v): i for i, v in enumerate(candidate)}
    terminals = np.array([lookup[int(v)] for v in anchors], dtype=int)
    distances, predecessors = dijkstra(
        H, directed=False, indices=terminals, return_predecessors=True
    )
    metric = distances[:, terminals]
    require(np.isfinite(metric).all(), "Nonfinite terminal metric.")
    metric_tree = minimum_spanning_tree(
        sparse.csr_matrix(metric)
    ).tocoo()
    require(metric_tree.nnz == len(anchors) - 1, "Metric MST incomplete.")

    expanded = set()
    for i, j in zip(metric_tree.row, metric_tree.col):
        local_path = recover_path(
            predecessors[int(i)], terminals[int(i)], terminals[int(j)]
        )
        expanded.update(path_edges(candidate[local_path]))

    Q, expanded_nodes = edge_graph(expanded, graph)
    steiner = prune_leaves(
        mst_edges(Q, expanded_nodes), set(map(int, anchors))
    )
    return {
        "Shortest path": shortest,
        "Candidate MST": mst,
        "KMB Steiner": steiner,
    }


def validate(edges, anchors, graph):
    nodes = {v for edge in edges for v in edge}
    invalid = [(u, v) for u, v in edges if graph[u, v] <= 0]
    matrix, _ = edge_graph(edges, graph)
    components = int(connected_components(matrix, directed=False)[0])
    covered = set(anchors).issubset(nodes)
    is_tree = components == 1 and len(edges) == len(nodes) - 1
    require(not invalid and covered and is_tree,
            "Validation failed: edge provenance, anchors, or tree structure.")
    return {
        "n_anchors": len(anchors),
        "n_nodes": len(nodes),
        "n_bridges": len(nodes - set(anchors)),
        "n_edges": len(edges),
        "invalid_knn_edges": len(invalid),
        "all_anchors_present": covered,
        "components": components,
        "is_tree": is_tree,
        "total_cost_2d": sum(float(graph[u, v]) for u, v in edges),
        "status": "PASS",
    }


def draw_example(xy, all_edges, methods, anchors, output, label):
    selected_nodes = sorted({
        v for edges in methods.values() for edge in edges for v in edge
    })
    selected_xy = xy[selected_nodes]
    low, high = selected_xy.min(axis=0), selected_xy.max(axis=0)
    padding = np.maximum((high - low) * 0.18, 0.12)
    low, high = low - padding, high + padding

    # Draw every background edge that intersects the zoom rectangle.
    endpoints = xy[all_edges]
    edge_low, edge_high = endpoints.min(axis=1), endpoints.max(axis=1)
    visible = np.all(edge_high >= low, axis=1) & np.all(edge_low <= high, axis=1)
    local_background = endpoints[visible]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for column, (name, edges) in enumerate(methods.items()):
        pairs = np.array(sorted(edges), dtype=int)
        nodes = {v for edge in edges for v in edge}
        bridges = np.array(sorted(nodes - set(anchors)), dtype=int)
        for row in range(2):
            ax = axes[row, column]
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c="0.83",
                       alpha=0.35, rasterized=True, zorder=1)
            if row == 1:
                ax.add_collection(LineCollection(
                    local_background, colors="0.68", linewidths=0.45,
                    alpha=0.45, rasterized=True, zorder=2,
                ))
            ax.add_collection(LineCollection(
                xy[pairs], colors="#1976b9", linewidths=1.3, zorder=3,
            ))
            if len(bridges):
                ax.scatter(xy[bridges, 0], xy[bridges, 1], s=11,
                           c="#1976b9", zorder=4)
            ax.scatter(xy[anchors, 0], xy[anchors, 1], s=24,
                       c="#d33b32", edgecolors="black",
                       linewidths=0.3, zorder=5)
            ax.set_title(name + (" — full view" if row == 0 else " — zoom"))
            ax.set_xlabel("Seurat UMAP 1")
            ax.set_ylabel("Seurat UMAP 2")
            ax.set_aspect("equal", adjustable="box")
            if row == 1:
                ax.set_xlim(low[0], high[0])
                ax.set_ylim(low[1], high[1])

    legend = [
        Line2D([], [], marker="o", linestyle="", color="#d33b32",
               label="Clonotype anchors"),
        Line2D([], [], marker="o", linestyle="", color="#1976b9",
               label="Selected bridge cells"),
        Line2D([], [], color="#1976b9", label="Reconstructed edges"),
        Line2D([], [], color="0.68", label="Available 2D kNN edges (zoom)"),
    ]
    fig.suptitle(
        f"{label}: 2D kNN reconstruction on the same UMAP coordinates",
        fontsize=15,
    )
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--seurat-coordinates", type=Path)
    parser.add_argument("--k", type=int, default=15,
                        help="Number of other cells per directed neighbor list.")
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--min-anchors", type=int, default=5)
    parser.add_argument("--max-anchors", type=int, default=40)
    parser.add_argument("--group-key", action="append",
                        help="Optional exact saved group_key; repeat for more.")
    args = parser.parse_args()
    require(args.n_examples > 0 and args.min_anchors >= 2,
            "Require positive example count and at least two anchors.")
    require(args.max_anchors >= args.min_anchors, "Invalid anchor bounds.")

    project = args.project_dir.resolve()
    stage = project / "10_2d_knn_umap_validation"
    output_root = stage / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    # A unique run directory prevents overwriting previous diagnostics.
    from datetime import datetime
    import os
    out = output_root / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    )
    out.mkdir()
    for folder in ("graph", "reconstructions", "figures", "tables"):
        (out / folder).mkdir()
    print(f"[OUTPUT] {out}", flush=True)

    cell_path = project / "02_knn_graph/outputs/02_T_cell_index.csv.gz"
    anchor_path = project / "03_tcr_anchors/outputs/03_T_anchor_cells.csv.gz"
    require(cell_path.is_file(), f"Missing: {cell_path}")
    require(anchor_path.is_file(), f"Missing: {anchor_path}")

    coordinate_path = args.seurat_coordinates
    if coordinate_path is None:
        # Locate the known Seurat export, without guessing among matches.
        base = project.parent.parent
        matches = sorted(
            path for pattern in (
                "Tcells_seurat_for_trajectory.csv",
                "Tcells_seurat_for_trajectory.csv.gz",
            )
            for path in base.rglob(pattern)
        )
        require(len(matches) == 1,
                "Could not identify one Seurat coordinate export. "
                "Set SEURAT_COORDS to its absolute path. "
                f"Matches: {[str(p) for p in matches]}")
        coordinate_path = matches[0]
    require(coordinate_path.is_file(), f"Missing: {coordinate_path}")
    print(f"[COORDINATES] {coordinate_path}", flush=True)

    cells = pd.read_csv(
        cell_path, dtype={"cell_id": str}
    ).sort_values("node_index").reset_index(drop=True)
    coords = pd.read_csv(coordinate_path, dtype={"cell_id": str})
    columns = ["cell_id", "seurat_umap_1", "seurat_umap_2"]
    require(set(columns).issubset(coords.columns),
            f"Coordinate file must contain {columns}")
    require(cells.cell_id.is_unique and coords.cell_id.is_unique,
            "Duplicate cell IDs.")
    require(np.array_equal(cells.node_index.to_numpy(), np.arange(len(cells))),
            "Stage 02 node_index must be consecutive from zero.")
    matched = cells[["node_index", "cell_id"]].merge(
        coords[columns], on="cell_id", how="left", validate="one_to_one"
    ).sort_values("node_index").reset_index(drop=True)
    xy = matched[columns[1:]].to_numpy(dtype=float)
    require(np.isfinite(xy).all(), "Missing/nonfinite matched UMAP coordinates.")
    require(1 <= args.k < len(xy), "Invalid k.")

    anchors = pd.read_csv(
        anchor_path, dtype={"cell_id": str, "group_key": str}
    )
    required = {"node_index", "cell_id", "group_key", "patient", "time_month"}
    require(required.issubset(anchors.columns),
            f"Anchor file needs columns: {sorted(required)}")
    require(anchors.node_index.is_unique,
            "Anchor file contains duplicate node indices.")
    ids = anchors.node_index.to_numpy(dtype=int)
    require(((ids >= 0) & (ids < len(cells))).all(), "Invalid anchor indices.")
    require(np.array_equal(
        cells.cell_id.to_numpy()[ids], anchors.cell_id.to_numpy()
    ), "Anchor cell IDs do not match Stage 02 node indices.")
    anchors["time_month"] = pd.to_numeric(anchors.time_month, errors="raise")
    require(np.isfinite(anchors.time_month).all(), "Invalid anchor months.")
    require((anchors.groupby("group_key").patient.nunique() == 1).all(),
            "A saved anchor group spans multiple patients.")

    print(f"[GRAPH] Building {args.k}-NN on {len(xy):,} cells in 2D",
          flush=True)
    distances, neighbors = cKDTree(xy).query(xy, k=args.k + 1, workers=2)
    rows, cols, weights = [], [], []
    for i in range(len(xy)):
        keep = neighbors[i] != i
        js, ds = neighbors[i][keep][:args.k], distances[i][keep][:args.k]
        require(len(js) == args.k, "Incomplete neighbor list.")
        # Zero-distance edges need a separate explicit sparse-graph policy.
        require((ds > 0).all(),
                "Duplicate UMAP coordinates produced zero-length edges. "
                "Stop here rather than silently dropping or reweighting them.")
        rows.extend([i] * args.k)
        cols.extend(js.tolist())
        weights.extend(ds.tolist())
    directed = sparse.csr_matrix(
        (weights, (rows, cols)), shape=(len(xy), len(xy))
    )
    graph = directed.maximum(directed.T).tocsr()
    graph.sort_indices()
    sparse.save_npz(out / "graph/10_2d_knn_graph.npz", graph)
    matched.to_csv(out / "graph/10_cell_coordinates.csv.gz", index=False)

    upper = sparse.triu(graph, k=1).tocoo()
    all_edges = np.column_stack((upper.row, upper.col))
    pd.DataFrame({
        "u": upper.row, "v": upper.col, "weight_2d": upper.data,
    }).to_csv(out / "graph/10_2d_knn_edges.csv.gz", index=False)
    n_components, labels = connected_components(graph, directed=False)
    print(f"[GRAPH] {graph.nnz // 2:,} edges; {n_components} components",
          flush=True)

    groups = {str(key): frame for key, frame in anchors.groupby("group_key")}
    eligible = []
    for key, frame in groups.items():
        if args.min_anchors <= len(frame) <= args.max_anchors:
            if len(set(labels[frame.node_index.to_numpy(dtype=int)])) == 1:
                eligible.append(key)
    eligible.sort(key=lambda key: (len(groups[key]), key))

    if args.group_key:
        chosen = args.group_key
        require(len(set(chosen)) == len(chosen), "Duplicate requested group.")
        for key in chosen:
            require(key in groups, f"Unknown group_key: {key}")
            nodes = groups[key].node_index.to_numpy(dtype=int)
            require(len(nodes) >= 2 and len(set(labels[nodes])) == 1,
                    f"Requested group lacks connected anchors: {key}")
    else:
        require(eligible, "No eligible connected groups. Adjust anchor bounds.")
        positions = np.linspace(
            0, len(eligible) - 1, min(args.n_examples, len(eligible)),
            dtype=int,
        )
        chosen = [eligible[i] for i in positions]

    results, selections = [], []
    for number, key in enumerate(chosen, 1):
        frame = groups[key].sort_values(["time_month", "node_index"])
        nodes = frame.node_index.to_numpy(dtype=int)
        root = int(nodes[0])
        stem = f"example_{number:02d}"
        print(f"[EXAMPLE] {stem}: {len(nodes)} anchors", flush=True)
        selections.append({
            "example": stem, "group_key": key,
            "n_anchors": len(nodes), "root_node_index": root,
        })
        methods = reconstruct(graph, nodes, root)
        for name, edges in methods.items():
            qc = validate(edges, nodes, graph)
            results.append({"example": stem, "method": name, **qc})
            pairs = np.array(sorted(edges), dtype=int)
            name_slug = name.lower().replace(" ", "_")
            pd.DataFrame({
                "u": pairs[:, 0], "v": pairs[:, 1],
                "cell_id_u": cells.cell_id.to_numpy()[pairs[:, 0]],
                "cell_id_v": cells.cell_id.to_numpy()[pairs[:, 1]],
                "weight_2d": [float(graph[u, v]) for u, v in pairs],
                "exists_in_2d_knn": True,
            }).to_csv(
                out / f"reconstructions/{stem}_{name_slug}_edges.csv.gz",
                index=False,
            )
            node_set = sorted({v for edge in edges for v in edge})
            node_table = matched.iloc[node_set].copy()
            node_table["role"] = np.where(
                node_table.node_index.isin(nodes), "anchor", "bridge"
            )
            node_table.to_csv(
                out / f"reconstructions/{stem}_{name_slug}_nodes.csv.gz",
                index=False,
            )
            print(f"  {name}: {qc}", flush=True)
        draw_example(
            xy, all_edges, methods, nodes, out / "figures" / stem, stem
        )

    pd.DataFrame(selections).to_csv(
        out / "tables/10_selected_groups.csv", index=False
    )
    pd.DataFrame(results).to_csv(
        out / "tables/10_validation_summary.csv", index=False
    )
    manifest = {
        "purpose": "2D diagnostic, not biological trajectory inference",
        "coordinate_space": "existing Seurat 2D UMAP",
        "k_other_neighbors": args.k,
        "symmetrization": "union via maximum(A, A.T)",
        "weights": "Euclidean distance in the same 2D coordinates",
        "n_cells": len(xy),
        "n_edges": graph.nnz // 2,
        "n_components": int(n_components),
        "eligible_groups": len(eligible),
        "examples_run": len(chosen),
        "input_paths": {
            "cells": str(cell_path), "anchors": str(anchor_path),
            "coordinates": str(coordinate_path.resolve()),
        },
        "input_sha256": {
            "cells": file_hash(cell_path),
            "anchors": file_hash(anchor_path),
            "coordinates": file_hash(coordinate_path),
        },
        "script_sha256": file_hash(Path(__file__)),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        },
    }
    (out / "10_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[DONE] All reconstructed edges passed validation.\n{out}",
          flush=True)


if __name__ == "__main__":
    main()
