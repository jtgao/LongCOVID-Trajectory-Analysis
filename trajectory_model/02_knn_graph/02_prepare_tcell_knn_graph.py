#!/usr/bin/env python3
"""
STEP 02 — Prepare and validate the full T-cell kNN graph.

Purpose:
    Reuse the existing 15-neighbor graph built from X_pca_harmony.
    Retain ALL T cells and only existing kNN cell-pair connections.

Method:
    Read the existing Scanpy distance matrix through AnnData.
    Symmetrize its existing neighbor relationships.
    Preserve Harmony-space Euclidean edge weights.
    Check connected components and edge provenance.

Outputs:
    02_T_knn_graph.npz
    02_T_cell_index.csv.gz
    02_T_knn_edges.csv.gz
    02_T_knn_graph_qc.json

No shortest paths, MSTs, Steiner trees, or TCR filtering.
Original H5AD is never modified.
"""

# This runs before importing any scientific Python packages.
print("[START] Step 02 Python process started.", flush=True)

import json
import sys
import traceback
from pathlib import Path

print(f"[IMPORT] Python executable: {sys.executable}", flush=True)

print("[IMPORT] NumPy...", flush=True)
import numpy as np
print("[IMPORT] NumPy OK", flush=True)

print("[IMPORT] Pandas...", flush=True)
import pandas as pd
print("[IMPORT] Pandas OK", flush=True)

print("[IMPORT] SciPy...", flush=True)
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
print("[IMPORT] SciPy OK", flush=True)

print("[IMPORT] AnnData...", flush=True)
import anndata as ad
print("[IMPORT] AnnData OK", flush=True)


import argparse

parser = argparse.ArgumentParser(description="Trajectory stage 02. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
parser.add_argument("--h5ad", type=Path, required=True, help="Cleaned T-cell H5AD with existing Harmony neighbors and metadata.")
args = parser.parse_args()
PROJECT = args.project_dir
OUT = PROJECT / "02_knn_graph" / "outputs"
H5AD = args.h5ad


def main():

    print("[01] Checking input path...", flush=True)

    if not H5AD.is_file():
        raise FileNotFoundError(H5AD)

    print(f"[INPUT] {H5AD}", flush=True)
    print(f"[INPUT] Size: {H5AD.stat().st_size:,} bytes", flush=True)

    print("[02] Opening H5AD in backed read-only mode...", flush=True)

    adata = ad.read_h5ad(H5AD, backed="r")

    try:

        print("[02] H5AD opened successfully.", flush=True)

        n = adata.n_obs

        print(
            f"[INFO] Cells: {n:,}; genes: {adata.n_vars:,}",
            flush=True,
        )

        if not adata.obs_names.is_unique:
            raise ValueError("Cell IDs are not unique.")

        for key in ["X_pca_harmony", "X_umap"]:
            if key not in adata.obsm:
                raise KeyError(f"Missing obsm[{key!r}]")

        if "distances" not in adata.obsp:
            raise KeyError("Missing obsp['distances']")

        for column in ["patient", "time_month", "manual_cell_type"]:
            if column not in adata.obs:
                raise KeyError(f"Missing obs column: {column}")

        neighbors = adata.uns.get("neighbors", {})
        params = neighbors.get("params", {})

        print(f"[INFO] Neighbor parameters: {params}", flush=True)

        if params.get("use_rep") != "X_pca_harmony":
            raise ValueError(
                "Existing neighbor graph does not use X_pca_harmony."
            )

        if params.get("metric") != "euclidean":
            raise ValueError(
                "Existing neighbor graph does not use Euclidean distance."
            )

        if int(params.get("n_neighbors", -1)) != 15:
            raise ValueError(
                "Existing neighbor graph does not use 15 neighbors."
            )

        if adata.obsm["X_pca_harmony"].shape != (n, 50):
            raise ValueError("Unexpected Harmony dimensions.")

        print(
            "[03] Loading existing sparse kNN distance matrix...",
            flush=True,
        )

        original = adata.obsp["distances"].tocsr().copy()

        print(
            f"[03] Matrix loaded: {original.shape}, "
            f"{original.nnz:,} stored entries",
            flush=True,
        )

        if original.shape != (n, n):
            raise ValueError("Incorrect distance matrix dimensions.")

        original.sum_duplicates()

        if not np.isfinite(original.data).all():
            raise ValueError("Non-finite kNN weights.")

        if np.any(original.data < 0):
            raise ValueError("Negative kNN weights.")

        # A sparse graph can contain explicit zero-weight edges.
        # Do not silently delete legitimate cell-pair connections.
        original.setdiag(0)
        original.eliminate_zeros()
        original.sort_indices()

        print(
            "[04] Symmetrizing existing kNN relationships...",
            flush=True,
        )

        # Undirected union of existing neighbor pairs.
        # This does NOT create connections between previously
        # non-neighboring cells.
        graph = original.maximum(original.T).tocsr()

        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.sort_indices()

        print(
            f"[04] Undirected edges: {graph.nnz // 2:,}",
            flush=True,
        )

        print("[05] Checking edge provenance...", flush=True)

        allowed = original.copy()
        allowed.data = np.ones(allowed.nnz, dtype=np.int8)
        allowed = allowed.maximum(allowed.T).tocsr()

        actual = graph.copy()
        actual.data = np.ones(actual.nnz, dtype=np.int8)

        unexpected = actual - allowed

        if unexpected.nnz:
            raise RuntimeError(
                f"Found {unexpected.nnz} graph entries absent "
                "from the original kNN neighbor relationships."
            )

        if (graph != graph.T).nnz:
            raise RuntimeError("Undirected graph is not symmetric.")

        print("[05] Edge provenance validated.", flush=True)

        print("[06] Checking connected components...", flush=True)

        n_components, component_labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

        component_sizes = np.bincount(
            component_labels,
            minlength=n_components,
        )

        degrees = np.diff(graph.indptr)

        print(
            f"[06] Components: {n_components:,}; "
            f"largest: {component_sizes.max():,} cells",
            flush=True,
        )

        print(
            "[07] Validating sampled edge weights "
            "against Harmony-space distances...",
            flush=True,
        )

        upper = sp.triu(graph, k=1, format="coo")

        sample_size = min(1000, upper.nnz)

        rng = np.random.default_rng(42)

        sampled = rng.choice(
            upper.nnz,
            size=sample_size,
            replace=False,
        )

        u = upper.row[sampled]
        v = upper.col[sampled]

        # Load only the sampled Harmony rows for the weight check.
        harmony = adata.obsm["X_pca_harmony"]

        expected = np.linalg.norm(
            np.asarray(harmony[u], dtype=np.float64)
            - np.asarray(harmony[v], dtype=np.float64),
            axis=1,
        )

        observed = upper.data[sampled]

        max_difference = (
            float(np.max(np.abs(expected - observed)))
            if sample_size
            else 0.0
        )

        print(
            f"[07] Maximum sampled distance difference: "
            f"{max_difference:.8g}",
            flush=True,
        )

        if not np.allclose(
            expected,
            observed,
            rtol=1e-4,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Some sampled edge weights do not match "
                "Harmony-space Euclidean distances."
            )

        print("[08] Saving validated graph...", flush=True)

        OUT.mkdir(parents=True, exist_ok=True)

        sp.save_npz(
            OUT / "02_T_knn_graph.npz",
            graph,
            compressed=True,
        )

        print("[08] Graph saved.", flush=True)

        print("[09] Saving cell index and UMAP...", flush=True)

        umap = np.asarray(adata.obsm["X_umap"])

        if umap.shape != (n, 2):
            raise ValueError("Unexpected UMAP dimensions.")

        cell_ids = adata.obs_names.astype(str).to_numpy()

        cell_table = pd.DataFrame({
            "node_index": np.arange(n, dtype=np.int64),
            "cell_id": cell_ids,
            "patient": adata.obs["patient"].astype(str).to_numpy(),
            "time_month": adata.obs["time_month"].to_numpy(),
            "manual_cell_type": (
                adata.obs["manual_cell_type"]
                .astype(str)
                .to_numpy()
            ),
            "umap_1": umap[:, 0],
            "umap_2": umap[:, 1],
            "knn_component": component_labels,
            "knn_degree": degrees,
        })

        cell_table.to_csv(
            OUT / "02_T_cell_index.csv.gz",
            index=False,
            compression="gzip",
        )

        print("[09] Cell index saved.", flush=True)

        print("[10] Exporting full kNN edge table...", flush=True)

        edge_path = OUT / "02_T_knn_edges.csv.gz"

        # Stream the edge table in chunks.
        # Each undirected cell pair appears exactly once.
        import gzip

        with gzip.open(edge_path, "wt", newline="") as handle:

            handle.write(
                "node_u_index,node_v_index,"
                "cell_u,cell_v,weight\n"
            )

            chunk_size = 100000

            for start in range(0, upper.nnz, chunk_size):

                stop = min(start + chunk_size, upper.nnz)

                uu = upper.row[start:stop]
                vv = upper.col[start:stop]

                chunk = pd.DataFrame({
                    "node_u_index": uu,
                    "node_v_index": vv,
                    "cell_u": cell_ids[uu],
                    "cell_v": cell_ids[vv],
                    "weight": upper.data[start:stop],
                })

                chunk.to_csv(
                    handle,
                    index=False,
                    header=False,
                )

                print(
                    f"[10] Exported {stop:,}/{upper.nnz:,} edges",
                    flush=True,
                )

        print("[10] Edge table saved.", flush=True)

        qc = {
            "source_h5ad": str(H5AD),
            "n_cells": int(n),
            "representation": "X_pca_harmony",
            "representation_dimensions": 50,
            "n_neighbors": 15,
            "distance_metric": "euclidean",
            "original_directed_entries": int(original.nnz),
            "undirected_edges": int(upper.nnz),
            "n_connected_components": int(n_components),
            "largest_component_size": int(
                component_sizes.max()
            ),
            "smallest_component_size": int(
                component_sizes.min()
            ),
            "isolated_cells": int(np.sum(degrees == 0)),
            "minimum_degree": int(degrees.min()),
            "median_degree": float(np.median(degrees)),
            "maximum_degree": int(degrees.max()),
            "sampled_edges_checked": int(sample_size),
            "maximum_sampled_weight_difference": max_difference,
            "edge_set_validated": True,
            "new_cell_pair_edges_created": 0,
        }

        qc_path = OUT / "02_T_knn_graph_qc.json"

        with open(qc_path, "w") as handle:
            json.dump(qc, handle, indent=2)

        print("\n===== GRAPH QC =====", flush=True)
        print(json.dumps(qc, indent=2), flush=True)

        print(
            "\n[SUCCESS] Step 02 completed. "
            "No trajectory methods were run.",
            flush=True,
        )

    finally:

        if adata.isbacked:
            adata.file.close()


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print("\n[ERROR] Step 02 failed:", flush=True)
        traceback.print_exc()
        sys.exit(1)
