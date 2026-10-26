# 2D kNN / UMAP diagnostic

Build a separate 15-neighbor graph on existing Seurat UMAP coordinates.
Compare rooted shortest paths, candidate-graph MST, and KMB Steiner
reconstruction using the same geometry for graph construction and plotting.

The production 50D Harmony graph remains unchanged. Each selected edge
must exist in the diagnostic graph. Unused cells and edges need not
enter the reconstruction.

## Requirements

Python with NumPy, pandas, SciPy, and matplotlib.

## Usage

    python 10_2d_knn_umap_validation.py --help

Provide --project-dir pointing to your private stage outputs and
--seurat-coordinates pointing to the private Seurat coordinate export.
Keep inputs and generated outputs outside this public repository.

## Validation

Inspect edge membership, anchor retention, connectivity, and tree checks.
Passing these diagnostic examples does not establish biological validity
or independently validate the production 50D results.

No biological data, generated results, or logs are included here.
