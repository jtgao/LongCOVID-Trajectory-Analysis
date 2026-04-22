# LongCOVID-Trajectory-Analysis

This repository contains code for analyzing single-cell RNA-seq (scRNA-seq) data from Long COVID studies, with a focus on preprocessing, lineage-specific clustering, and downstream analysis toward trajectory modeling.

## Overview

This project integrates single-cell transcriptomic data with immune receptor (TCR/BCR) information to explore immune cell states and their potential transitions.

The workflow currently includes:
- Data preprocessing and integration
- Lineage-specific reclustering (T and B cells)
- Marker-based annotation and validation
- Pseudobulk extraction for downstream differential expression analysis

Future work aims to incorporate:
- Trajectory inference
- Clonotype-informed modeling of cell state transitions

---

## Repository Structure

### `Integration-Lineage-by-Clusters`
Scripts and workflows for:
- Subsetting T and B cell lineages
- Reintegrating lineage-specific data
- Reclustering using PCA / Harmony / Leiden
- Generating UMAP visualizations and marker-based annotations

### `Extract-Pseudobulk`
Code for:
- Aggregating single-cell data into pseudobulk profiles
- Preparing inputs for differential expression analysis

---

## Data

This project operates on:
- scRNA-seq data (AnnData `.h5ad` format)
- Associated metadata (sample, patient, timepoint, etc.)
- Optional TCR/BCR clonotype information

Data is not included in this repository due to size and privacy constraints.

---

## Methods Summary

1. **Preprocessing**
   - Quality control (gene counts, mitochondrial percentage)
   - Normalization and log transformation

2. **Dimensionality Reduction**
   - PCA (with optional Harmony batch correction)

3. **Clustering**
   - Leiden clustering on lineage-specific subsets

4. **Annotation**
   - Marker gene-based manual cell type assignment

5. **Validation**
   - Heatmaps of canonical markers
   - UMAP visualization of gene expression

---

## Notes

- Heatmaps are generated without per-gene standardization to preserve relative expression levels across groups.
- T cell populations exhibit more continuous transcriptional states compared to B cells, which show more discrete marker patterns.

---

## Future Directions

- Incorporation of TCR/BCR clonotype information
- Graph-based modeling of cell-state trajectories
- Integration with longitudinal timepoint data

---

## Requirements

Typical dependencies include:
- Python 3.9+
- scanpy
- anndata
- pandas
- numpy
- matplotlib

---

## Author

Jiatong Gao  
Applied Mathematics & Statistics, Johns Hopkins University

---

## License

See `LICENSE` for details.
