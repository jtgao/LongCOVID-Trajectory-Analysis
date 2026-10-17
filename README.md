# LongCOVID-Trajectory-Analysis

## Overview

Computational research on longitudinal Long COVID PBMC single-cell data, combining lineage-specific T/B-cell processing, TCR clonotype integration, and graph-based T-cell trajectory reconstruction. The longer-term goal is uncertainty-aware, probabilistic trajectory modeling.

The implemented trajectory methods focus on **T cells and TCRs**. B-cell processing is included; B-cell/BCR trajectory modeling is a future extension. This is a public source-code portfolio, **not currently a raw-data-to-final-result end-to-end reproducible package**. Private biological data and generated results are not included.

## Research Questions

- How can patient-specific clonotypes provide anchors for transcriptional-state reconstruction?
- How do rooted shortest paths, MSTs, and Steiner approximations differ in selected edges and bridge cells?
- Which reconstruction features are stable across methods?
- How can uncertainty over alternative feasible trees be quantified and assessed with longitudinal evidence?

## Repository Structure

```text
LongCOVID-Trajectory-Analysis/
├── README.md
├── LICENSE
├── .gitignore
├── trajectory_model/
│   ├── 01_input_audit/01_audit_tcr_knn_inputs.py
│   ├── 02_knn_graph/02_prepare_tcell_knn_graph.py
│   ├── 03_tcr_anchors/03_prepare_patient_tcr_anchors.py
│   ├── 04_root_shortest_paths/04_root_to_anchor_shortest_paths.py
│   ├── 05_mst_reconstruction/05_patient_tcr_mst.py
│   ├── 06_steiner_reconstruction/06_patient_tcr_steiner.py
│   ├── 07_cross_method_comparison/07_compare_reconstruction_methods.py
│   ├── 08_visualization/08_visualize_reconstruction_examples.py
│   └── 09_method_validation/09_handwritten_method_validation.py
└── lineage_processing/
    ├── reintegration/reintegrate_lineages_by_cluster.py
    ├── clustering/
    │   ├── run_seurat_tcells.r
    │   └── run_seurat_bcells.r
    ├── annotation/
    │   ├── annotate_lineage_manual_celltypes.py
    │   └── run_celltypist_seurat_cluster_review.py
    ├── marker_review/
    │   ├── integrate_protein.py
    │   ├── plot_tcell_seurat_cluster_marker_profiles.py
    │   └── plot_bcell_seurat_cluster_marker_profiles.py
    └── cleanup/
        ├── annotate_remove_manual_contaminants.py
        └── remove_contam_clusters.py
```

Source directories and runtime output directories are separate. No cluster submission wrappers are included.

## Lineage Processing

The Scanpy branch selects T/B cells by user-supplied overall cluster IDs, normalizes expression, selects highly variable genes, performs PCA and optional Harmony integration, and computes neighbors, UMAP, and Leiden subclusters. Manual annotation mappings and contaminant-removal lists are runtime inputs.

The cleaned-object sequence is:

1. `reintegrate_lineages_by_cluster.py` → `processed_reintegrated.h5ad`.
2. `annotate_lineage_manual_celltypes.py` → `processed_reintegrated_manual_annotated.h5ad`.
3. `annotate_remove_manual_contaminants.py` → `processed_reintegrated_manual_annotated_clean.h5ad`.
4. `remove_contam_clusters.py` → explicitly supplied cleaned T/B output paths.

The final cleaned T-cell object supplies the graph for trajectory stage 02. Cleanup subsets the existing object; it does not rebuild neighbors.

The native Seurat scripts are a **parallel branch** starting from an overall RNA object. They retain lineage-specific cluster whitelists and analysis settings, use CCA integration, and produce native Seurat clusters. Review these study-specific cluster whitelists before applying the scripts to another dataset. They require Python/Scanpy for their embedded matrix-transfer helper as well as R/Seurat; they create and remove temporary transfer files under the supplied output directory.

CellTypist review matches Seurat cells to overall RNA counts, normalizes to log1p counts per 10,000, predicts labels, and summarizes support by cluster. This produces review evidence, not automatically validated biological annotations.

ADT integration aligns antibody-capture features to cleaned RNA cells. Its documented conventions are `<sample>_sample_filtered_feature_bc_matrix.h5` and cell IDs `<sample>_<barcode>`. Missing aligned protein values are filled with zero, following the original implementation.

T/B marker-profile scripts use native Seurat cluster IDs, RNA log-normalized cluster means, and per-marker CLR ADT cluster means, followed by marker-wise z-scores across clusters. They expect named protein columns in `obsm["protein"]`; panel definitions and feature aliases are in each script. RNA and ADT evidence supports manual review rather than assigning biological identities automatically.

### Runtime interfaces

Python scripts are command-line programs, not an importable package. Their argument parsers provide input descriptions. Scripts that read Seurat objects also require an available `Rscript` with Seurat and SeuratObject.

| Component | Input/output configuration |
|---|---|
| Reintegration | Existing CLI: `--h5ad`, `--out_base`, `--t_clusters`, `--b_clusters`; optional `--do_harmony`, batch key and metadata mapping |
| Manual annotation | Existing T/B input and mapping arguments; per-patient plots require `--per-patient-plots` |
| Cleanup | Existing T/B paths and removal arguments; label cleanup writes its fixed basename beside each input |
| Native Seurat T/B scripts | Two positional arguments: overall H5AD and output directory |
| CellTypist review | `--overall-h5ad`, `--t-seurat-rds`, `--b-seurat-rds`, `--out-dir`; optional `--model`, `--rscript` |
| ADT integration | `--rna_h5ad`, `--raw_h5_dir`, `--out_h5ad` |
| T/B marker profiles | `--seurat-rds`, `--h5ad`, `--out-dir`; optional `--rscript` |

Use user-owned working copies and output directories when running analyses. Some annotation scripts write new objects beside their inputs.

## TCR Anchor Definition

Stage 03 joins externally supplied per-cell TCR metadata to the validated graph cell index. A group is defined by patient identity and cleaned TRB CDR3 amino-acid sequence. Multi-anchor reconstruction requires at least two anchors; this grouping is an operational definition, not proof of a complete biological lineage.

The external metadata input must provide:
`cell_id`, `patient_from_sample`, `cdr3_aa_TRB_clean`, `has_TRB`, and `patient_clonotype_key`.
The implementation checks cell identifiers, patient consistency, and grouping consistency. This repository does **not** currently generate that metadata end-to-end.

## Shared kNN Graph

Stage 02 reads an existing cleaned T-cell neighbor-distance graph in **Harmony-corrected PCA space**. It validates the representation and symmetrizes existing neighbor relationships; it does not infer graph geometry from UMAP.

The preserved implementation expects `X_pca_harmony` with 50 dimensions, `X_umap`, `obsp["distances"]`, and neighbor metadata specifying 15 neighbors and Euclidean distance. Required observation fields include `patient`, `time_month`, and `manual_cell_type`.

All reconstructed edges come from this existing kNN graph. UMAP coordinates are for visualization only.

## Rooted Shortest Paths

Stage 04 chooses the earliest observed anchor as the root, breaking same-time ties by graph node index, then unions root-to-anchor shortest paths. The current implementation supports observation months 1, 3, and 6. This temporal root convention does not establish biological ancestry.

## MST Reconstruction

Stage 05 computes anchor-pair shortest paths on the shared graph. Their union of nodes defines a candidate set; the candidate graph includes existing kNN edges induced by that set. A minimum spanning tree connects the candidate nodes.

This is an MST of the candidate graph, not a direct MST of UMAP coordinates or merely a tree on anchor coordinates.

## KMB Steiner Reconstruction

Stage 06 uses the candidate-node sets from stage 05. It constructs the terminal metric closure, computes its MST, expands selected terminal connections into graph paths, computes an MST on the expanded union, and prunes nonterminal leaves.

This is a KMB-style **approximation**, not an exact Steiner-tree solver. The expanded reconstruction uses original graph edges; metric-closure edges are computational intermediates.

## Bridge Cells

Non-anchor cells included in a reconstruction are bridge cells. They may come from **any patient**, carry another TCR, or have no valid TCR. They are transcriptional/geometric intermediates, **not inferred biological ancestors**. Patient-specific anchor grouping does not impose a patient-only bridge search.

## Cross-Method Stability

Stage 07 compares reconstruction node, edge, and bridge-cell sets, including Jaccard overlap and bridge-state summaries. Cross-method agreement measures sensitivity to the reconstruction method; it is not a posterior probability or biological validation.

Stage 08 plots selected disagreements using separately supplied **Seurat coordinates**, aligned by `cell_id`. The external export needs `seurat_umap_1` and `seurat_umap_2` columns. Coordinates change only the display, not the graph or reconstruction.

Figure titles and filenames use generic example labels by default. `--show-identifiers` explicitly enables patient/clonotype labels. Generated tables, manifests, and console output can still contain identifiers and must remain private; generic figure labels are not a complete de-identification procedure.

### Trajectory runtime configuration

Stages 01–08 take `--project-dir`: a user-selected run directory containing numbered stage folders and their `outputs/` subdirectories. It is independent of the source-code directory.

| Stage | Additional required inputs |
|---|---|
| 01 | `--annotated-t`, `--protein-h5ad`, `--seurat-rds` |
| 02 | `--h5ad`: cleaned T-cell object with the existing graph |
| 03 | `--tcr-metadata`: external per-cell metadata CSV/CSV.GZ |
| 04–07 | Upstream stage outputs under `--project-dir` |
| 08 | `--seurat-coordinates`: external coordinate CSV/CSV.GZ |
| 09 | Independent `--out-dir`; no biological input |

Stages 04 and 05 use stage-02/03 outputs; stage 06 additionally uses stage-05 candidate information. Stage 07 compares stages 04–06; stage 08 consumes the comparison and reconstruction tables.

## Probabilistic Extension

**In development:** MCMC sampling over alternative feasible trajectory trees and probabilistic edge/bridge inclusion summaries. These are research directions, not implemented or completed MCMC experiments in this release. Longitudinal biological validation is also ongoing.

## Validation

Stage 09 contains a synthetic weighted graph, manually specified expected reconstructions, and comparisons of reconstructed nodes, edges, and weights. Synthetic examples are retained and contain no real patient data.

The validation implementation is separate from the production scripts. It demonstrates algorithmic behavior but is not a regression-test suite directly importing stages 04–06, nor evidence of biological correctness.

This public-copy sanitization was reviewed statically only. No project code or tests were executed as part of that process; runtime portability remains unverified.

## Data Availability

Private raw data, processed objects, patient metadata, TCR tables, and generated results are not included. Users must supply compatible authorized inputs. External stage-03 metadata and stage-08 coordinates are explicit dependencies.

Consequently, this repository is **not currently an end-to-end reproducible package from raw data to final results**. The ignore rules keep common biological objects, tables, figures, and logs out of ordinary version-control additions.

## Technical Stack

- Python: NumPy, pandas, SciPy, Matplotlib, Scanpy, AnnData.
- Harmony integration through Scanpy/harmonypy; CellTypist for annotation review.
- R: Seurat v5 APIs, SeuratObject, dplyr.
- Sparse weighted graphs, Dijkstra shortest paths, minimum spanning trees, and KMB-style Steiner approximation.

Dependencies and model files must be available in the user's environment. This release does not provide a tested environment lockfile or bundled models.

## Research Context

This research is conducted in the **Ji Lab**, Department of Biostatistics, **Johns Hopkins Bloomberg School of Public Health**, under **Prof. Hongkai Ji**.

The trajectory/probabilistic component also supports an **M.S.E. thesis in Applied Mathematics and Statistics at Johns Hopkins University**, advised by **Prof. Holden Lee**.

## Current Status

**Implemented in source:** lineage reintegration/reclustering; annotation and marker review; patient-specific TCR anchors; shared T-cell kNN graph preparation; rooted shortest paths; MST reconstruction; KMB Steiner approximation; bridge-cell summaries; cross-method comparison; synthetic method validation.

**In development:** MCMC over feasible alternative trees, probabilistic inclusion summaries, and longitudinal biological validation. B-cell/BCR trajectory modeling is a future extension. No completed probabilistic or biological findings are claimed here.

## Author

Jiatong Gao.
