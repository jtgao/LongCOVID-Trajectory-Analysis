# LongCOVID-Trajectory-Analysis

Computational analysis of longitudinal Long COVID single-cell data, with a focus on **clonotype-informed immune cell trajectory reconstruction**, graph-based modeling, and uncertainty quantification.

This work is conducted in the **Ji Lab, Department of Biostatistics, Johns Hopkins Bloomberg School of Public Health**, under Prof. Hongkai Ji. The trajectory-modeling component also supports my M.S.E. thesis in Applied Mathematics and Statistics at Johns Hopkins University, advised by Prof. Holden Lee.

---

## Overview

This project develops a computational framework for reconstructing immune cell-state trajectories from longitudinal single-cell RNA sequencing data.

The analysis integrates:

- single-cell RNA-seq data
- longitudinal patient and sampling metadata
- T-cell receptor (TCR) clonotype information
- lineage-specific transcriptional embeddings
- graph-based trajectory reconstruction
- probabilistic uncertainty modeling

The current implemented trajectory framework focuses on **T cells and patient-specific TCR clonotypes**.

Cells sharing the same patient-specific clonotype are treated as biologically informed **anchor cells**, while intermediate cells are selected from a shared transcriptional-state graph according to graph geometry.

Bridge cells are interpreted as **geometric/transcriptional intermediates**, rather than inferred biological ancestors.

---

## Current Research Questions

The project asks:

1. How can clonotype information constrain trajectory reconstruction beyond transcriptional similarity alone?
2. Which graph-based reconstruction methods produce stable connections between clonotype-sharing cells?
3. Which intermediate cell states repeatedly occur along reconstructed trajectories?
4. How sensitive are inferred trajectories to the reconstruction method?
5. How can uncertainty over alternative feasible trajectory structures be quantified probabilistically?
6. How can longitudinal timepoints and biological annotations be used to validate inferred trajectory structures?

---

## Dataset

The current analysis uses longitudinal PBMC single-cell data from **16 patients**.

The trajectory pipeline operates on a cleaned T-cell dataset containing more than **155,000 cells**.

Available information includes:

- scRNA-seq expression profiles
- patient identifiers
- longitudinal sampling timepoints
- TCR clonotype information
- lineage-specific clustering
- provisional biological annotations
- RNA and ADT marker profiles

Raw and processed biological data are not included in this repository because of data size and privacy restrictions.

---

## Analysis Workflow

### 1. Single-Cell Preprocessing

The upstream workflow includes:

- cell- and gene-level quality control
- normalization and log transformation
- highly variable gene selection
- dimensionality reduction with PCA
- sample-level batch integration with Harmony
- nearest-neighbor graph construction
- Leiden clustering
- UMAP visualization

Immune receptor genes are retained for biological interpretation but excluded from the features used to construct the transcriptional embedding.

---

### 2. Lineage-Specific Reintegration and Reclustering

T- and B-cell populations are processed independently after global integration.

Current workflows include:

- lineage subsetting
- lineage-specific PCA
- Harmony integration
- nearest-neighbor construction
- reclustering
- comparison of clustering configurations
- generation of cluster-level summaries

The current trajectory implementation uses the cleaned **T-cell lineage**.

---

### 3. Cell-State Annotation

Computational subclusters are reviewed using multiple sources of evidence:

- CellTypist predictions
- canonical RNA markers
- ADT / surface-protein profiles
- cluster-level marker heatmaps
- UMAP visualization
- expert biological review

Current annotations are treated as provisional biological interpretations rather than ground-truth labels.

---

## TCR Clonotype Integration

TCR metadata are integrated with the cleaned T-cell object using exact cell identifiers.

Clonotypes are defined using a **patient-specific TCR key**:

```text
(patient identity, cleaned TCR clonotype)
