#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T-CELL SEURAT CLUSTER MARKER PROFILES
=====================================

Purpose
-------
Generate an unbiased marker-expression summary for the native
Seurat T-cell clusters.

IMPORTANT:
    - Seurat clusters only
    - Rows are native Seurat clusters in numeric order
    - NO manual_cell_type
    - NO Scanpy annotation labels
    - NO overlap/Hungarian biological labels
    - NO phenotype-specific sign flipping
    - NO reference-derived cluster naming

This is the evidence figure to inspect BEFORE assigning or
reviewing biological identities.

RNA workflow
------------
raw counts
-> normalize each cell to total 10,000
-> log1p
-> mean within each Seurat cluster
-> marker-wise z-score across all Seurat clusters
   using population SD, ddof=0

Protein workflow
----------------
raw ADT counts
-> per-marker CLR across cells

   g_j = exp(mean_i(log(1 + x_ij)))
   CLR_ij = log(1 + x_ij / g_j)

-> mean within each Seurat cluster
-> marker-wise z-score across all Seurat clusters
   using population SD, ddof=0

Output
------
One paired heatmap:
    left  = RNA marker profiles
    right = surface-protein marker profiles

Rows:
    Native Seurat cluster identifiers, ordered numerically.

Marker groups are visualization aids only.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc


# ============================================================
# CONFIGURATION
# ============================================================

RNA_TARGET_SUM = 10000.0

DISPLAY_VMIN = -2.5
DISPLAY_VMAX = 2.5


# ============================================================
# PATHS
# ============================================================

import argparse

parser = argparse.ArgumentParser(description="RNA and ADT marker profiles by native Seurat cluster.")
parser.add_argument("--seurat-rds", type=Path, required=True, help="Native Seurat object with seurat_clusters metadata.")
parser.add_argument("--h5ad", type=Path, required=True, help="RNA/counts H5AD with named ADT columns in obsm['protein']; marker aliases are defined below.")
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--rscript", default="Rscript", help="Rscript executable on PATH or explicit executable path.")
args = parser.parse_args()
SEURAT_RDS = args.seurat_rds
T_H5AD = args.h5ad
OUT = args.out_dir

FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
QC_DIR = OUT / "qc"


# ============================================================
# RNA MARKER PANEL
#
# Rich/common T-cell marker panel used for visualization.
# These marker groups DO NOT assign cluster identities.
# ============================================================

RNA_MARKER_GROUPS = OrderedDict({

    "Naive / memory": [
        "CCR7",
        "TCF7",
        "LEF1",
        "MAL",
    ],

    "CD4 helper": [
        "CD4",
        "IL7R",
        "LTB",
    ],

    "Activation": [
        "CD69",
        "HLA-DRA",
        "FOS",
    ],

    "IFN response": [
        "ISG15",
        "IFIT1",
        "IFIT3",
        "MX1",
    ],

    "CD8": [
        "CD8A",
        "CD8B",
    ],

    "Cytotoxic": [
        "CCL5",
        "GZMB",
        "PRF1",
        "NKG7",
        "GNLY",
    ],

    "GZMK": [
        "GZMK",
    ],

    "Regulatory": [
        "FOXP3",
        "CTLA4",
        "IL2RA",
        "TIGIT",
        "IKZF2",
    ],
})


# ============================================================
# SURFACE-PROTEIN MARKER PANEL
#
# Exact feature names from obsm["protein"] where verified.
# Optional markers are included only if present.
# ============================================================

ADT_MARKER_GROUPS = OrderedDict({

    "T-cell core": [
        ("CD3", "Hu.CD3_UCHT1"),
    ],

    "Naive / memory": [
        ("CCR7", "Hu.CD197"),
        ("CD45RA", "Hu.CD45RA"),
    ],

    "CD4 helper": [
        ("CD4", "Hu.CD4_RPA.T4"),
        ("CD127", "Hu.CD127"),
    ],

    "CD8": [
        ("CD8", "Hu.CD8"),
    ],

    "Activation": [
        ("CD38", "Hu.CD38_HIT2"),
        ("HLA-DR", "Hu.HLA.DR"),
    ],

    "Regulatory": [
        ("CD25", "Hu.CD25"),
    ],

    "NK / NKT": [
        ("CD56", "Hu.CD56"),
    ],
})


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dirs():

    for path in [
        OUT,
        FIG_DIR,
        TABLE_DIR,
        QC_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def find_rscript():

    rscript = shutil.which(
        args.rscript
    )

    if rscript is None:
        raise RuntimeError(
            "Rscript is not available in PATH. "
            "Supply --rscript or make Rscript available on PATH."
        )

    return rscript


def extract_seurat_metadata(
    rds_path,
    out_csv,
):

    """
    Extract only native Seurat information.

    NO biological labels are imported.
    """

    rscript = find_rscript()

    r_code = f'''
    suppressPackageStartupMessages(library(Seurat))
    suppressPackageStartupMessages(library(SeuratObject))

    obj <- readRDS("{rds_path}")

    meta <- obj@meta.data
    meta$cell_id <- rownames(meta)
    meta$seurat_identity <- as.character(Idents(obj))

    if (!("seurat_clusters" %in% colnames(meta))) {{
        stop("seurat_clusters missing from Seurat metadata")
    }}

    write.csv(
        meta[, c(
            "cell_id",
            "seurat_identity",
            "seurat_clusters"
        )],
        file="{out_csv}",
        row.names=FALSE,
        quote=TRUE
    )
    '''

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".R",
        delete=False,
    ) as handle:

        handle.write(
            r_code
        )

        r_file = handle.name

    try:

        subprocess.run(
            [
                rscript,
                r_file,
            ],
            check=True,
        )

    finally:

        if os.path.exists(
            r_file
        ):
            os.remove(
                r_file
            )


def numeric_cluster_order(
    values,
):

    values = [
        str(
            value
        )
        for value in values
    ]

    return sorted(
        values,
        key=lambda x: (
            0,
            int(
                x
            ),
        )
        if x.isdigit()
        else (
            1,
            x,
        ),
    )


def flatten_groups(
    groups,
):

    markers = []

    for group_markers in groups.values():

        for marker in group_markers:

            if marker not in markers:
                markers.append(
                    marker
                )

    return markers


# ============================================================
# Z-SCORE
# ============================================================

def markerwise_cluster_zscore(
    cluster_mean,
):

    """
    Marker-wise z-score across cluster means.

    For each marker j:

        z_cj = (mean_cj - mean_j) / sd_j

    where mean_j and sd_j are calculated across Seurat clusters.

    Population standard deviation:
        ddof = 0
    """

    values = cluster_mean.to_numpy(
        dtype=float
    )

    mu = np.nanmean(
        values,
        axis=0,
        keepdims=True,
    )

    sd = np.nanstd(
        values,
        axis=0,
        ddof=0,
        keepdims=True,
    )

    bad = (
        ~np.isfinite(
            sd
        )
        |
        (
            sd == 0
        )
    )

    sd[
        bad
    ] = 1.0

    z = (
        values
        - mu
    ) / sd

    return pd.DataFrame(
        z,
        index=cluster_mean.index,
        columns=cluster_mean.columns,
    )


# ============================================================
# CLUSTER MEANS
# ============================================================

def cluster_mean_from_matrix(
    matrix,
    cluster_labels,
    columns,
):

    """
    Compute cluster means with a sparse membership matrix.

    Rows of matrix must correspond exactly to cluster_labels.
    """

    cluster_labels = np.asarray(
        cluster_labels,
        dtype=str,
    )

    clusters = numeric_cluster_order(
        np.unique(
            cluster_labels
        )
    )

    cluster_lookup = {
        cluster: i
        for i, cluster in enumerate(
            clusters
        )
    }

    codes = np.array(
        [
            cluster_lookup[
                cluster
            ]
            for cluster in cluster_labels
        ],
        dtype=int,
    )

    n_cells = len(
        cluster_labels
    )

    n_clusters = len(
        clusters
    )

    membership = sp.csr_matrix(
        (
            np.ones(
                n_cells,
                dtype=float,
            ),
            (
                np.arange(
                    n_cells
                ),
                codes,
            ),
        ),
        shape=(
            n_cells,
            n_clusters,
        ),
    )

    if sp.issparse(
        matrix
    ):

        sums = (
            membership.T
            @ matrix
        )

        if sp.issparse(
            sums
        ):
            sums = sums.toarray()
        else:
            sums = np.asarray(
                sums
            )

    else:

        sums = np.asarray(
            membership.T
            @ np.asarray(
                matrix
            )
        )

    counts = np.bincount(
        codes,
        minlength=n_clusters,
    ).astype(
        float
    )

    means = (
        sums
        / counts[
            :,
            None
        ]
    )

    return pd.DataFrame(
        means,
        index=clusters,
        columns=columns,
    )


# ============================================================
# RNA COUNT-SOURCE DETECTION
# ============================================================

def looks_like_raw_counts(
    matrix,
    max_test_values=200000,
):

    """
    Conservative test for raw count-like data.

    Raw counts should be:
        - nonnegative
        - almost entirely integer-valued
    """

    if sp.issparse(
        matrix
    ):
        values = matrix.data
    else:
        values = np.asarray(
            matrix
        ).reshape(
            -1
        )

    if values.size == 0:
        return False

    if values.size > max_test_values:

        rng = np.random.default_rng(
            42
        )

        idx = rng.choice(
            values.size,
            size=max_test_values,
            replace=False,
        )

        values = values[
            idx
        ]

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return False

    if np.nanmin(
        values
    ) < 0:
        return False

    integer_fraction = np.mean(
        np.isclose(
            values,
            np.round(
                values
            ),
            atol=1e-6,
        )
    )

    return (
        integer_fraction
        >= 0.995
    )


def choose_rna_count_source(
    adata,
):

    """
    Prefer an explicit count layer.

    Do not silently treat normalized expression as raw counts.
    """

    preferred_layers = [
        "counts",
        "raw_counts",
        "count",
    ]

    for layer in preferred_layers:

        if layer in adata.layers:

            matrix = adata.layers[
                layer
            ]

            if looks_like_raw_counts(
                matrix
            ):

                return (
                    "layer",
                    layer,
                )

    if looks_like_raw_counts(
        adata.X
    ):

        return (
            "X",
            None,
        )

    if (
        adata.raw
        is not None
        and looks_like_raw_counts(
            adata.raw.X
        )
    ):

        return (
            "raw",
            None,
        )

    raise RuntimeError(
        "Could not identify a trustworthy raw RNA-count source. "
        "Checked counts/raw_counts/count layers, X, and adata.raw. "
        "Stopping rather than renormalizing already-normalized data."
    )


def get_full_rna_counts(
    adata,
    cells,
    source_type,
    source_name,
):

    if source_type == "raw":

        return adata.raw[
            cells,
            :
        ].X

    sub = adata[
        cells,
        :
    ]

    if source_type == "layer":

        return sub.layers[
            source_name
        ]

    return sub.X


def get_marker_rna_counts(
    adata,
    cells,
    genes,
    source_type,
    source_name,
):

    if source_type == "raw":

        var_names = adata.raw.var_names

        available = [
            gene
            for gene in genes
            if gene in var_names
        ]

        matrix = adata.raw[
            cells,
            available
        ].X

    else:

        var_names = adata.var_names

        available = [
            gene
            for gene in genes
            if gene in var_names
        ]

        sub = adata[
            cells,
            available
        ]

        if source_type == "layer":

            matrix = sub.layers[
                source_name
            ]

        else:

            matrix = sub.X

    if len(
        available
    ) == 0:

        raise RuntimeError(
            "None of the requested RNA markers were found."
        )

    return (
        matrix,
        available,
    )


def normalize_total_log1p(
    marker_counts,
    full_counts,
):

    """
    Normalize each cell to total 10,000 using the full RNA
    library size, then log1p the selected marker matrix.
    """

    if sp.issparse(
        full_counts
    ):

        totals = np.asarray(
            full_counts.sum(
                axis=1
            )
        ).reshape(
            -1
        )

    else:

        totals = np.asarray(
            full_counts
        ).sum(
            axis=1
        )

    totals = totals.astype(
        float
    )

    factors = np.zeros_like(
        totals,
        dtype=float,
    )

    positive = (
        totals > 0
    )

    factors[
        positive
    ] = (
        RNA_TARGET_SUM
        / totals[
            positive
        ]
    )

    if sp.issparse(
        marker_counts
    ):

        normalized = marker_counts.multiply(
            factors[
                :,
                None
            ]
        ).tocsr(
            copy=True
        )

        normalized.data = np.log1p(
            normalized.data
        )

    else:

        normalized = (
            np.asarray(
                marker_counts,
                dtype=float,
            )
            * factors[
                :,
                None
            ]
        )

        normalized = np.log1p(
            normalized
        )

    return normalized


# ============================================================
# PROTEIN
# ============================================================

def get_protein_dataframe(
    adata,
):

    if "protein" not in adata.obsm:

        raise KeyError(
            "Expected obsm['protein'] in T_cells_with_protein.h5ad."
        )

    protein = adata.obsm[
        "protein"
    ]

    if not isinstance(
        protein,
        pd.DataFrame,
    ):

        raise TypeError(
            "Expected obsm['protein'] to be a pandas DataFrame "
            "with named protein features."
        )

    out = protein.copy()

    out.index = (
        adata.obs_names
        .astype(str)
    )

    return out


def resolve_adt_panel(
    protein_columns,
):

    resolved = OrderedDict()

    records = []

    for group_name, entries in ADT_MARKER_GROUPS.items():

        current = []

        for display_name, exact_feature in entries:

            available = (
                exact_feature
                in protein_columns
            )

            records.append(
                {
                    "marker_group":
                        group_name,

                    "display_marker":
                        display_name,

                    "exact_feature":
                        exact_feature,

                    "available":
                        bool(
                            available
                        ),
                }
            )

            if available:

                current.append(
                    (
                        display_name,
                        exact_feature,
                    )
                )

            else:

                print(
                    "[WARN] Optional ADT absent:",
                    display_name,
                    exact_feature,
                    flush=True,
                )

        if current:

            resolved[
                group_name
            ] = current

    return (
        resolved,
        pd.DataFrame(
            records
        ),
    )


def protein_clr_per_marker(
    protein_df,
):

    """
    Per-marker CLR across cells.

    For protein marker j:

        g_j = exp(mean_i(log(1 + x_ij)))

        CLR_ij = log(1 + x_ij / g_j)

    Zero-count cells remain part of N.
    """

    X = protein_df.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if np.nanmin(
        X
    ) < 0:

        raise RuntimeError(
            "Protein matrix contains negative values before CLR."
        )

    log1p_X = np.log1p(
        X
    )

    g = np.exp(
        np.nanmean(
            log1p_X,
            axis=0,
        )
    )

    bad = (
        ~np.isfinite(
            g
        )
        |
        (
            g <= 0
        )
    )

    g[
        bad
    ] = 1.0

    clr = np.log1p(
        X
        / g[
            None,
            :
        ]
    )

    return pd.DataFrame(
        clr,
        index=protein_df.index,
        columns=protein_df.columns,
    )


# ============================================================
# PLOT HELPERS
# ============================================================

def marker_group_spans(
    groups,
    columns,
):

    lookup = {
        marker: i
        for i, marker in enumerate(
            columns
        )
    }

    spans = OrderedDict()

    for group_name, markers in groups.items():

        idx = [
            lookup[
                marker
            ]
            for marker in markers
            if marker in lookup
        ]

        if idx:

            spans[
                group_name
            ] = (
                min(
                    idx
                ),
                max(
                    idx
                ),
            )

    return spans


def draw_marker_headers(
    ax,
    groups,
    columns,
):

    spans = marker_group_spans(
        groups,
        columns,
    )

    first = True

    for group_name, (
        start,
        end,
    ) in spans.items():

        if not first:

            ax.axvline(
                start - 0.5,
                color="0.60",
                linewidth=0.8,
            )

        first = False

        ax.text(
            (
                start + end
            ) / 2,
            1.018,
            group_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )


def plot_cluster_profiles(
    rna_z,
    adt_z,
    rna_groups,
    adt_groups,
    out_png,
    out_pdf,
):

    fig = plt.figure(
        figsize=(
            25,
            12,
        )
    )

    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[
            max(
                5,
                rna_z.shape[
                    1
                ] * 0.32
            ),
            max(
                3,
                adt_z.shape[
                    1
                ] * 0.45
            ),
        ],
        wspace=0.18,
    )

    ax_rna = fig.add_subplot(
        gs[
            0,
            0
        ]
    )

    ax_adt = fig.add_subplot(
        gs[
            0,
            1
        ]
    )

    # --------------------------------------------------------
    # RNA
    # --------------------------------------------------------

    im_rna = ax_rna.imshow(
        rna_z.to_numpy(
            dtype=float
        ),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=
            DISPLAY_VMIN,
        vmax=
            DISPLAY_VMAX,
    )

    ax_rna.set_xticks(
        np.arange(
            rna_z.shape[
                1
            ]
        )
    )

    ax_rna.set_xticklabels(
        rna_z.columns,
        rotation=90,
        fontsize=8,
    )

    ax_rna.set_yticks(
        np.arange(
            rna_z.shape[
                0
            ]
        )
    )

    ax_rna.set_yticklabels(
        [
            f"Cluster {cluster}"
            for cluster in rna_z.index
        ],
        fontsize=9,
    )

    draw_marker_headers(
        ax_rna,
        rna_groups,
        list(
            rna_z.columns
        ),
    )

    ax_rna.set_title(
        "RNA marker expression — cluster-mean z-scores",
        fontsize=13,
        pad=30,
    )

    ax_rna.set_xlabel(
        "RNA markers",
        fontsize=10,
    )

    ax_rna.set_ylabel(
        "Seurat cluster",
        fontsize=10,
    )

    cbar_rna = fig.colorbar(
        im_rna,
        ax=ax_rna,
        fraction=0.022,
        pad=0.012,
    )

    cbar_rna.set_label(
        "Cluster-mean RNA z-score",
        fontsize=9,
    )

    # --------------------------------------------------------
    # ADT
    # --------------------------------------------------------

    im_adt = ax_adt.imshow(
        adt_z.to_numpy(
            dtype=float
        ),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=
            DISPLAY_VMIN,
        vmax=
            DISPLAY_VMAX,
    )

    ax_adt.set_xticks(
        np.arange(
            adt_z.shape[
                1
            ]
        )
    )

    ax_adt.set_xticklabels(
        adt_z.columns,
        rotation=90,
        fontsize=8,
    )

    ax_adt.set_yticks(
        np.arange(
            adt_z.shape[
                0
            ]
        )
    )

    ax_adt.set_yticklabels(
        [
            f"Cluster {cluster}"
            for cluster in adt_z.index
        ],
        fontsize=9,
    )

    draw_marker_headers(
        ax_adt,
        adt_groups,
        list(
            adt_z.columns
        ),
    )

    ax_adt.set_title(
        "Surface-protein expression — CLR cluster-mean z-scores",
        fontsize=13,
        pad=30,
    )

    ax_adt.set_xlabel(
        "Surface-protein markers",
        fontsize=10,
    )

    ax_adt.set_ylabel(
        "Seurat cluster",
        fontsize=10,
    )

    cbar_adt = fig.colorbar(
        im_adt,
        ax=ax_adt,
        fraction=0.034,
        pad=0.018,
    )

    cbar_adt.set_label(
        "Cluster-mean CLR z-score",
        fontsize=9,
    )

    # --------------------------------------------------------
    # Overall title / methods note
    # --------------------------------------------------------

    fig.suptitle(
        "T-Cell Seurat Cluster Marker Profiles",
        fontsize=17,
        fontweight="bold",
        y=0.992,
    )

    fig.text(
        0.50,
        0.018,
        (
            "Rows are native Seurat clusters only. "
            "RNA: per-cell total normalization to 10,000, log1p, "
            "cluster means, then marker-wise z-scoring across clusters. "
            "Surface protein: per-marker CLR, cluster means, then "
            "marker-wise z-scoring across clusters. "
            "No manual cell-type labels or phenotype assignments are used."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.15,
        left=0.08,
        right=0.97,
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    print(
        "[START] Seurat-only T-cell cluster marker profiles.",
        flush=True,
    )

    for path in [
        SEURAT_RDS,
        T_H5AD,
    ]:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    # --------------------------------------------------------
    # 01. EXTRACT NATIVE SEURAT CLUSTERS
    # --------------------------------------------------------

    seurat_csv = (
        TABLE_DIR
        / "Tcells_native_Seurat_metadata.csv"
    )

    extract_seurat_metadata(
        rds_path=
            SEURAT_RDS,
        out_csv=
            seurat_csv,
    )

    seurat = pd.read_csv(
        seurat_csv,
        dtype=str,
    )

    if seurat[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Seurat cell IDs detected."
        )

    seurat = (
        seurat
        .set_index(
            "cell_id"
        )
    )

    seurat[
        "seurat_cluster"
    ] = (
        seurat[
            "seurat_clusters"
        ]
        .astype(str)
    )

    native_clusters = numeric_cluster_order(
        seurat[
            "seurat_cluster"
        ].unique()
    )

    print(
        "[01] Native Seurat clusters:",
        native_clusters,
        flush=True,
    )

    print(
        "[01] Number of clusters:",
        len(
            native_clusters
        ),
        flush=True,
    )

    # --------------------------------------------------------
    # 02. LOAD T-CELL RNA + PROTEIN OBJECT
    # --------------------------------------------------------

    adata = sc.read_h5ad(
        T_H5AD,
        backed="r",
    )

    protein_df = get_protein_dataframe(
        adata
    )

    shared_cells = (
        seurat.index
        .intersection(
            adata.obs_names.astype(str)
        )
        .intersection(
            protein_df.index
        )
    )

    if len(
        shared_cells
    ) == 0:

        raise RuntimeError(
            "No shared cells between Seurat RDS "
            "and T_cells_with_protein.h5ad."
        )

    metadata = seurat.loc[
        shared_cells
    ].copy()

    print(
        "[02] Shared Seurat/RNA/ADT cells:",
        f"{len(metadata):,}",
        flush=True,
    )

    # --------------------------------------------------------
    # 03. CLUSTER COUNTS
    # --------------------------------------------------------

    cluster_counts = (
        metadata[
            "seurat_cluster"
        ]
        .value_counts()
        .reindex(
            numeric_cluster_order(
                metadata[
                    "seurat_cluster"
                ].unique()
            )
        )
        .rename_axis(
            "seurat_cluster"
        )
        .reset_index(
            name="n_cells"
        )
    )

    cluster_counts.to_csv(
        TABLE_DIR
        / "Seurat_cluster_cell_counts.csv",
        index=False,
    )

    # ========================================================
    # 04. RNA
    # ========================================================

    print(
        "[04] Determining RNA raw-count source...",
        flush=True,
    )

    (
        rna_source_type,
        rna_source_name,
    ) = choose_rna_count_source(
        adata
    )

    print(
        "[04] RNA source:",
        rna_source_type,
        rna_source_name,
        flush=True,
    )

    requested_rna = flatten_groups(
        RNA_MARKER_GROUPS
    )

    full_counts = get_full_rna_counts(
        adata=
            adata,
        cells=
            shared_cells.tolist(),
        source_type=
            rna_source_type,
        source_name=
            rna_source_name,
    )

    (
        marker_counts,
        available_rna,
    ) = get_marker_rna_counts(
        adata=
            adata,
        cells=
            shared_cells.tolist(),
        genes=
            requested_rna,
        source_type=
            rna_source_type,
        source_name=
            rna_source_name,
    )

    rna_resolution = pd.DataFrame(
        [
            {
                "marker":
                    marker,

                "available":
                    marker in available_rna,
            }
            for marker in requested_rna
        ]
    )

    rna_resolution.to_csv(
        TABLE_DIR
        / "RNA_marker_resolution.csv",
        index=False,
    )

    normalized_rna = normalize_total_log1p(
        marker_counts=
            marker_counts,
        full_counts=
            full_counts,
    )

    rna_mean = cluster_mean_from_matrix(
        matrix=
            normalized_rna,
        cluster_labels=
            metadata[
                "seurat_cluster"
            ].to_numpy(),
        columns=
            available_rna,
    )

    rna_z = markerwise_cluster_zscore(
        rna_mean
    )

    # Preserve requested biological marker-group order.
    rna_groups_present = OrderedDict()

    for group_name, markers in RNA_MARKER_GROUPS.items():

        present = [
            marker
            for marker in markers
            if marker in available_rna
        ]

        if present:

            rna_groups_present[
                group_name
            ] = present

    rna_order = flatten_groups(
        rna_groups_present
    )

    rna_mean = rna_mean[
        rna_order
    ]

    rna_z = rna_z[
        rna_order
    ]

    rna_mean.to_csv(
        TABLE_DIR
        / "RNA_cluster_means_log1p_norm1e4.csv"
    )

    rna_z.to_csv(
        TABLE_DIR
        / "RNA_cluster_mean_zscores.csv"
    )

    # ========================================================
    # 05. SURFACE PROTEIN
    # ========================================================

    print(
        "[05] Resolving ADT markers...",
        flush=True,
    )

    (
        resolved_adt_groups,
        adt_resolution,
    ) = resolve_adt_panel(
        protein_df.columns
    )

    adt_resolution.to_csv(
        TABLE_DIR
        / "ADT_marker_resolution.csv",
        index=False,
    )

    if len(
        resolved_adt_groups
    ) == 0:

        raise RuntimeError(
            "No requested ADT markers were available."
        )

    # Per-marker CLR across all matched T cells.
    protein_shared = protein_df.loc[
        shared_cells
    ]

    protein_clr = protein_clr_per_marker(
        protein_shared
    )

    display_to_exact = {}

    adt_groups_present = OrderedDict()

    for group_name, entries in resolved_adt_groups.items():

        current = []

        for display_name, exact_feature in entries:

            display_to_exact[
                display_name
            ] = exact_feature

            current.append(
                display_name
            )

        adt_groups_present[
            group_name
        ] = current

    adt_selected = pd.DataFrame(
        {
            display_name:
                protein_clr[
                    exact_feature
                ].to_numpy()
            for display_name, exact_feature
            in display_to_exact.items()
        },
        index=protein_clr.index,
    )

    adt_mean = cluster_mean_from_matrix(
        matrix=
            adt_selected.to_numpy(
                dtype=float
            ),
        cluster_labels=
            metadata[
                "seurat_cluster"
            ].to_numpy(),
        columns=
            list(
                adt_selected.columns
            ),
    )

    adt_z = markerwise_cluster_zscore(
        adt_mean
    )

    adt_order = flatten_groups(
        adt_groups_present
    )

    adt_mean = adt_mean[
        adt_order
    ]

    adt_z = adt_z[
        adt_order
    ]

    adt_mean.to_csv(
        TABLE_DIR
        / "ADT_cluster_means_per_marker_CLR.csv"
    )

    adt_z.to_csv(
        TABLE_DIR
        / "ADT_cluster_mean_zscores.csv"
    )

    # ========================================================
    # 06. VERIFY IDENTICAL ROW SET / ORDER
    # ========================================================

    common_clusters = [
        cluster
        for cluster in numeric_cluster_order(
            rna_z.index
        )
        if cluster in adt_z.index
    ]

    rna_z = rna_z.loc[
        common_clusters
    ]

    adt_z = adt_z.loc[
        common_clusters
    ]

    rna_mean = rna_mean.loc[
        common_clusters
    ]

    adt_mean = adt_mean.loc[
        common_clusters
    ]

    print(
        "[06] Clusters plotted:",
        common_clusters,
        flush=True,
    )

    # ========================================================
    # 07. PLOT
    # ========================================================

    out_png = (
        FIG_DIR
        / "Tcells_Seurat_cluster_marker_profiles.png"
    )

    out_pdf = (
        FIG_DIR
        / "Tcells_Seurat_cluster_marker_profiles.pdf"
    )

    plot_cluster_profiles(
        rna_z=
            rna_z,
        adt_z=
            adt_z,
        rna_groups=
            rna_groups_present,
        adt_groups=
            adt_groups_present,
        out_png=
            out_png,
        out_pdf=
            out_pdf,
    )

    # ========================================================
    # 08. QC
    # ========================================================

    qc = {

        "scope":
            "T cells only",

        "clustering_source":
            "native Seurat clusters",

        "manual_cell_type_used":
            False,

        "manual_cell_type_display_used":
            False,

        "Scanpy_annotation_labels_used":
            False,

        "overlap_or_Hungarian_labels_used":
            False,

        "phenotype_names_assigned":
            False,

        "reference_direction_sign_flips_used":
            False,

        "Seurat_RDS":
            str(
                SEURAT_RDS
            ),

        "RNA_ADT_H5AD":
            str(
                T_H5AD
            ),

        "n_shared_cells":
            int(
                len(
                    metadata
                )
            ),

        "n_clusters_plotted":
            int(
                len(
                    common_clusters
                )
            ),

        "clusters_plotted":
            common_clusters,

        "RNA_processing":
            {
                "count_source_type":
                    rna_source_type,

                "count_source_name":
                    rna_source_name,

                "target_sum":
                    RNA_TARGET_SUM,

                "transformation":
                    "log1p",

                "summary":
                    "cluster mean",

                "scaling":
                    "marker-wise z-score across Seurat clusters",

                "zscore_ddof":
                    0,
            },

        "ADT_processing":
            {
                "normalization":
                    "per-marker CLR across cells",

                "summary":
                    "cluster mean",

                "scaling":
                    "marker-wise z-score across Seurat clusters",

                "zscore_ddof":
                    0,
            },

        "RNA_markers":
            list(
                rna_z.columns
            ),

        "ADT_markers":
            list(
                adt_z.columns
            ),

        "figure_png":
            str(
                out_png
            ),

        "figure_pdf":
            str(
                out_pdf
            ),

        "input_files_modified":
            False,
    }

    with open(
        QC_DIR
        / "Tcells_Seurat_cluster_marker_profiles_qc.json",
        "w",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    try:

        adata.file.close()

    except Exception:

        pass

    print(
        "\n===== QC =====",
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
        "\n[SUCCESS] Figure:",
        out_png,
        flush=True,
    )


if __name__ == "__main__":

    try:

        main()

    except Exception:

        print(
            "\n[ERROR] Seurat cluster marker-profile analysis failed.",
            flush=True,
        )

        traceback.print_exc()

        sys.exit(1)
