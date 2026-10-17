#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B-CELL SEURAT CLUSTER MARKER PROFILES
=====================================

Native Seurat clusters only.

NO:
    - manual cell-type annotations
    - Scanpy biological labels
    - overlap/Hungarian labels
    - phenotype assignments

RNA:
    raw counts
    -> normalize each cell to 10,000
    -> log1p
    -> cluster mean
    -> marker-wise z-score across native Seurat clusters (ddof=0)

ADT:
    raw ADT
    -> per-marker CLR across cells
    -> cluster mean
    -> marker-wise z-score across native Seurat clusters (ddof=0)
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
# CONFIG
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
B_H5AD = args.h5ad
OUT = args.out_dir

FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
QC_DIR = OUT / "qc"


# ============================================================
# B-CELL RNA MARKER PANEL
#
# Same marker panel used in the earlier B-cell validation work.
# These are visualization groups only; they do NOT label clusters.
# ============================================================

RNA_MARKER_GROUPS = OrderedDict({

    "Naive": [
        "IGHD",
        "IGHM",
        "TCL1A",
        "FCER2",
    ],

    "Memory": [
        "CD27",
        "AIM2",
        "CD82",
    ],

    "Activated": [
        "CD86",
        "HLA-DRA",
    ],

    "Antigen presentation": [
        "CD74",
        "HLA-DPA1",
    ],

    "IGLL5": [
        "IGLL5",
    ],

    "IGHV3": [
        "IGHV3-23",
    ],

    "Light chain": [
        "IGKC",
        "IGLC2",
    ],

    "Proliferation": [
        "MKI67",
        "TOP2A",
    ],

    "Plasma / ASC": [
        "MZB1",
        "JCHAIN",
        "XBP1",
        "PRDM1",
        "SDC1",
    ],
})


# ============================================================
# B-CELL ADT PANEL
#
# Exact protein names previously used in B-cell validation.
# Missing optional markers are reported and omitted.
# ============================================================

ADT_MARKER_GROUPS = OrderedDict({

    "B-cell core": [
        ("CD19", "Hu.CD19"),
        ("CD20", "Hu.CD20_2H7"),
        ("HLA-DR", "Hu.HLA.DR"),
    ],

    "Naive": [
        ("IgD", "Hu.IgD"),
        ("IgM", "Hu.IgM"),
    ],

    "Memory": [
        ("CD27", "Hu.CD27"),
        ("CD21", "Hu.CD21"),
    ],

    "Activated / ASC": [
        ("CD38", "Hu.CD38_HIT2"),
    ],

    "Plasma": [
        ("CD138", "Hu.CD138"),
    ],
})


# ============================================================
# HELPERS
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
            "Rscript not found. Supply --rscript or make it available on PATH."
        )

    return rscript


def extract_seurat_metadata(
    rds_path,
    out_csv,
):

    """
    Read ONLY native Seurat metadata.
    No biological annotation mapping is imported.
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
        stop("seurat_clusters missing")
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

    return sorted(
        [
            str(
                x
            )
            for x in values
        ],
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

    for group in groups.values():

        for marker in group:

            if marker not in markers:

                markers.append(
                    marker
                )

    return markers


# ============================================================
# CLUSTER MEANS + Z-SCORES
# ============================================================

def cluster_mean_from_matrix(
    matrix,
    cluster_labels,
    columns,
):

    cluster_labels = np.asarray(
        cluster_labels,
        dtype=str,
    )

    clusters = numeric_cluster_order(
        np.unique(
            cluster_labels
        )
    )

    lookup = {
        cluster: i
        for i, cluster in enumerate(
            clusters
        )
    }

    codes = np.array(
        [
            lookup[
                cluster
            ]
            for cluster in cluster_labels
        ],
        dtype=int,
    )

    n_cells = len(
        cluster_labels
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
            len(
                clusters
            ),
        ),
    )

    if sp.issparse(
        matrix
    ):

        sums = (
            membership.T
            @ matrix
        )

        sums = (
            sums.toarray()
            if sp.issparse(
                sums
            )
            else np.asarray(
                sums
            )
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
        minlength=len(
            clusters
        ),
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


def markerwise_cluster_zscore(
    cluster_mean,
):

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

    sd[
        (
            ~np.isfinite(
                sd
            )
        )
        |
        (
            sd == 0
        )
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
# RNA COUNT SOURCE
# ============================================================

def looks_like_raw_counts(
    matrix,
    max_test_values=200000,
):

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

    for layer in [
        "counts",
        "raw_counts",
        "count",
    ]:

        if (
            layer in adata.layers
            and looks_like_raw_counts(
                adata.layers[
                    layer
                ]
            )
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
        "Could not identify trustworthy raw B-cell RNA counts. "
        "Stopping rather than treating normalized data as counts."
    )


def get_full_counts(
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


def get_marker_counts(
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

    if not available:

        raise RuntimeError(
            "None of the requested B-cell RNA markers are present."
        )

    return (
        matrix,
        available,
    )


def normalize_total_log1p(
    marker_counts,
    full_counts,
):

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
        totals
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
            "obsm['protein'] missing from B_cells_with_protein.h5ad."
        )

    protein = adata.obsm[
        "protein"
    ]

    if not isinstance(
        protein,
        pd.DataFrame,
    ):

        raise TypeError(
            "Expected obsm['protein'] to contain named columns."
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

        present = []

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

                present.append(
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

        if present:

            resolved[
                group_name
            ] = present

    return (
        resolved,
        pd.DataFrame(
            records
        ),
    )


def protein_clr_per_marker(
    protein_df,
):

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

    g = np.exp(
        np.nanmean(
            np.log1p(
                X
            ),
            axis=0,
        )
    )

    g[
        (
            ~np.isfinite(
                g
            )
        )
        |
        (
            g <= 0
        )
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
# PLOT
# ============================================================

def group_spans(
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

    for name, markers in groups.items():

        idx = [
            lookup[
                marker
            ]
            for marker in markers
            if marker in lookup
        ]

        if idx:

            spans[
                name
            ] = (
                min(
                    idx
                ),
                max(
                    idx
                ),
            )

    return spans


def draw_headers(
    ax,
    groups,
    columns,
):

    spans = group_spans(
        groups,
        columns,
    )

    first = True

    for name, (
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
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )


def plot_profiles(
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
            11,
        )
    )

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[
            max(
                5,
                rna_z.shape[
                    1
                ] * 0.34
            ),
            max(
                3,
                adt_z.shape[
                    1
                ] * 0.48
            ),
        ],
        wspace=0.18,
    )

    ax1 = fig.add_subplot(
        gs[
            0,
            0
        ]
    )

    ax2 = fig.add_subplot(
        gs[
            0,
            1
        ]
    )

    # RNA
    im1 = ax1.imshow(
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

    ax1.set_xticks(
        np.arange(
            rna_z.shape[
                1
            ]
        )
    )

    ax1.set_xticklabels(
        rna_z.columns,
        rotation=90,
        fontsize=8,
    )

    ax1.set_yticks(
        np.arange(
            len(
                rna_z.index
            )
        )
    )

    ax1.set_yticklabels(
        [
            f"Cluster {x}"
            for x in rna_z.index
        ],
        fontsize=9,
    )

    draw_headers(
        ax1,
        rna_groups,
        list(
            rna_z.columns
        ),
    )

    ax1.set_title(
        "RNA marker expression — cluster-mean z-scores",
        fontsize=13,
        pad=30,
    )

    ax1.set_xlabel(
        "RNA markers"
    )

    ax1.set_ylabel(
        "Seurat cluster"
    )

    cb1 = fig.colorbar(
        im1,
        ax=ax1,
        fraction=0.022,
        pad=0.012,
    )

    cb1.set_label(
        "Cluster-mean RNA z-score"
    )

    # ADT
    im2 = ax2.imshow(
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

    ax2.set_xticks(
        np.arange(
            adt_z.shape[
                1
            ]
        )
    )

    ax2.set_xticklabels(
        adt_z.columns,
        rotation=90,
        fontsize=8,
    )

    ax2.set_yticks(
        np.arange(
            len(
                adt_z.index
            )
        )
    )

    ax2.set_yticklabels(
        [
            f"Cluster {x}"
            for x in adt_z.index
        ],
        fontsize=9,
    )

    draw_headers(
        ax2,
        adt_groups,
        list(
            adt_z.columns
        ),
    )

    ax2.set_title(
        "Surface-protein expression — CLR cluster-mean z-scores",
        fontsize=13,
        pad=30,
    )

    ax2.set_xlabel(
        "Surface-protein markers"
    )

    ax2.set_ylabel(
        "Seurat cluster"
    )

    cb2 = fig.colorbar(
        im2,
        ax=ax2,
        fraction=0.035,
        pad=0.018,
    )

    cb2.set_label(
        "Cluster-mean CLR z-score"
    )

    fig.suptitle(
        "B-Cell Seurat Cluster Marker Profiles",
        fontsize=17,
        fontweight="bold",
        y=0.992,
    )

    fig.text(
        0.50,
        0.018,
        (
            "Rows are native Seurat clusters only. "
            "RNA: per-cell normalization to 10,000, log1p, cluster means, "
            "then marker-wise z-scoring across clusters. "
            "Surface protein: per-marker CLR, cluster means, then "
            "marker-wise z-scoring across clusters. "
            "No biological annotation labels are used."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.16,
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
        "[START] Seurat-only B-cell cluster marker profiles.",
        flush=True,
    )

    for path in [
        SEURAT_RDS,
        B_H5AD,
    ]:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    # Native Seurat metadata
    seurat_csv = (
        TABLE_DIR
        / "Bcells_native_Seurat_metadata.csv"
    )

    extract_seurat_metadata(
        SEURAT_RDS,
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
            "Duplicate Seurat cell IDs."
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

    # Load B RNA/ADT object
    adata = sc.read_h5ad(
        B_H5AD,
        backed="r",
    )

    protein_df = get_protein_dataframe(
        adata
    )

    shared = (
        seurat.index
        .intersection(
            adata.obs_names.astype(str)
        )
        .intersection(
            protein_df.index
        )
    )

    if len(
        shared
    ) == 0:

        raise RuntimeError(
            "No shared Seurat/RNA/ADT B cells."
        )

    metadata = (
        seurat.loc[
            shared
        ]
        .copy()
    )

    print(
        "[MATCH] Shared B cells:",
        f"{len(metadata):,}",
        flush=True,
    )

    # Cell counts
    counts = (
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

    counts.to_csv(
        TABLE_DIR
        / "Seurat_cluster_cell_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # RNA
    # --------------------------------------------------------

    (
        source_type,
        source_name,
    ) = choose_rna_count_source(
        adata
    )

    print(
        "[RNA] count source:",
        source_type,
        source_name,
        flush=True,
    )

    requested_rna = flatten_groups(
        RNA_MARKER_GROUPS
    )

    full_counts = get_full_counts(
        adata,
        shared.tolist(),
        source_type,
        source_name,
    )

    (
        marker_counts,
        available_rna,
    ) = get_marker_counts(
        adata,
        shared.tolist(),
        requested_rna,
        source_type,
        source_name,
    )

    normalized = normalize_total_log1p(
        marker_counts,
        full_counts,
    )

    rna_mean = cluster_mean_from_matrix(
        normalized,
        metadata[
            "seurat_cluster"
        ].to_numpy(),
        available_rna,
    )

    rna_z = markerwise_cluster_zscore(
        rna_mean
    )

    rna_groups_present = OrderedDict()

    for group_name, markers in RNA_MARKER_GROUPS.items():

        present = [
            x
            for x in markers
            if x in available_rna
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

    # --------------------------------------------------------
    # ADT
    # --------------------------------------------------------

    (
        resolved_adt,
        adt_resolution,
    ) = resolve_adt_panel(
        protein_df.columns
    )

    adt_resolution.to_csv(
        TABLE_DIR
        / "ADT_marker_resolution.csv",
        index=False,
    )

    protein_shared = protein_df.loc[
        shared
    ]

    clr = protein_clr_per_marker(
        protein_shared
    )

    display_to_exact = {}

    adt_groups_present = OrderedDict()

    for group_name, entries in resolved_adt.items():

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
                clr[
                    exact_feature
                ].to_numpy()
            for display_name, exact_feature
            in display_to_exact.items()
        },
        index=clr.index,
    )

    adt_mean = cluster_mean_from_matrix(
        adt_selected.to_numpy(
            dtype=float
        ),
        metadata[
            "seurat_cluster"
        ].to_numpy(),
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

    # Same row set / order
    clusters = [
        cluster
        for cluster in numeric_cluster_order(
            rna_z.index
        )
        if cluster in adt_z.index
    ]

    rna_z = rna_z.loc[
        clusters
    ]

    adt_z = adt_z.loc[
        clusters
    ]

    # Plot
    out_png = (
        FIG_DIR
        / "Bcells_Seurat_cluster_marker_profiles.png"
    )

    out_pdf = (
        FIG_DIR
        / "Bcells_Seurat_cluster_marker_profiles.pdf"
    )

    plot_profiles(
        rna_z,
        adt_z,
        rna_groups_present,
        adt_groups_present,
        out_png,
        out_pdf,
    )

    qc = {
        "scope":
            "B cells only",

        "clustering_source":
            "native Seurat clusters",

        "manual_labels_used":
            False,

        "Scanpy_labels_used":
            False,

        "overlap_Hungarian_labels_used":
            False,

        "phenotype_names_assigned":
            False,

        "n_shared_cells":
            int(
                len(
                    metadata
                )
            ),

        "clusters_plotted":
            clusters,

        "RNA_count_source_type":
            source_type,

        "RNA_count_source_name":
            source_name,

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
    }

    with open(
        QC_DIR
        / "Bcells_Seurat_cluster_marker_profiles_qc.json",
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
        "[SUCCESS]",
        out_png,
        flush=True,
    )


if __name__ == "__main__":

    try:

        main()

    except Exception:

        print(
            "\n[ERROR] B-cell cluster marker analysis failed.",
            flush=True,
        )

        traceback.print_exc()

        sys.exit(1)
