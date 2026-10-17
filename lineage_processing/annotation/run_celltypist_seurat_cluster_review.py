#!/usr/bin/env python3

from pathlib import Path
import json
import subprocess
import tempfile
import traceback
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import celltypist


# ============================================================
# PATHS
# ============================================================

import argparse

parser = argparse.ArgumentParser(description="Review CellTypist predictions against native Seurat clusters.")
parser.add_argument("--overall-h5ad", type=Path, required=True, help="Overall RNA H5AD with raw counts and matching cell IDs.")
parser.add_argument("--t-seurat-rds", type=Path, required=True)
parser.add_argument("--b-seurat-rds", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--model", default="Immune_All_Low.pkl", help="Installed CellTypist model name or model file.")
parser.add_argument("--rscript", default="Rscript", help="Rscript executable on PATH or explicit executable path.")
args = parser.parse_args()
OVERALL_H5AD = args.overall_h5ad
RSCRIPT = args.rscript
OUT_ROOT = args.out_dir
MODEL_NAME = args.model
CONFIGS = [
    {"lineage_name": "T cells", "short_name": "Tcells", "rds": args.t_seurat_rds},
    {"lineage_name": "B cells", "short_name": "Bcells", "rds": args.b_seurat_rds},
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def numeric_cluster_order(values):
    values = [str(x) for x in values]

    def key(x):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    return sorted(
        values,
        key=key,
    )


def ensure_lineage_dirs(short_name):
    root = OUT_ROOT / short_name

    table_dir = root / "tables"
    fig_dir = root / "figures"
    qc_dir = root / "qc"

    for d in [
        root,
        table_dir,
        fig_dir,
        qc_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    return root, table_dir, fig_dir, qc_dir


# ============================================================
# EXTRACT AUTHORITATIVE SEURAT CLUSTERS + UMAP
# ============================================================

def extract_seurat_metadata(
    rds_path: Path,
    out_csv: Path,
):
    r_code = r'''
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(SeuratObject))

args <- commandArgs(trailingOnly=TRUE)

rds_path <- args[1]
out_csv <- args[2]

obj <- readRDS(rds_path)

if (!"seurat_clusters" %in% colnames(obj@meta.data)) {
    stop("seurat_clusters not found in Seurat metadata.")
}

reductions <- names(obj@reductions)

if ("umap.seurat" %in% reductions) {
    reduction_name <- "umap.seurat"
} else if ("umap" %in% reductions) {
    reduction_name <- "umap"
} else {
    stop(
        paste(
            "No UMAP reduction found. Available reductions:",
            paste(reductions, collapse=", ")
        )
    )
}

coords <- Embeddings(
    obj,
    reduction=reduction_name
)

cell_ids <- colnames(obj)

df <- data.frame(
    cell_id=cell_ids,
    seurat_cluster=as.character(
        obj@meta.data[cell_ids, "seurat_clusters"]
    ),
    seurat_umap_1=coords[cell_ids, 1],
    seurat_umap_2=coords[cell_ids, 2],
    stringsAsFactors=FALSE
)

write.csv(
    df,
    out_csv,
    row.names=FALSE
)

cat("[R] RDS:", rds_path, "\n")
cat("[R] UMAP reduction:", reduction_name, "\n")
cat("[R] Cells:", nrow(df), "\n")
cat(
    "[R] Clusters:",
    length(unique(df$seurat_cluster)),
    "\n"
)
'''

    result = subprocess.run(
        [
            str(RSCRIPT),
            "-e",
            r_code,
            str(rds_path),
            str(out_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(
            result.stdout,
            flush=True,
        )

    if result.stderr:
        print(
            result.stderr,
            flush=True,
        )


# ============================================================
# RNA SOURCE
# ============================================================

def looks_like_counts(x):
    """
    Inspect a small sample without densifying the full matrix.
    """
    if sp.issparse(x):
        vals = x.data
    else:
        vals = np.asarray(x).ravel()

    if vals.size == 0:
        return False

    if vals.size > 100000:
        vals = vals[:100000]

    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return False

    nonnegative = np.all(vals >= 0)

    integer_like = np.mean(
        np.isclose(
            vals,
            np.round(vals),
            atol=1e-6,
        )
    )

    return (
        nonnegative
        and integer_like > 0.98
    )


def select_count_matrix(adata):
    """
    Prefer an explicit counts layer, then raw.X, then X if it
    appears count-like.
    """

    if "counts" in adata.layers:
        print(
            "[RNA] Using layers['counts'].",
            flush=True,
        )
        return (
            adata.layers["counts"],
            np.asarray(
                adata.var_names
            ).astype(str),
            "layer",
            "counts",
        )

    if adata.raw is not None:

        if looks_like_counts(
            adata.raw.X
        ):
            print(
                "[RNA] Using adata.raw.X.",
                flush=True,
            )
            return (
                adata.raw.X,
                np.asarray(
                    adata.raw.var_names
                ).astype(str),
                "raw",
                "raw.X",
            )

    if looks_like_counts(
        adata.X
    ):
        print(
            "[RNA] Using adata.X.",
            flush=True,
        )
        return (
            adata.X,
            np.asarray(
                adata.var_names
            ).astype(str),
            "X",
            "X",
        )

    raise RuntimeError(
        "Could not identify a trustworthy raw-count matrix "
        "for CellTypist normalization."
    )


def subset_matrix_rows(
    matrix,
    row_indices,
):
    return matrix[
        row_indices,
        :
    ]


# ============================================================
# CELLTYPIST
# ============================================================

def prepare_celltypist_adata(
    overall,
    seurat_df,
):
    overall_names = pd.Index(
        overall.obs_names.astype(str)
    )

    requested = pd.Index(
        seurat_df[
            "cell_id"
        ].astype(str)
    )

    present_mask = requested.isin(
        overall_names
    )

    present_cells = requested[
        present_mask
    ]

    missing_cells = requested[
        ~present_mask
    ]

    if len(
        present_cells
    ) == 0:
        raise RuntimeError(
            "No Seurat cells were found in the original H5AD."
        )

    print(
        "[RNA] Seurat cells:",
        f"{len(requested):,}",
        flush=True,
    )

    print(
        "[RNA] Present in original H5AD:",
        f"{len(present_cells):,}",
        flush=True,
    )

    print(
        "[RNA] Missing from original H5AD:",
        f"{len(missing_cells):,}",
        flush=True,
    )

    positions = overall_names.get_indexer(
        present_cells
    )

    (
        source_matrix,
        source_genes,
        source_type,
        source_name,
    ) = select_count_matrix(
        overall
    )

    X = subset_matrix_rows(
        source_matrix,
        positions,
    )

    if sp.issparse(X):
        X = X.tocsr().copy()
    else:
        X = np.asarray(
            X
        ).copy()

    ct_adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(
            index=present_cells
        ),
        var=pd.DataFrame(
            index=source_genes
        ),
    )

    # CellTypist expects log1p expression normalized to 10,000
    # counts per cell.
    sc.pp.normalize_total(
        ct_adata,
        target_sum=1e4,
    )

    sc.pp.log1p(
        ct_adata
    )

    return (
        ct_adata,
        present_cells,
        missing_cells,
        source_type,
        source_name,
    )


def run_celltypist(
    adata,
):
    print(
        f"[CELLTYPIST] Model: {MODEL_NAME}",
        flush=True,
    )

    predictions = celltypist.annotate(
        adata,
        model=MODEL_NAME,
        majority_voting=False,
    )

    pred_df = (
        predictions
        .predicted_labels
        .copy()
    )

    if "predicted_labels" in pred_df.columns:
        labels = pred_df[
            "predicted_labels"
        ].astype(str)
    else:
        labels = pred_df.iloc[
            :,
            0
        ].astype(str)

    probability = (
        predictions
        .probability_matrix
        .copy()
    )

    probability.index = (
        probability.index.astype(str)
    )

    confidence = probability.max(
        axis=1
    )

    per_cell = pd.DataFrame(
        {
            "cell_id":
                labels.index.astype(str),

            "celltypist_label":
                labels.to_numpy(),

            "celltypist_confidence":
                confidence
                .reindex(
                    labels.index
                )
                .to_numpy(),
        }
    )

    return per_cell


# ============================================================
# CLUSTER SUMMARY
# ============================================================

def make_cluster_review_table(
    seurat_df,
    per_cell,
):
    merged = seurat_df.merge(
        per_cell,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    cluster_order = numeric_cluster_order(
        merged[
            "seurat_cluster"
        ].unique()
    )

    rows = []

    for cluster in cluster_order:

        sub = merged[
            merged[
                "seurat_cluster"
            ] == cluster
        ]

        scored = sub.dropna(
            subset=[
                "celltypist_label"
            ]
        )

        n_cells = len(
            sub
        )

        n_scored = len(
            scored
        )

        if n_scored == 0:

            majority_label = ""
            majority_fraction = np.nan
            mean_confidence = np.nan
            top3 = ""

        else:

            counts = (
                scored[
                    "celltypist_label"
                ]
                .value_counts()
            )

            majority_label = str(
                counts.index[
                    0
                ]
            )

            majority_fraction = (
                counts.iloc[
                    0
                ]
                / n_scored
            )

            majority_conf = scored.loc[
                scored[
                    "celltypist_label"
                ] == majority_label,
                "celltypist_confidence",
            ]

            mean_confidence = float(
                majority_conf.mean()
            )

            top_entries = []

            for label, count in counts.head(
                3
            ).items():

                frac = (
                    count
                    / n_scored
                )

                top_entries.append(
                    f"{label}: {frac:.3f}"
                )

            top3 = " | ".join(
                top_entries
            )

        rows.append(
            {
                "seurat_cluster":
                    cluster,

                "n_cells":
                    int(
                        n_cells
                    ),

                "n_cells_celltypist_scored":
                    int(
                        n_scored
                    ),

                "celltypist_majority_label":
                    majority_label,

                "celltypist_majority_fraction":
                    majority_fraction,

                "celltypist_mean_confidence":
                    mean_confidence,

                "celltypist_top3_labels":
                    top3,

                "RNA_evidence":
                    "",

                "ADT_evidence":
                    "",

                "provisional_assignment":
                    "",

                "final_reviewed_assignment":
                    "",

                "decision":
                    "",

                "notes":
                    "",
            }
        )

    review = pd.DataFrame(
        rows
    )

    return (
        merged,
        review,
    )


# ============================================================
# PLOTTING
# ============================================================

def make_palette(labels):
    cmap = plt.get_cmap(
        "tab20"
    )

    return {
        label:
            cmap(
                i % 20
            )
        for i, label in enumerate(
            labels
        )
    }


def add_text_with_outline(
    ax,
    x,
    y,
    text,
    fontsize=9,
):
    txt = ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        weight="bold",
        ha="center",
        va="center",
        color="black",
        zorder=20,
    )

    txt.set_path_effects(
        [
            pe.Stroke(
                linewidth=3,
                foreground="white",
            ),
            pe.Normal(),
        ]
    )


def plot_side_by_side(
    merged,
    review,
    lineage_name,
    short_name,
    fig_dir,
):
    cluster_to_label = dict(
        zip(
            review[
                "seurat_cluster"
            ].astype(str),

            review[
                "celltypist_majority_label"
            ].astype(str),
        )
    )

    merged = merged.copy()

    merged[
        "cluster_celltypist_label"
    ] = (
        merged[
            "seurat_cluster"
        ]
        .astype(str)
        .map(
            cluster_to_label
        )
    )

    merged[
        "cluster_celltypist_label"
    ] = (
        merged[
            "cluster_celltypist_label"
        ]
        .replace(
            "",
            "No CellTypist call",
        )
        .fillna(
            "No CellTypist call"
        )
    )

    cluster_order = numeric_cluster_order(
        merged[
            "seurat_cluster"
        ].unique()
    )

    label_order = list(
        dict.fromkeys(
            review[
                "celltypist_majority_label"
            ]
            .replace(
                "",
                "No CellTypist call",
            )
            .tolist()
        )
    )

    label_palette = make_palette(
        label_order
    )

    cluster_palette = make_palette(
        cluster_order
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            19,
            8.5,
        ),
        dpi=200,
    )

    ax_left = axes[
        0
    ]

    ax_right = axes[
        1
    ]

    # --------------------------------------------------------
    # LEFT: CellTypist majority biological label PER CLUSTER
    # --------------------------------------------------------

    for label in label_order:

        sub = merged[
            merged[
                "cluster_celltypist_label"
            ] == label
        ]

        ax_left.scatter(
            sub[
                "seurat_umap_1"
            ],
            sub[
                "seurat_umap_2"
            ],
            s=1.0,
            alpha=0.55,
            linewidths=0,
            color=label_palette[
                label
            ],
            rasterized=True,
        )

    # Label each Seurat cluster individually, even if two
    # clusters share the same CellTypist majority identity.
    centers = (
        merged
        .groupby(
            "seurat_cluster"
        )[
            [
                "seurat_umap_1",
                "seurat_umap_2",
            ]
        ]
        .median()
    )

    for cluster in cluster_order:

        center = centers.loc[
            cluster
        ]

        label = cluster_to_label.get(
            cluster,
            "",
        )

        if not label:
            label = "No CellTypist call"

        add_text_with_outline(
            ax_left,
            center[
                "seurat_umap_1"
            ],
            center[
                "seurat_umap_2"
            ],
            label,
            fontsize=8,
        )

    ax_left.set_title(
        f"{lineage_name}: CellTypist majority label by Seurat cluster",
        fontsize=13,
    )

    ax_left.set_xlabel(
        "Seurat UMAP 1"
    )

    ax_left.set_ylabel(
        "Seurat UMAP 2"
    )

    # --------------------------------------------------------
    # RIGHT: NATIVE SEURAT CLUSTER NUMBERS
    # --------------------------------------------------------

    for cluster in cluster_order:

        sub = merged[
            merged[
                "seurat_cluster"
            ] == cluster
        ]

        ax_right.scatter(
            sub[
                "seurat_umap_1"
            ],
            sub[
                "seurat_umap_2"
            ],
            s=1.0,
            alpha=0.55,
            linewidths=0,
            color=cluster_palette[
                cluster
            ],
            rasterized=True,
        )

    for cluster in cluster_order:

        center = centers.loc[
            cluster
        ]

        add_text_with_outline(
            ax_right,
            center[
                "seurat_umap_1"
            ],
            center[
                "seurat_umap_2"
            ],
            cluster,
            fontsize=11,
        )

    ax_right.set_title(
        f"{lineage_name}: native Seurat cluster numbers",
        fontsize=13,
    )

    ax_right.set_xlabel(
        "Seurat UMAP 1"
    )

    ax_right.set_ylabel(
        "Seurat UMAP 2"
    )

    for ax in axes:

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    fig.suptitle(
        f"{lineage_name} — CellTypist cluster annotation review",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    fig.text(
        0.50,
        0.012,
        (
            "Left: CellTypist majority prediction within each native "
            "Seurat cluster. Right: the same cells and UMAP coordinates "
            "labeled by native Seurat cluster number. "
            "CellTypist labels are provisional computational annotations "
            "for RNA/ADT and expert review."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.96,
        ]
    )

    out_png = (
        fig_dir
        / f"{short_name}_celltypist_label_vs_cluster_umap.png"
    )

    out_pdf = (
        fig_dir
        / f"{short_name}_celltypist_label_vs_cluster_umap.pdf"
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

    return (
        out_png,
        out_pdf,
    )


# ============================================================
# ONE LINEAGE
# ============================================================

def process_lineage(
    cfg,
    overall,
):
    (
        root,
        table_dir,
        fig_dir,
        qc_dir,
    ) = ensure_lineage_dirs(
        cfg[
            "short_name"
        ]
    )

    print(
        "\n"
        + "=" * 75,
        flush=True,
    )

    print(
        f"{cfg['lineage_name'].upper()}",
        flush=True,
    )

    print(
        "=" * 75,
        flush=True,
    )

    print(
        "[RDS]",
        cfg[
            "rds"
        ],
        flush=True,
    )

    if not cfg[
        "rds"
    ].is_file():

        raise FileNotFoundError(
            cfg[
                "rds"
            ]
        )

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as handle:

        seurat_temp = Path(
            handle.name
        )

    try:

        extract_seurat_metadata(
            rds_path=
                cfg[
                    "rds"
                ],
            out_csv=
                seurat_temp,
        )

        seurat_df = pd.read_csv(
            seurat_temp,
            dtype={
                "cell_id":
                    str,
                "seurat_cluster":
                    str,
            },
        )

    finally:

        seurat_temp.unlink(
            missing_ok=True
        )

    if seurat_df[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Seurat cell IDs detected."
        )

    (
        ct_adata,
        present_cells,
        missing_cells,
        source_type,
        source_name,
    ) = prepare_celltypist_adata(
        overall=
            overall,
        seurat_df=
            seurat_df,
    )

    per_cell = run_celltypist(
        ct_adata
    )

    merged, review = make_cluster_review_table(
        seurat_df=
            seurat_df,
        per_cell=
            per_cell,
    )

    per_cell_out = (
        table_dir
        / f"{cfg['short_name']}_celltypist_per_cell.csv.gz"
    )

    per_cell.to_csv(
        per_cell_out,
        index=False,
        compression="gzip",
    )

    review_out = (
        table_dir
        / f"{cfg['short_name']}_cluster_annotation_review.csv"
    )

    review.to_csv(
        review_out,
        index=False,
    )

    (
        out_png,
        out_pdf,
    ) = plot_side_by_side(
        merged=
            merged,
        review=
            review,
        lineage_name=
            cfg[
                "lineage_name"
            ],
        short_name=
            cfg[
                "short_name"
            ],
        fig_dir=
            fig_dir,
    )

    qc = {
        "lineage":
            cfg[
                "lineage_name"
            ],

        "Seurat_RDS":
            str(
                cfg[
                    "rds"
                ]
            ),

        "RNA_source_H5AD":
            str(
                OVERALL_H5AD
            ),

        "CellTypist_model":
            MODEL_NAME,

        "n_Seurat_cells":
            int(
                len(
                    seurat_df
                )
            ),

        "n_cells_found_in_H5AD":
            int(
                len(
                    present_cells
                )
            ),

        "n_cells_missing_from_H5AD":
            int(
                len(
                    missing_cells
                )
            ),

        "n_clusters":
            int(
                seurat_df[
                    "seurat_cluster"
                ].nunique()
            ),

        "clusters":
            numeric_cluster_order(
                seurat_df[
                    "seurat_cluster"
                ].unique()
            ),

        "RNA_count_source_type":
            source_type,

        "RNA_count_source_name":
            source_name,

        "old_manual_mapping_used":
            False,

        "Scanpy_annotation_used":
            False,

        "Hungarian_mapping_used":
            False,

        "review_table":
            str(
                review_out
            ),

        "per_cell_predictions":
            str(
                per_cell_out
            ),

        "figure_png":
            str(
                out_png
            ),

        "figure_pdf":
            str(
                out_pdf
            ),
    }

    qc_out = (
        qc_dir
        / f"{cfg['short_name']}_celltypist_cluster_review_qc.json"
    )

    with open(
        qc_out,
        "w",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    print(
        "\n[REVIEW TABLE]",
        review_out,
        flush=True,
    )

    print(
        review[
            [
                "seurat_cluster",
                "n_cells",
                "celltypist_majority_label",
                "celltypist_majority_fraction",
                "celltypist_mean_confidence",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\n[FIGURE]",
        out_png,
        flush=True,
    )

    del ct_adata


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not OVERALL_H5AD.is_file():
        raise FileNotFoundError(
            OVERALL_H5AD
        )

    print(
        "=" * 75,
        flush=True,
    )

    print(
        "CELLTYPIST + AUTHORITATIVE SEURAT CLUSTER ANNOTATION REVIEW",
        flush=True,
    )

    print(
        "=" * 75,
        flush=True,
    )

    print(
        "[INPUT]",
        OVERALL_H5AD,
        flush=True,
    )

    # Backed mode keeps the large source object from being
    # loaded fully into RAM at once.
    overall = sc.read_h5ad(
        OVERALL_H5AD,
        backed="r",
    )

    try:

        for cfg in CONFIGS:

            process_lineage(
                cfg=
                    cfg,
                overall=
                    overall,
            )

    finally:

        try:
            overall.file.close()
        except Exception:
            pass

    print(
        "\n[SUCCESS] T- and B-cell CellTypist review outputs complete.",
        flush=True,
    )


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print(
            "\n[ERROR] CellTypist cluster-review analysis failed.",
            flush=True,
        )

        traceback.print_exc()

        sys.exit(
            1
        )
