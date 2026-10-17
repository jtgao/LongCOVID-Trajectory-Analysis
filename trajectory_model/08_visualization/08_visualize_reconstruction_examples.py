#!/usr/bin/env python3

"""
STEP 08A — Cross-method visualization of strongest-disagreement examples.

For each selected patient-specific TCR clonotype:

Top row:
    Full Seurat native UMAP background with:
        SP
        MST
        Steiner

Bottom row:
    The exact same three reconstructed graphs,
    shown with shared zoom bounds.

Selection:
    - >= MIN_ANCHORS anchors
    - >= MIN_NODES reconstructed nodes in at least one method
    - select groups with strongest method disagreement
      according to mean edge Jaccard

IMPORTANT
---------
The Seurat UMAP is used ONLY for visualization.

All reconstruction nodes and edges remain exactly those produced
previously by Steps 04–06 from the Step 02 kNN graph.

The Step 02 node_index <-> cell_id mapping remains authoritative.
Seurat UMAP coordinates are matched to those node indices by exact
cell_id.

No trajectory, edge, anchor, bridge, Jaccard value, or selected group
is recomputed from UMAP coordinates.
"""

print(
    "[START] Step 08A Python started.",
    flush=True,
)

import json
import sys
import traceback
from pathlib import Path

print(
    "[IMPORT] NumPy...",
    flush=True,
)
import numpy as np

print(
    "[IMPORT] pandas...",
    flush=True,
)
import pandas as pd

print(
    "[IMPORT] matplotlib...",
    flush=True,
)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

print(
    "[IMPORT] Imports complete.",
    flush=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_ANCHORS = 15
MIN_NODES = 20

N_DISAGREEMENT = 3

ZOOM_PADDING_FRACTION = 0.10


# ============================================================
# PATHS
# ============================================================

import argparse

parser = argparse.ArgumentParser(description="Trajectory stage 08. Run as a command-line script.")
parser.add_argument("--project-dir", type=Path, required=True, help="User-owned run directory containing numbered stage outputs; independent of source-code location.")
parser.add_argument("--seurat-coordinates", type=Path, required=True, help="External Seurat coordinate CSV/CSV.GZ, joined by cell_id.")
parser.add_argument("--show-identifiers", action="store_true", help="Explicitly expose patient/clonotype identifiers in figure titles and filenames.")
args = parser.parse_args()
PROJECT = args.project_dir

# Authoritative reconstruction node_index <-> cell_id mapping.
CELL_PATH = (
    PROJECT
    / "02_knn_graph"
    / "outputs"
    / "02_T_cell_index.csv.gz"
)

# Newer Seurat-native UMAP coordinates used by the previous
# Seurat trajectory workflow.
SEURAT_UMAP_PATH = args.seurat_coordinates

COMPARE_PATH = (
    PROJECT
    / "07_cross_method_comparison"
    / "outputs"
    / "07_cross_method_group_comparison.csv"
)

METHODS = {

    "SP": {

        "nodes": (
            PROJECT
            / "04_root_shortest_paths"
            / "outputs"
            / "04_method1_nodes.csv.gz"
        ),

        "edges": (
            PROJECT
            / "04_root_shortest_paths"
            / "outputs"
            / "04_method1_edges.csv.gz"
        ),

        "title":
            "Rooted Shortest-Path Tree",
    },

    "MST": {

        "nodes": (
            PROJECT
            / "05_mst_reconstruction"
            / "outputs"
            / "05_method2_mst_nodes.csv.gz"
        ),

        "edges": (
            PROJECT
            / "05_mst_reconstruction"
            / "outputs"
            / "05_method2_mst_edges.csv.gz"
        ),

        "title":
            "Minimum Spanning Tree",
    },

    "Steiner": {

        "nodes": (
            PROJECT
            / "06_steiner_reconstruction"
            / "outputs"
            / "06_method3_steiner_nodes.csv.gz"
        ),

        "edges": (
            PROJECT
            / "06_steiner_reconstruction"
            / "outputs"
            / "06_method3_steiner_edges.csv.gz"
        ),

        "title":
            "Steiner Approximation",
    },
}

OUT = (
    PROJECT
    / "08_visualization"
    / "outputs"
)

FIG_DIR = (
    OUT
    / "figures"
)


# ============================================================
# HELPERS
# ============================================================

def safe_filename(text):

    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def calculate_zoom_bounds(
    all_nodes,
    umap,
    padding_fraction=0.10,
):

    """
    Calculate one common zoom window from the union of all
    reconstruction nodes across the three methods.
    """

    all_nodes = np.array(
        sorted(
            set(
                map(
                    int,
                    all_nodes,
                )
            )
        ),
        dtype=np.int64,
    )

    coords = umap[
        all_nodes,
        :
    ]

    xmin = float(
        coords[:, 0].min()
    )

    xmax = float(
        coords[:, 0].max()
    )

    ymin = float(
        coords[:, 1].min()
    )

    ymax = float(
        coords[:, 1].max()
    )

    xspan = (
        xmax
        - xmin
    )

    yspan = (
        ymax
        - ymin
    )

    if xspan == 0:
        xspan = 1.0

    if yspan == 0:
        yspan = 1.0

    xpad = (
        xspan
        * padding_fraction
    )

    ypad = (
        yspan
        * padding_fraction
    )

    return (
        xmin - xpad,
        xmax + xpad,
        ymin - ypad,
        ymax + ypad,
    )


def plot_method(
    ax,
    ndf,
    edf,
    umap,
    title,
    zoom_bounds=None,
):

    """
    Plot one previously reconstructed graph on the Seurat UMAP.

    If zoom_bounds is None:
        show the full Seurat UMAP.

    Otherwise:
        show the exact same reconstruction with zoomed axis limits.
    """

    # --------------------------------------------------------
    # FULL CLEANED T-CELL BACKGROUND
    # --------------------------------------------------------

    ax.scatter(
        umap[:, 0],
        umap[:, 1],
        s=1,
        alpha=0.05,
        c="lightgray",
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # --------------------------------------------------------
    # RECONSTRUCTION EDGES
    # --------------------------------------------------------

    for edge in edf.itertuples(
        index=False
    ):

        u = int(
            edge.node_u_index
        )

        v = int(
            edge.node_v_index
        )

        ax.plot(
            [
                umap[u, 0],
                umap[v, 0],
            ],
            [
                umap[u, 1],
                umap[v, 1],
            ],
            c="black",
            linewidth=1.0,
            alpha=0.75,
            zorder=2,
        )

    # --------------------------------------------------------
    # BRIDGE NODES
    # --------------------------------------------------------

    bridge_nodes = (
        ndf.loc[
            ndf[
                "node_role"
            ].eq(
                "bridge"
            ),
            "node_index",
        ]
        .astype(int)
        .to_numpy()
    )

    if len(
        bridge_nodes
    ) > 0:

        ax.scatter(
            umap[
                bridge_nodes,
                0
            ],
            umap[
                bridge_nodes,
                1
            ],
            s=18,
            c="gray",
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )

    # --------------------------------------------------------
    # ANCHOR NODES
    # --------------------------------------------------------

    anchor_nodes = (
        ndf.loc[
            ndf[
                "node_role"
            ].eq(
                "anchor"
            ),
            "node_index",
        ]
        .astype(int)
        .to_numpy()
    )

    ax.scatter(
        umap[
            anchor_nodes,
            0
        ],
        umap[
            anchor_nodes,
            1
        ],
        s=34,
        c="red",
        edgecolors="black",
        linewidths=0.35,
        zorder=4,
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    ax.set_title(
        f"{title}\n"
        f"{len(ndf)} nodes, "
        f"{len(bridge_nodes)} bridges",
        fontsize=11,
    )

    ax.set_xlabel(
        "Seurat UMAP 1"
    )

    ax.set_ylabel(
        "Seurat UMAP 2"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    if zoom_bounds is not None:

        (
            xmin,
            xmax,
            ymin,
            ymax,
        ) = zoom_bounds

        ax.set_xlim(
            xmin,
            xmax,
        )

        ax.set_ylim(
            ymin,
            ymax,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    print(
        "[01] Checking input files...",
        flush=True,
    )

    required = [
        CELL_PATH,
        SEURAT_UMAP_PATH,
        COMPARE_PATH,
    ]

    for info in METHODS.values():

        required.append(
            info["nodes"]
        )

        required.append(
            info["edges"]
        )

    for path in required:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    print(
        "[01] Inputs found.",
        flush=True,
    )

    # --------------------------------------------------------
    # STEP 02 NODE INDEX + NEWER SEURAT UMAP
    # --------------------------------------------------------

    print(
        "[02] Loading Step 02 node index and Seurat UMAP...",
        flush=True,
    )

    # Keep only the authoritative Step 02 node indexing.
    cells = pd.read_csv(
        CELL_PATH,
        usecols=[
            "node_index",
            "cell_id",
        ],
        dtype={
            "cell_id":
                "string",
        },
    )

    # Load visualization coordinates from the newer Seurat export.
    seurat_umap = pd.read_csv(
        SEURAT_UMAP_PATH,
        usecols=[
            "cell_id",
            "seurat_umap_1",
            "seurat_umap_2",
        ],
        dtype={
            "cell_id":
                "string",
        },
    )

    # --------------------------------------------------------
    # MATCHING QC
    # --------------------------------------------------------

    if cells[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate cell_id values found in "
            "02_T_cell_index.csv.gz."
        )

    if seurat_umap[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate cell_id values found in "
            "Tcells_seurat_for_trajectory.csv.gz."
        )

    n_step02_cells = int(
        len(cells)
    )

    n_seurat_cells = int(
        len(seurat_umap)
    )

    cells = cells.merge(
        seurat_umap,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    # Restore exact node_index order used by all saved graph outputs.
    cells = (
        cells
        .sort_values(
            "node_index"
        )
        .reset_index(
            drop=True
        )
    )

    if not np.array_equal(
        cells[
            "node_index"
        ].to_numpy(),
        np.arange(
            len(cells)
        ),
    ):

        raise RuntimeError(
            "Step 02 node indices are not sequential "
            "after Seurat UMAP matching."
        )

    missing_umap = (
        cells[
            [
                "seurat_umap_1",
                "seurat_umap_2",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    n_missing_umap = int(
        missing_umap.sum()
    )

    if n_missing_umap > 0:

        examples = (
            cells.loc[
                missing_umap,
                "cell_id",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{n_missing_umap:,} Step 02 cells are missing "
            f"Seurat UMAP coordinates. "
            f"Example cell IDs: {examples}"
        )

    umap = (
        cells[
            [
                "seurat_umap_1",
                "seurat_umap_2",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    print(
        f"[02] Step 02 cells: "
        f"{n_step02_cells:,}",
        flush=True,
    )

    print(
        f"[02] Seurat export cells: "
        f"{n_seurat_cells:,}",
        flush=True,
    )

    print(
        f"[02] Exact Seurat UMAP matches: "
        f"{len(cells):,}",
        flush=True,
    )

    print(
        f"[02] Missing Seurat UMAP coordinates: "
        f"{n_missing_umap:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # STEP 07 RESULTS
    # --------------------------------------------------------

    print(
        "[03] Loading Step 07 comparison...",
        flush=True,
    )

    comparison = pd.read_csv(
        COMPARE_PATH,
        dtype={
            "group_key":
                "string",
            "patient":
                "string",
            "clonotype":
                "string",
        },
    )

    comparison[
        "mean_edge_jaccard"
    ] = comparison[
        [
            "sp_mst_edge_jaccard",
            "sp_steiner_edge_jaccard",
            "mst_steiner_edge_jaccard",
        ]
    ].mean(
        axis=1
    )

    comparison[
        "mean_node_jaccard"
    ] = comparison[
        [
            "sp_mst_node_jaccard",
            "sp_steiner_node_jaccard",
            "mst_steiner_node_jaccard",
        ]
    ].mean(
        axis=1
    )

    comparison[
        "max_method_nodes"
    ] = comparison[
        [
            "sp_nodes",
            "mst_nodes",
            "steiner_nodes",
        ]
    ].max(
        axis=1
    )

    eligible = comparison[
        (
            comparison[
                "n_anchor_cells"
            ]
            >= MIN_ANCHORS
        )
        &
        (
            comparison[
                "max_method_nodes"
            ]
            >= MIN_NODES
        )
    ].copy()

    print(
        f"[03] Eligible groups: "
        f"{len(eligible):,}",
        flush=True,
    )

    if len(
        eligible
    ) == 0:

        raise RuntimeError(
            "No groups satisfy visualization criteria."
        )

    # --------------------------------------------------------
    # SELECT STRONGEST DISAGREEMENT GROUPS
    # --------------------------------------------------------

    selected = (
        eligible
        .sort_values(
            [
                "mean_edge_jaccard",
                "mean_node_jaccard",
                "n_anchor_cells",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .head(
            N_DISAGREEMENT
        )
        .copy()
    )

    selected[
        "selection_category"
    ] = (
        "method_disagreement"
    )

    selected.to_csv(
        OUT
        / "08_selected_visualization_groups.csv",
        index=False,
    )

    print(
        "\n[04] Selected groups:",
        flush=True,
    )

    print(
        selected[
            [
                "group_key",
                "n_anchor_cells",
                "sp_nodes",
                "mst_nodes",
                "steiner_nodes",
                "mean_edge_jaccard",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    selected_keys = set(
        selected[
            "group_key"
        ].astype(
            str
        )
    )

    # --------------------------------------------------------
    # LOAD PREVIOUSLY COMPUTED METHOD TABLES
    # --------------------------------------------------------

    print(
        "[05] Loading selected reconstruction outputs...",
        flush=True,
    )

    method_nodes = {}
    method_edges = {}

    for (
        method,
        info,
    ) in METHODS.items():

        nodes = pd.read_csv(
            info[
                "nodes"
            ],
            dtype={
                "group_key":
                    "string",
                "node_role":
                    "string",
            },
        )

        edges = pd.read_csv(
            info[
                "edges"
            ],
            dtype={
                "group_key":
                    "string",
            },
        )

        nodes = nodes[
            nodes[
                "group_key"
            ].isin(
                selected_keys
            )
        ].copy()

        edges = edges[
            edges[
                "group_key"
            ].isin(
                selected_keys
            )
        ].copy()

        method_nodes[
            method
        ] = {

            str(key):
                group.copy()

            for (
                key,
                group,
            ) in nodes.groupby(
                "group_key",
                sort=False,
            )
        }

        method_edges[
            method
        ] = {

            str(key):
                group.copy()

            for (
                key,
                group,
            ) in edges.groupby(
                "group_key",
                sort=False,
            )
        }

        print(
            f"[05] {method}: "
            f"{len(nodes):,} node rows; "
            f"{len(edges):,} edge rows.",
            flush=True,
        )

    # --------------------------------------------------------
    # VERIFY SELECTED GROUPS EXIST IN ALL METHODS
    # --------------------------------------------------------

    for key in selected_keys:

        for method in METHODS:

            if key not in method_nodes[
                method
            ]:

                raise RuntimeError(
                    f"Missing node table for "
                    f"group {key} in {method}."
                )

            if key not in method_edges[
                method
            ]:

                raise RuntimeError(
                    f"Missing edge table for "
                    f"group {key} in {method}."
                )

    # --------------------------------------------------------
    # CREATE 2 x 3 FIGURES
    # --------------------------------------------------------

    print(
        "[06] Creating Seurat-UMAP full + zoomed figures...",
        flush=True,
    )

    figure_manifest = []

    method_order = [
        "SP",
        "MST",
        "Steiner",
    ]

    for example_number, row in enumerate(selected.itertuples(index=False), start=1):

        key = str(
            row.group_key
        )

        display_key = key if args.show_identifiers else f"example_{example_number:02d}"
        display_title = (
            f"Patient {row.patient} | TCR {row.clonotype}"
            if args.show_identifiers else f"Example {example_number:02d}"
        )

        # ----------------------------------------------------
        # COMMON ZOOM REGION FROM UNION OF ALL THREE METHODS
        # ----------------------------------------------------

        all_reconstruction_nodes = set()

        for method in method_order:

            ndf = (
                method_nodes[
                    method
                ][
                    key
                ]
            )

            all_reconstruction_nodes.update(
                ndf[
                    "node_index"
                ]
                .astype(int)
                .tolist()
            )

        zoom_bounds = calculate_zoom_bounds(
            all_nodes=
                all_reconstruction_nodes,
            umap=
                umap,
            padding_fraction=
                ZOOM_PADDING_FRACTION,
        )

        # ----------------------------------------------------
        # 2 ROWS x 3 METHODS
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(
                18,
                11,
            ),
            sharex=False,
            sharey=False,
        )

        for (
            col,
            method,
        ) in enumerate(
            method_order
        ):

            ndf = (
                method_nodes[
                    method
                ][
                    key
                ]
            )

            edf = (
                method_edges[
                    method
                ][
                    key
                ]
            )

            # Top row: complete Seurat UMAP.
            plot_method(
                ax=
                    axes[
                        0,
                        col,
                    ],
                ndf=
                    ndf,
                edf=
                    edf,
                umap=
                    umap,
                title=
                    METHODS[
                        method
                    ][
                        "title"
                    ],
                zoom_bounds=
                    None,
            )

            # Bottom row: same exact reconstruction,
            # only change the axis limits.
            plot_method(
                ax=
                    axes[
                        1,
                        col,
                    ],
                ndf=
                    ndf,
                edf=
                    edf,
                umap=
                    umap,
                title=
                    f"{METHODS[method]['title']} — Zoom",
                zoom_bounds=
                    zoom_bounds,
            )

        # ----------------------------------------------------
        # ROW LABELS
        # ----------------------------------------------------

        axes[
            0,
            0,
        ].text(
            -0.18,
            0.5,
            "Full Seurat UMAP",
            transform=
                axes[
                    0,
                    0,
                ].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

        axes[
            1,
            0,
        ].text(
            -0.18,
            0.5,
            "Zoomed View",
            transform=
                axes[
                    1,
                    0,
                ].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

        # ----------------------------------------------------
        # LEGEND
        # ----------------------------------------------------

        legend_handles = [

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="red",
                markeredgecolor="black",
                markersize=8,
                label="Anchor",
            ),

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="none",
                markersize=8,
                label="Bridge",
            ),

            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.2,
                label="Reconstruction edge",
            ),
        ]

        fig.legend(
            handles=
                legend_handles,
            loc=
                "center right",
            bbox_to_anchor=(
                0.985,
                0.5,
            ),
            frameon=True,
            title=
                "Node / Edge Key",
        )

        # ----------------------------------------------------
        # MAIN TITLE
        # ----------------------------------------------------

        fig.suptitle(
            f"{display_title}\n"
            f"{row.n_anchor_cells} anchors | "
            f"method disagreement | "
            f"mean edge Jaccard = "
            f"{row.mean_edge_jaccard:.3f}",
            fontsize=14,
        )

        fig.tight_layout(
            rect=[
                0.03,
                0.02,
                0.90,
                0.93,
            ]
        )

        filename_base = (
            f"{safe_filename(display_key)}"
            "__method_disagreement"
            "__seurat_umap"
            "__full_and_zoom"
        )

        png_path = (
            FIG_DIR
            / f"{filename_base}.png"
        )

        pdf_path = (
            FIG_DIR
            / f"{filename_base}.pdf"
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        figure_manifest.append({

            "group_key":
                key,

            "patient":
                str(
                    row.patient
                ),

            "clonotype":
                str(
                    row.clonotype
                ),

            "selection_category":
                "method_disagreement",

            "n_anchor_cells":
                int(
                    row.n_anchor_cells
                ),

            "mean_node_jaccard":
                float(
                    row.mean_node_jaccard
                ),

            "mean_edge_jaccard":
                float(
                    row.mean_edge_jaccard
                ),

            "zoom_xmin":
                float(
                    zoom_bounds[0]
                ),

            "zoom_xmax":
                float(
                    zoom_bounds[1]
                ),

            "zoom_ymin":
                float(
                    zoom_bounds[2]
                ),

            "zoom_ymax":
                float(
                    zoom_bounds[3]
                ),

            "png":
                png_path.name,

            "pdf":
                pdf_path.name,
        })

        print(
            f"[06] Saved {key}",
            flush=True,
        )

    # --------------------------------------------------------
    # SAVE MANIFEST
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        figure_manifest
    )

    manifest.to_csv(
        OUT
        / "08_figure_manifest.csv",
        index=False,
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    qc = {

        "minimum_anchor_threshold":
            MIN_ANCHORS,

        "minimum_method_node_threshold":
            MIN_NODES,

        "eligible_groups":
            int(
                len(
                    eligible
                )
            ),

        "selected_groups":
            int(
                len(
                    selected
                )
            ),

        "selection_rule":
            (
                "three groups with lowest mean edge Jaccard "
                "among eligible groups"
            ),

        "figure_layout":
            "2 rows x 3 columns",

        "top_row":
            (
                "full Seurat native UMAP view "
                "for SP, MST, Steiner"
            ),

        "bottom_row":
            (
                "same exact reconstructions "
                "with shared zoom bounds"
            ),

        "zoom_rule":
            (
                "bounding box of union of reconstruction nodes "
                "across all three methods plus 10 percent padding"
            ),

        "legend": {

            "red":
                "anchor",

            "gray":
                "bridge",

            "black_line":
                "reconstruction edge",
        },

        "step02_cell_index_source":
            str(
                CELL_PATH
            ),

        "visualization_embedding":
            "Seurat native UMAP",

        "visualization_source":
            str(
                SEURAT_UMAP_PATH
            ),

        "visualization_columns": [
            "seurat_umap_1",
            "seurat_umap_2",
        ],

        "step02_cells":
            int(
                n_step02_cells
            ),

        "seurat_export_cells":
            int(
                n_seurat_cells
            ),

        "visualization_matched_cells":
            int(
                len(
                    cells
                )
            ),

        "visualization_missing_coordinates":
            int(
                n_missing_umap
            ),

        "coordinate_match_rule":
            (
                "exact cell_id match from Seurat export "
                "to Step 02 node_index mapping"
            ),

        "umap_use":
            (
                "visualization only; not used for graph "
                "construction, reconstruction, edge weights, "
                "group selection, or Jaccard"
            ),

        "reconstruction_outputs_recomputed":
            False,

        "figures_created_png":
            int(
                len(
                    manifest
                )
            ),

        "figures_created_pdf":
            int(
                len(
                    manifest
                )
            ),
    }

    with open(
        OUT
        / "08_visualization_qc.json",
        "w",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    print(
        "\n===== STEP 08A QC =====",
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
        "\n[SUCCESS] Step 08A completed.",
        flush=True,
    )


if __name__ == "__main__":

    try:

        main()

    except Exception:

        print(
            "\n[ERROR] Step 08A failed.",
            flush=True,
        )

        traceback.print_exc()

        sys.exit(1)
