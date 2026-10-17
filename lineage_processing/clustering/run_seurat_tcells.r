#!/usr/bin/env Rscript
# run_seurat_tcells_native.R - T-Cell Lineage Re-clustering via Seurat Native CCA Integration

library(Seurat)
library(SeuratObject)
library(dplyr)

cat("[INFO] Starting Seurat native integration validation pipeline...\n")

# --- 1. Paths Setup ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
    stop("Usage: Rscript <this-script.r> <overall-counts.h5ad> <output-directory>")
}
overall_h5ad <- normalizePath(args[1], mustWork = TRUE)
out_dir <- args[2]

mtx_dir <- file.path(out_dir, "temp_mtx_transfer")
dir.create(mtx_dir, showWarnings = FALSE, recursive = TRUE)

scripts_dir <- file.path(out_dir, "scripts")
dir.create(scripts_dir, showWarnings = FALSE, recursive = TRUE)

py_script_path <- file.path(scripts_dir, "extract_t_cells.py")

# --- 2. Text Matrix Bridge ---
if (!file.exists(file.path(mtx_dir, "matrix.mtx"))) {
    cat("[INFO] Writing temporary Python extraction helper script to disk...\n")
    py_lines <- c(
        "import scanpy as sc",
        "import scipy.io as io",
        "import os",
        "import pandas as pd",
        "",
        sprintf("h5ad_in = '%s'", overall_h5ad),
        sprintf("out_path = '%s'", mtx_dir),
        "",
        "print('[PY_INFO] Loading large h5ad matrix...')",
        "adata = sc.read_h5ad(h5ad_in)",
        "whitelist = ['0', '16', '2', '3', '6', '7']",
        "",
        "print('[PY_INFO] Slicing matching T-cell lineage index arrays...')",
        "t_cells = adata[adata.obs['leiden'].astype(str).isin(whitelist)].copy()",
        "",
        "X_mat = t_cells.layers['counts'] if 'counts' in t_cells.layers else t_cells.X",
        "print('[PY_INFO] Writing raw Matrix Market (.mtx) arrays...')",
        "io.mmwrite(os.path.join(out_path, 'matrix.mtx'), X_mat.T)",
        "",
        "print('[PY_INFO] Dumping metadata...')",
        "t_cells.obs[['leiden', 'sample']].to_csv(os.path.join(out_path, 'metadata.csv'))",
        "",
        "with open(os.path.join(out_path, 'barcodes.tsv'), 'w') as f:",
        "    for item in t_cells.obs_names:",
        "        f.write(f'{item}\\n')",
        "",
        "with open(os.path.join(out_path, 'features.tsv'), 'w') as f:",
        "    for item in t_cells.var_names:",
        "        f.write(f'{item}\\n')",
        "print('[PY_INFO] Python data translation complete.')"
    )
    writeLines(py_lines, con = py_script_path)

    cat("[INFO] Executing Python script to generate matrix files...\n")
    system(sprintf("python3 %s", py_script_path))
    if (file.exists(py_script_path)) file.remove(py_script_path)
}

cat("[INFO] Loading raw matrix into Seurat...\n")
counts_mat <- ReadMtx(
    mtx = file.path(mtx_dir, "matrix.mtx"),
    cells = file.path(mtx_dir, "barcodes.tsv"),
    features = file.path(mtx_dir, "features.tsv"),
    feature.column = 1
)

metadata_df <- read.csv(file.path(mtx_dir, "metadata.csv"), row.names = 1)

seurat_obj <- CreateSeuratObject(
    counts = counts_mat,
    meta.data = metadata_df,
    assay = "RNA"
)

unlink(mtx_dir, recursive = TRUE)

seurat_obj$scanpy_initial_leiden <- seurat_obj$leiden

# --- 3. Seurat Native Preprocessing ---
cat("[INFO] Splitting RNA layers by patient sample...\n")
Idents(seurat_obj) <- "sample"
seurat_obj[["RNA"]] <- split(seurat_obj[["RNA"]], f = seurat_obj$sample)

cat("[INFO] Running Seurat preprocessing: NormalizeData, FindVariableFeatures, ScaleData, PCA...\n")
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)

# --- 4. Seurat Native Integration: CCAIntegration ---
cat("[INFO] Running Seurat native CCA integration across patient/sample layers...\n")
seurat_obj <- IntegrateLayers(
    object = seurat_obj,
    method = CCAIntegration,
    orig.reduction = "pca",
    new.reduction = "integrated.cca",
    verbose = FALSE
)

seurat_obj <- JoinLayers(seurat_obj)

# --- 5. Graph Clustering via Seurat Native Integrated Space ---
cat("[INFO] Computing neighbor graph, Louvain clusters, and UMAP on integrated.cca...\n")
seurat_obj <- FindNeighbors(seurat_obj, reduction = "integrated.cca", dims = 1:30)
seurat_obj <- FindClusters(
    seurat_obj,
    resolution = 0.8,
    algorithm = 1,
    cluster.name = "seurat_clusters"
)
seurat_obj <- RunUMAP(
    seurat_obj,
    reduction = "integrated.cca",
    dims = 1:30,
    reduction.name = "umap.seurat"
)

# --- 6. Marker Discovery ---
cat("[INFO] Running FindAllMarkers on Seurat-native subclusters...\n")
Idents(seurat_obj) <- "seurat_clusters"

seurat_markers <- FindAllMarkers(
    seurat_obj,
    only.pos = TRUE,
    min.pct = 0.25,
    logfc.threshold = 0.25,
    max.cells.per.ident = 2000,
    test.use = "wilcox"
)

# --- 7. Export Outputs ---
cat("[INFO] Exporting final processed files...\n")
write.csv(
    seurat_markers,
    file = file.path(out_dir, "Tcells_seurat_native_CCA_highly_expressive_genes.csv"),
    row.names = FALSE
)

saveRDS(
    seurat_obj,
    file = file.path(out_dir, "Tcells_seurat_native_CCA_integrated_validated.rds")
)

cat("[SUCCESS] Seurat native CCA integration pipeline complete!\n")
