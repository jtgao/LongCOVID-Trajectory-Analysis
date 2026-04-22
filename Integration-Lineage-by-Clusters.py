#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-integrate T and B lineages by selecting cells from the OVERALL integrated object
using Leiden cluster IDs, then re-running HVGs/PCA/UMAP/Leiden on each subset.

Place this file at:
  /dcs07/antar/data/analysis/long_covid_clonal/scripts_cluster/reintegrate_lineages_by_cluster.py

Example:
python /dcs07/antar/data/analysis/long_covid_clonal/scripts_cluster/reintegrate_lineages_by_cluster.py \
  --h5ad /dcs07/antar/data/analysis/harry_qc/adata_all_raw_qc_harmony_umap.h5ad \
  --out_base /dcs07/antar/data/analysis/long_covid_clonal \
  --cluster_key leiden \
  --t_clusters 0,16,2,3,6,7 \
  --b_clusters 12,15,5,8 \
  --leiden_resolution 0.8 \
  --n_hvg 3000
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# Optional: CellTypist (annotation)
try:
    import celltypist
except ImportError:
    celltypist = None



def _parse_cluster_list(s: str):
    s = s.strip().strip('"').strip("'")
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]



def _maybe_use_counts_and_log1p(adata_sub, target_sum=1e4, out_layer="log1p_norm"):
    """
    Keep raw counts and store normalized+log1p in a layer.

    - If adata_sub.layers['counts'] exists: use it as input counts.
    - Else: assume adata_sub.X is counts-like and copy it into layers['counts'].
    - Write normalized+log1p into layers[out_layer].
    - Set adata_sub.X = layers[out_layer] for downstream HVG/PCA/UMAP convenience.
    """
    if "counts" in adata_sub.layers:
        counts = adata_sub.layers["counts"].copy()
    else:
        # best-effort: treat current X as counts-like, preserve it
        # SAFETY: if X looks already log-ish, skip re-normalization
        import numpy as np
        x_max = float(adata_sub.X.max()) if hasattr(adata_sub.X, "max") else float(np.max(adata_sub.X))
        if x_max < 50:  # heuristic: log1p values typically not huge
            print("[WARN] No counts layer and X looks log-normalized already; skipping normalize/log1p.")
            adata_sub.layers[out_layer] = adata_sub.X.copy()
            adata_sub.X = adata_sub.layers[out_layer]
            return
        counts = adata_sub.X.copy()
        adata_sub.layers["counts"] = counts

    # build normalized matrix in a temp adata to avoid overwriting counts
    tmp = adata_sub.copy()
    tmp.X = counts.copy()
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)

    adata_sub.layers[out_layer] = tmp.X.copy()
    adata_sub.X = adata_sub.layers[out_layer]  # make plotting/HVG/PCA use normalized log
    print(f"[INFO] Stored normalized+log1p in .layers['{out_layer}'] and set .X to it.")



def run_reintegration(
    adata_sub: ad.AnnData,
    out_dir: str,
    lineage_name: str,
    n_hvg: int = 3000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    leiden_resolution: float = 0.8,
    seed: int = 0,
    do_harmony: bool = False,
    batch_key: str = "sample",
    run_celltypist: bool = False,
    celltypist_model: str = "Immune_All_Low.pkl",
    celltypist_key: str = "celltypist_label",
    celltypist_majority_voting: bool = True,
    celltypist_n_jobs: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] === Reintegrating {lineage_name}: n={adata_sub.n_obs} cells, p={adata_sub.n_vars} genes ===")

    # Work on a copy to avoid view issues
    adata_sub = adata_sub.copy()

    # Normalize/log if possible
    _maybe_use_counts_and_log1p(adata_sub)
    
        # --- Optional: CellTypist annotation on the reintegrated lineage ---
    if run_celltypist:
        if celltypist is None:
            raise ImportError(
                "celltypist is not installed in this env. "
                "Install it or run without --run_celltypist."
            )

        # store lineage-specific key to avoid collisions if you merge later
        # ct_key = f"{celltypist_key}_{lineage_name}"
        ct_key = celltypist_key   # usually "cell_type"

        print(f"[INFO] Running CellTypist for {lineage_name} (model={celltypist_model}) -> obs['{ct_key}']")
        try:
            res = celltypist.annotate(
                adata_sub,
                model=celltypist_model,
                majority_voting=celltypist_majority_voting,
                n_jobs=celltypist_n_jobs,
            )
            # predicted_labels is a pandas Series aligned to obs
            adata_sub.obs[ct_key] = res.predicted_labels.values

            # Optional: keep confidence if available (lightweight)
            if hasattr(res, "confidence"):
                adata_sub.obs[f"{ct_key}_confidence"] = res.confidence.values

        except Exception as e:
            print(f"[WARN] CellTypist failed for {lineage_name}: {e}")


    # HVGs

    # print(f"[INFO] Selecting HVGs (n_hvg={n_hvg})")
    # try:
    #     # Works best with counts, but OK on log data too (fallback)
    #     sc.pp.highly_variable_genes(adata_sub, n_top_genes=n_hvg, flavor="seurat_v3", subset=False)
    # except Exception as e:
    #     print(f"[WARN] seurat_v3 HVG failed ({e}). Falling back to flavor='seurat'.")
    #     sc.pp.highly_variable_genes(adata_sub, n_top_genes=n_hvg, flavor="seurat", subset=False)

    print(f"[INFO] Selecting HVGs (n_hvg={n_hvg})")

    # seurat_v3 wants raw counts; if we don't have trustworthy counts, use 'seurat'
    use_flavor = "seurat_v3" if "counts" in adata_sub.layers else "seurat"

    try:
        sc.pp.highly_variable_genes(
            adata_sub,
            n_top_genes=n_hvg,
            flavor=use_flavor,
            subset=False,
            layer="counts" if use_flavor == "seurat_v3" else None,
        )
    except Exception as e:
        print(f"[WARN] HVG failed with flavor={use_flavor} ({e}). Falling back to flavor='seurat'.")
        sc.pp.highly_variable_genes(adata_sub, n_top_genes=n_hvg, flavor="seurat", subset=False)


    # Scale / PCA / neighbors / UMAP / Leiden
    # print("[INFO] Scaling")
    # sc.pp.scale(adata_sub, max_value=10, zero_center=True, mask_var=adata_sub.var["highly_variable"])

    print("[INFO] Scaling (HVGs only)")
    hvg_mask = adata_sub.var["highly_variable"].to_numpy()

    # scale only HVGs by making a temporary view, then write back into X
    X_scaled = adata_sub.X.copy()
    adata_tmp = adata_sub[:, hvg_mask].copy()
    sc.pp.scale(adata_tmp, max_value=10, zero_center=True)
    X_scaled[:, hvg_mask] = adata_tmp.X
    adata_sub.X = X_scaled
    del adata_tmp


    print("[INFO] PCA")
    sc.tl.pca(
        adata_sub,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=seed,
        use_highly_variable=True
    )


    # --- Harmony integration on PCA to remove batch effects (Yi's note) ---
    rep_for_neighbors = "X_pca"
    if do_harmony:
        if batch_key not in adata_sub.obs.columns:
            raise KeyError(
                f"[Harmony] batch_key='{batch_key}' not found in adata.obs. "
                f"Available keys include: {list(adata_sub.obs.columns)[:30]} ..."
            )

        print(f"[INFO] Harmony integration on PCA using batch_key='{batch_key}'")
        # requires harmonypy + scanpy external
        sc.external.pp.harmony_integrate(
            adata_sub,
            key=batch_key,
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
        )
        rep_for_neighbors = "X_pca_harmony"




    print(f"[INFO] Neighbors (use_rep={rep_for_neighbors})")
    sc.pp.neighbors(
        adata_sub,
        n_neighbors=n_neighbors,
        use_rep=rep_for_neighbors,
        random_state=seed
    )


    print("[INFO] UMAP")
    sc.tl.umap(adata_sub, random_state=seed)

    print(f"[INFO] Leiden (resolution={leiden_resolution})")
    sc.tl.leiden(adata_sub, resolution=leiden_resolution, key_added="leiden_sub", random_state=seed)

    # Save plots
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sc.pl.umap(
        adata_sub,
        color="leiden_sub",
        title=f"{lineage_name} (reintegrated) — Leiden subclusters",
        legend_loc="on data",
        show=False,
    )
    umap_leiden_png = os.path.join(fig_dir, f"{lineage_name}_reintegrated_umap_leiden_sub.png")
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(umap_leiden_png, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Sanity check layers before saving ---
    print("[INFO] layers present:", list(adata_sub.layers.keys()))
    print("[INFO] X shape:", adata_sub.X.shape)
    print("[INFO] n_vars (genes):", adata_sub.n_vars)

    if "counts" not in adata_sub.layers:
        raise RuntimeError(
            f"[ERROR] counts layer is missing before writing {lineage_name}. "
            "Reintegrated output must retain raw counts for downstream pseudobulk."
        )

    # Write h5ad
    out_h5ad = os.path.join(out_dir, "processed_reintegrated.h5ad")
    print("[INFO] Writing:", out_h5ad)

    # --- H5AD safety: make sure obs has only HDF5-writable dtypes ---
    for col in adata_sub.obs.columns:
        s = adata_sub.obs[col]

        # If it's categorical, make categories strings (and keep as category)
        if pd.api.types.is_categorical_dtype(s):
            s = s.cat.rename_categories(lambda x: "" if pd.isna(x) else str(x))
            adata_sub.obs[col] = s
            continue

        # If it's object/string-like, force to plain Python strings
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            adata_sub.obs[col] = s.map(lambda x: "" if pd.isna(x) else str(x))

    # Also explicitly handle patient (since it's the one crashing)
    if "patient" in adata_sub.obs.columns:
        adata_sub.obs["patient"] = adata_sub.obs["patient"].map(lambda x: "" if pd.isna(x) else str(x))

    adata_sub.write_h5ad(out_h5ad, compression="gzip")

    print("[INFO] Saved UMAP:", umap_leiden_png)
    return out_h5ad, umap_leiden_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="OVERALL integrated AnnData (.h5ad) with cluster_key in .obs")
    ap.add_argument("--out_base", required=True, help="Base output dir, e.g. /.../long_covid_clonal")
    ap.add_argument("--cluster_key", default="leiden", help="Overall cluster key in adata.obs")
    ap.add_argument("--t_clusters", required=True, help='Comma-separated overall clusters for T, e.g. "0,16,2,3,6,7"')
    ap.add_argument("--b_clusters", required=True, help='Comma-separated overall clusters for B, e.g. "12,15,5,8"')
    ap.add_argument("--n_hvg", type=int, default=3000)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--leiden_resolution", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_to_patient_csv", default=None,
                help="CSV with columns sample,patient (or sample,donor). Adds adata.obs['patient'] to T/B.")
    # --- Optional: re-run CellTypist on reintegrated T/B ---
    ap.add_argument("--run_celltypist", action="store_true",
                    help="If set, run CellTypist on each reintegrated lineage and store labels in .obs.")
    ap.add_argument("--celltypist_model", default="Immune_All_Low.pkl",
                    help="CellTypist model name or path (e.g. Immune_All_Low.pkl or Immune_All_High.pkl).")
    ap.add_argument("--celltypist_key", default="celltypist",
                    help="obs key to store CellTypist labels (will be lineage-specific with suffix).")
    ap.add_argument("--celltypist_majority_voting", action="store_true",
                    help="Enable CellTypist majority voting refinement.")
    ap.add_argument("--celltypist_n_jobs", type=int, default=8,
                    help="Number of CPU workers for CellTypist.")
    # -------
    ap.add_argument("--batch_key", default="sample",
                help="obs column used as batch key for Harmony integration (usually 'sample').")
    ap.add_argument("--do_harmony", action="store_true",
                help="If set, run Harmony on PCA for each lineage subset using batch_key.")
    args = ap.parse_args()

    t_clusters = _parse_cluster_list(args.t_clusters)
    b_clusters = _parse_cluster_list(args.b_clusters)

    print("[INFO] Reading OVERALL (backed='r'):", args.h5ad)
    adata = sc.read_h5ad(args.h5ad, backed="r")

    if args.cluster_key not in adata.obs:
        raise KeyError(f"Missing adata.obs['{args.cluster_key}'] in OVERALL h5ad")

    # Make sure comparisons match dtype (often categorical/int)
    cluster_series = adata.obs[args.cluster_key].astype(str)

    mask_t = cluster_series.isin(set(t_clusters)).values
    mask_b = cluster_series.isin(set(b_clusters)).values

    # Load only subsets into memory (saves RAM vs loading full object)
    print(f"[INFO] Subsetting T clusters={t_clusters}")
    adata_t = adata[mask_t].to_memory()
    print(f"[INFO] Subsetting B clusters={b_clusters}")
    adata_b = adata[mask_b].to_memory()

    # --- OPTIONAL: add patient column from sample->patient mapping ---
    if args.sample_to_patient_csv is not None:
        mp = pd.read_csv(args.sample_to_patient_csv)

        # accept a few common column names
        if "sample" not in mp.columns:
            raise KeyError("Mapping CSV must contain a 'sample' column.")
        patient_col = None
        for c in ["patient", "donor", "subject", "participant"]:
            if c in mp.columns:
                patient_col = c
                break
        if patient_col is None:
            raise KeyError("Mapping CSV must contain one of: patient/donor/subject/participant columns.")

        sample2patient = mp.set_index("sample")[patient_col].astype(str).to_dict()

        adata_t.obs["patient"] = adata_t.obs["sample"].astype(str).map(sample2patient)
        adata_b.obs["patient"] = adata_b.obs["sample"].astype(str).map(sample2patient)

        # warn if unmapped
        n_miss_t = int(adata_t.obs["patient"].isna().sum())
        n_miss_b = int(adata_b.obs["patient"].isna().sum())
        if n_miss_t or n_miss_b:
            print(f"[WARN] patient mapping missing for T: {n_miss_t} cells, B: {n_miss_b} cells")


    # Output dirs (new cluster-based pipeline outputs)
    out_t = os.path.join(args.out_base, "lineage_T_cluster")
    out_b = os.path.join(args.out_base, "lineage_B_cluster")

    # Save a quick summary CSV
    summary_csv = os.path.join(args.out_base, "figures", "cluster_based_lineage_subset_summary.csv")
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    pd.DataFrame(
        [
            {"lineage": "T", "cluster_key": args.cluster_key, "clusters": ",".join(t_clusters), "n_cells": int(adata_t.n_obs)},
            {"lineage": "B", "cluster_key": args.cluster_key, "clusters": ",".join(b_clusters), "n_cells": int(adata_b.n_obs)},
        ]
    ).to_csv(summary_csv, index=False)
    print("[INFO] Written subset summary:", summary_csv)

    # Reintegrate each lineage
    t_h5ad, t_umap = run_reintegration(
        adata_t, out_t, "T_lineage",
        n_hvg=args.n_hvg,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
        seed=args.seed,
        do_harmony=args.do_harmony,
        batch_key=args.batch_key,
        run_celltypist=args.run_celltypist,
        celltypist_model=args.celltypist_model,
        celltypist_key=args.celltypist_key,
        celltypist_majority_voting=args.celltypist_majority_voting,
        celltypist_n_jobs=args.celltypist_n_jobs,
    )

    b_h5ad, b_umap = run_reintegration(
        adata_b, out_b, "B_lineage",
        n_hvg=args.n_hvg,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
        seed=args.seed,
        do_harmony=args.do_harmony,
        batch_key=args.batch_key,
        run_celltypist=args.run_celltypist,
        celltypist_model=args.celltypist_model,
        celltypist_key=args.celltypist_key,
        celltypist_majority_voting=args.celltypist_majority_voting,
        celltypist_n_jobs=args.celltypist_n_jobs,
    )



    print("[INFO] DONE.")
    print("[INFO] T processed:", t_h5ad)
    print("[INFO] B processed:", b_h5ad)
    print("[INFO] T UMAP:", t_umap)
    print("[INFO] B UMAP:", b_umap)


if __name__ == "__main__":
    main()
