#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pseudobulk computation (per-sample per-group) WITH global group proportions.

Key features
- Aggregates single-cell expression to pseudobulk per (sample, group), where
  "group" can be cell types (CellTypist) or subclusters (e.g. leiden_sub).
- Saves, for each group g:
    output_dir/<g>/pseudobulk.h5ad
    output_dir/<g>/pseudobulk_expression.csv
    output_dir/<g>/pseudobulk_metadata.csv   (includes pb.obs['n_cells'])
- Saves global (groups x samples) tables:
    output_dir/group_counts.csv
    output_dir/group_proportions.csv
  and optionally:
    output_dir/group_proportions.h5ad

Notes
- Prefer aggregating from raw counts. Use --layer counts if present.
- If --layer is not provided or not found, falls back to .X.
"""

import os
import gc
import argparse
import warnings
import contextlib
import io
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse, csr_matrix

# Optional GPU
try:
    import cupy as cp
    import rapids_singlecell as rsc
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# =============================================================================
# Utils
# =============================================================================

def _as_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _sanitize_name(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def clear_gpu_memory():
    if not GPU_AVAILABLE:
        return
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def to_gpu(adata_obj: ad.AnnData) -> ad.AnnData:
    if GPU_AVAILABLE:
        rsc.get.anndata_to_GPU(adata_obj)
    return adata_obj


def to_cpu(adata_obj: ad.AnnData) -> ad.AnnData:
    if GPU_AVAILABLE:
        rsc.get.anndata_to_CPU(adata_obj)
    return adata_obj


def _get_matrix(adata_obj: ad.AnnData, layer: Optional[str], verbose: bool = False):
    """
    Return expression matrix to aggregate from.

    Rules:
    - If layer is None: use .X
    - If layer == 'counts': require adata.layers['counts']
    - If layer is another valid layer: use that layer
    - Otherwise raise an error
    """
    if layer is None:
        if verbose:
            print("[INFO] Aggregating from adata.X")
        return adata_obj.X

    if layer == "counts":
        if "counts" not in adata_obj.layers:
            raise KeyError(
                "Requested layer='counts' but adata.layers['counts'] was not found. "
                "This pseudobulk step should aggregate raw counts if available."
            )
        if verbose:
            print("[INFO] Aggregating from adata.layers['counts']")
        return adata_obj.layers["counts"]

    if layer in adata_obj.layers:
        if verbose:
            print(f"[INFO] Aggregating from adata.layers['{layer}']")
        return adata_obj.layers[layer]

    raise KeyError(
        f"Requested layer='{layer}' but it was not found in adata.layers. "
        f"Available layers: {list(adata_obj.layers.keys())}"
    )


# =============================================================================
# Sample filtering (optional)
# =============================================================================

def filter_samples(
    adata_obj: ad.AnnData,
    sample_col: str = "sample",
    group_col: Optional[str] = None,
    keep_groups: Optional[Union[str, List[str]]] = None,
    verbose: bool = False
) -> ad.AnnData:
    if group_col is None or keep_groups is None:
        if verbose:
            print("[INFO] No sample filtering applied.")
        return adata_obj

    if group_col not in adata_obj.obs.columns:
        if verbose:
            print(f"[WARN] group_col='{group_col}' not found; skipping filtering.")
        return adata_obj

    keep_groups_list = _as_list(keep_groups)
    mask = adata_obj.obs[group_col].isin(keep_groups_list)

    out = adata_obj[mask].copy()
    if verbose:
        print(f"[INFO] Filter by {group_col} in {keep_groups_list}: "
              f"{adata_obj.n_obs} -> {out.n_obs} cells.")
        print(f"[INFO] Samples: {adata_obj.obs[sample_col].nunique()} -> {out.obs[sample_col].nunique()}")
    return out


# =============================================================================
# Global counts / proportions table (groups x samples)
# =============================================================================

def compute_group_counts_and_proportions(
    adata_obj: ad.AnnData,
    sample_col: str,
    group_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    samples = sorted(adata_obj.obs[sample_col].astype(str).unique())
    groups = sorted(adata_obj.obs[group_col].astype(str).unique())

    counts = pd.crosstab(
        adata_obj.obs[group_col].astype(str),
        adata_obj.obs[sample_col].astype(str),
    )
    counts = counts.reindex(index=groups, columns=samples, fill_value=0)

    totals = counts.sum(axis=0)
    totals[totals == 0] = 1
    props = (counts / totals).astype(float)

    return counts, props


def save_group_summary_tables(
    counts_df: pd.DataFrame,
    props_df: pd.DataFrame,
    output_dir: str,
    verbose: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    counts_path = os.path.join(output_dir, "group_counts.csv")
    props_path = os.path.join(output_dir, "group_proportions.csv")
    counts_df.to_csv(counts_path)
    props_df.to_csv(props_path)
    if verbose:
        print(f"[INFO] Saved: {counts_path}")
        print(f"[INFO] Saved: {props_path}")


def save_group_proportions_anndata(
    props_df: pd.DataFrame,
    output_dir: str,
    verbose: bool = False
):
    """
    Save proportions as AnnData:
      obs = samples
      var = groups
      X   = proportions (samples x groups)
    """
    os.makedirs(output_dir, exist_ok=True)
    X = props_df.T.values.astype(np.float32)
    ad_obj = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=props_df.columns),
        var=pd.DataFrame(index=props_df.index),
    )
    ad_obj.obs.index.name = "sample"
    ad_obj.var.index.name = "group"
    ad_obj.uns["note"] = "Group proportions per sample computed from adata.obs."
    path = os.path.join(output_dir, "group_proportions.h5ad")
    ad_obj.write_h5ad(path, compression="gzip")
    if verbose:
        print(f"[INFO] Saved: {path}")


# =============================================================================
# Pseudobulk aggregation (per-sample per-group)
# =============================================================================

def aggregate_to_pseudobulk_per_group(
    adata_obj: ad.AnnData,
    sample_col: str,
    group_col: str,
    layer: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, ad.AnnData]:
    samples = sorted(adata_obj.obs[sample_col].astype(str).unique())
    groups = sorted(adata_obj.obs[group_col].astype(str).unique())
    n_samples = len(samples)

    sample_to_idx = {s: i for i, s in enumerate(samples)}
    pseudobulk: Dict[str, ad.AnnData] = {}

    # sample-level metadata (constant within a sample)
    sample_metadata = {}
    for col in adata_obj.obs.columns:
        if col in [sample_col, group_col]:
            continue
        grouped = adata_obj.obs.groupby(sample_col)[col].apply(lambda x: x.dropna().unique())
        if grouped.apply(lambda u: len(u) <= 1).all():
            sample_metadata[col] = grouped.apply(lambda u: u[0] if len(u) > 0 else np.nan)

    for g in groups:
        g_mask = adata_obj.obs[group_col].astype(str) == g
        g_ad = adata_obj[g_mask]
        if g_ad.n_obs == 0:
            continue

        cell_to_sample = g_ad.obs[sample_col].astype(str).map(sample_to_idx).values
        indicator = csr_matrix(
            (np.ones(g_ad.n_obs, dtype=np.float32), (cell_to_sample, np.arange(g_ad.n_obs))),
            shape=(n_samples, g_ad.n_obs)
        )

        X = _get_matrix(g_ad, layer, verbose=verbose)
        if issparse(X):
            pb_mat = (indicator @ X).toarray().astype(np.float32)
        else:
            pb_mat = (indicator @ X).astype(np.float32)

        pb = ad.AnnData(
            X=pb_mat,
            obs=pd.DataFrame(index=samples),
            var=g_ad.var.copy(),
        )
        pb.obs.index.name = "sample"

        if sample_metadata:
            pb.obs = pb.obs.join(pd.DataFrame(sample_metadata), how="left")

        # per-sample cell counts for this group
        pb.obs["n_cells"] = indicator.sum(axis=1).A.flatten().astype(int)

        pseudobulk[g] = pb

        if verbose:
            print(f"[INFO] Group={g}: {g_ad.n_obs} cells -> {n_samples} samples")

    return pseudobulk


# =============================================================================
# Processing options (normalize/log/combat)
# =============================================================================

def apply_combat(
    adata_obj: ad.AnnData,
    batch_col: Union[str, List[str]],
    covariates: Optional[List[str]] = None,
    verbose: bool = False
) -> bool:
    batch_cols = _as_list(batch_col)
    batch_cols = [c for c in batch_cols if c in adata_obj.obs.columns]
    if not batch_cols:
        return False

    if len(batch_cols) == 1:
        batch_key = batch_cols[0]
    else:
        batch_key = "_combined_batch_"
        adata_obj.obs[batch_key] = adata_obj.obs[batch_cols].astype(str).agg("|".join, axis=1)

    if adata_obj.obs[batch_key].nunique() <= 1:
        if verbose:
            print("[INFO] ComBat skipped: only 1 batch.")
        return False

    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc.pp.combat(adata_obj, key=batch_key, covariates=covariates, inplace=True)
        if verbose:
            print("[INFO] ComBat done.")
        return True
    except Exception as e:
        if verbose:
            print(f"[WARN] ComBat failed: {type(e).__name__}: {e}")
        return False


def process_pseudobulk(
    pb: ad.AnnData,
    normalize: bool,
    log_transform: bool,
    use_gpu: bool,
    batch_col: Optional[Union[str, List[str]]],
    covariates: Optional[Union[str, List[str]]],
    verbose: bool
) -> ad.AnnData:
    # keep genes with signal in at least 1 sample
    sc.pp.filter_genes(pb, min_cells=1)

    if normalize:
        if use_gpu and GPU_AVAILABLE:
            clear_gpu_memory()
            to_gpu(pb)
            rsc.pp.normalize_total(pb, target_sum=1e6)
            if log_transform:
                rsc.pp.log1p(pb)
            to_cpu(pb)
            clear_gpu_memory()
        else:
            sc.pp.normalize_total(pb, target_sum=1e6)
            if log_transform:
                sc.pp.log1p(pb)

    if batch_col:
        cov_list = [c for c in _as_list(covariates) if c in pb.obs.columns]
        apply_combat(pb, batch_col=batch_col, covariates=cov_list, verbose=verbose)

    return pb


# =============================================================================
# Save
# =============================================================================

def save_one_group(
    pb: ad.AnnData,
    group_name: str,
    output_dir: str,
    prefix: str,
    verbose: bool
):
    g_dir = os.path.join(output_dir, _sanitize_name(group_name))
    os.makedirs(g_dir, exist_ok=True)

    h5ad_path = os.path.join(g_dir, f"{prefix}.h5ad")
    pb.write_h5ad(h5ad_path, compression="gzip")

    expr_path = os.path.join(g_dir, f"{prefix}_expression.csv")
    X = pb.X.toarray() if issparse(pb.X) else pb.X
    pd.DataFrame(X, index=pb.obs.index, columns=pb.var_names).to_csv(expr_path)

    meta_path = os.path.join(g_dir, f"{prefix}_metadata.csv")
    pb.obs.to_csv(meta_path)

    if verbose:
        print(f"[INFO] Saved: {h5ad_path}")
        print(f"[INFO] Saved: {expr_path}")
        print(f"[INFO] Saved: {meta_path}")


# =============================================================================
# Orchestrator
# =============================================================================

def compute_pseudobulk_per_group(
    adata_obj: ad.AnnData,
    sample_col: str,
    group_col: str,
    layer: Optional[str],
    output_dir: str,
    prefix: str,
    normalize: bool,
    log_transform: bool,
    batch_col: Optional[Union[str, List[str]]],
    covariates: Optional[Union[str, List[str]]],
    use_gpu: bool,
    verbose: bool,
    save_global_props_anndata: bool
):
    if sample_col not in adata_obj.obs.columns:
        raise KeyError(f"sample_col='{sample_col}' not found in adata.obs")
    if group_col not in adata_obj.obs.columns:
        raise KeyError(f"group_col='{group_col}' not found in adata.obs")

    if verbose:
        print(f"[INFO] Input: {adata_obj.n_obs} cells x {adata_obj.n_vars} genes")
        print(f"[INFO] sample_col={sample_col} | group_col={group_col} | layer={layer}")
        print(f"[INFO] Available layers: {list(adata_obj.layers.keys())}")

    if layer is None and verbose:
        print("[WARN] layer=None, so pseudobulk aggregation will use adata.X.")
        print("[WARN] Only do this if X truly contains raw counts.")

    # global group proportions (from cell-level)
    counts_df, props_df = compute_group_counts_and_proportions(adata_obj, sample_col, group_col)
    save_group_summary_tables(counts_df, props_df, output_dir, verbose=verbose)
    if save_global_props_anndata:
        save_group_proportions_anndata(props_df, output_dir, verbose=verbose)

    # pseudobulk per group
    pb_dict = aggregate_to_pseudobulk_per_group(
        adata_obj=adata_obj,
        sample_col=sample_col,
        group_col=group_col,
        layer=layer,
        verbose=verbose
    )

    os.makedirs(output_dir, exist_ok=True)

    for g, pb in pb_dict.items():
        pb = process_pseudobulk(
            pb,
            normalize=normalize,
            log_transform=log_transform,
            use_gpu=use_gpu,
            batch_col=batch_col,
            covariates=covariates,
            verbose=verbose
        )
        save_one_group(pb, g, output_dir, prefix, verbose)

    if verbose:
        print(f"[INFO] Done. Groups saved: {len(pb_dict)}")
        print(f"[INFO] Output: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="Input AnnData (.h5ad), e.g. processed_reintegrated.h5ad")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--sample_col", default="sample", help="obs column for samples")
    ap.add_argument("--group_col", default="leiden_sub", help="obs column for grouping (e.g. leiden_sub or cell_type)")
    ap.add_argument("--layer", default="counts", help="Layer to aggregate from (use 'counts' if present). Use 'X' to force .X.")
    ap.add_argument("--prefix", default="pseudobulk", help="Prefix for saved files")
    ap.add_argument("--normalize", action="store_true", help="Normalize to CPM (1e6)")
    ap.add_argument("--log1p", action="store_true", help="Apply log1p after normalization")
    ap.add_argument("--batch_col", default=None, help="Comma-separated obs columns for ComBat batch correction")
    ap.add_argument("--covariates", default=None, help="Comma-separated covariates to preserve in ComBat")
    ap.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save_global_props_anndata", action="store_true",
                    help="Also save group_proportions.h5ad (samples x groups)")

    args = ap.parse_args()

    layer = None
    if args.layer and args.layer != "X":
        layer = args.layer

    batch_col = args.batch_col.split(",") if args.batch_col else None
    covariates = args.covariates.split(",") if args.covariates else None

    if args.verbose:
        print("[INFO] Reading:", args.h5ad)

    adata_obj = sc.read_h5ad(args.h5ad)

    compute_pseudobulk_per_group(
        adata_obj=adata_obj,
        sample_col=args.sample_col,
        group_col=args.group_col,
        layer=layer,
        output_dir=args.out_dir,
        prefix=args.prefix,
        normalize=args.normalize,
        log_transform=args.log1p,
        batch_col=batch_col,
        covariates=covariates,
        use_gpu=args.use_gpu,
        verbose=args.verbose,
        save_global_props_anndata=args.save_global_props_anndata,
    )


if __name__ == "__main__":
    main()
