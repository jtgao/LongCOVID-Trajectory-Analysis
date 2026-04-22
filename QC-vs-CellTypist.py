#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_celltypist_vs_markers.py

Goal (Yi/Ji sanity check):
1) Cross-tab: leiden_sub (recluster) vs CellTypist predicted labels
2) CellTypist confidence summary per cluster
3) Marker genes per cluster (Scanpy rank_genes_groups, Wilcoxon)
4) Optional: export VDJ/TCR/BCR-ish columns if present (CDR3 / chain / V/J genes etc.)

Usage example:
python /dcs07/antar/data/analysis/long_covid_clonal/scripts_cluster/qc_celltypist_vs_markers.py \
  --h5ad /dcs07/antar/data/analysis/long_covid_clonal/lineage_B_cluster/processed_reintegrated.h5ad \
  --out_dir /dcs07/antar/data/analysis/long_covid_clonal/figures/qc_celltypist_vs_markers/B_lineage \
  --cluster_key leiden_sub \
  --celltypist_label_key auto \
  --celltypist_conf_key celltypist_conf_score \
  --top_n_markers 50

Notes:
- Assumes adata.X already log-normalized (your pipeline does that).
- Does NOT re-run CellTypist. It *evaluates* and exports QC tables + markers.
"""

import os
import re
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


# -----------------------------
# helpers
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _guess_celltypist_label_key(obs_cols: List[str]) -> Optional[str]:
    """
    Try to find the CellTypist predicted labels column in adata.obs.
    Your pipeline may store it with lineage-specific naming.
    """
    # common direct hits
    common = [
        "celltypist", "celltypist_label", "celltypist_labels",
        "celltypist_predicted_labels", "celltypist_prediction",
        "celltypist_majority_voting", "celltypist_majority_vote",
    ]
    for c in common:
        if c in obs_cols:
            return c

    # search by pattern: contains celltypist and label/pred/majority
    cands = []
    for c in obs_cols:
        cl = c.lower()
        if "celltypist" in cl and any(k in cl for k in ["label", "pred", "major", "vote"]):
            cands.append(c)

    if len(cands) == 0:
        return None

    # prefer "predicted_labels" if exists
    for pref in cands:
        if "predicted" in pref.lower() and "label" in pref.lower():
            return pref

    # else first candidate
    return cands[0]


def _infer_lineage_name_from_path(h5ad_path: str) -> str:
    p = h5ad_path.lower()
    if "lineage_t" in p or "/t_" in p:
        return "T_lineage"
    if "lineage_b" in p or "/b_" in p:
        return "B_lineage"
    return "lineage"


def _detect_vdj_cols(obs_cols: List[str]) -> List[str]:
    """
    Best-effort: columns that look like VDJ/TCR/BCR annotations.
    You can refine later once you confirm exact column names.
    """
    keys = ["cdr3", "tra", "trb", "trd", "trg", "igh", "igk", "igl", "v_gene", "j_gene", "d_gene", "chain"]
    out = []
    for c in obs_cols:
        cl = c.lower()
        if any(k in cl for k in keys):
            out.append(c)
    return out


def _safe_crosstab(a: pd.Series, b: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tab = pd.crosstab(a.astype(str), b.astype(str))
    tab_prop = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return tab, tab_prop


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="QC: CellTypist vs reclustering + marker genes export.")
    ap.add_argument("--h5ad", required=True, help="Input reintegrated lineage .h5ad (e.g., processed_reintegrated.h5ad)")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSVs and summaries")
    ap.add_argument("--cluster_key", default="leiden_sub", help="Cluster key in adata.obs (default: leiden_sub)")

    ap.add_argument("--celltypist_label_key", default="auto",
                    help="CellTypist label column in adata.obs; 'auto' to guess")
    ap.add_argument("--celltypist_conf_key", default="celltypist_conf_score",
                    help="CellTypist confidence score column in adata.obs (default: celltypist_conf_score)")

    ap.add_argument("--top_n_markers", type=int, default=50, help="Top N markers per cluster to export (default 50)")
    ap.add_argument("--marker_method", default="wilcoxon", choices=["wilcoxon", "t-test", "logreg"],
                    help="scanpy rank_genes_groups method (default wilcoxon)")

    ap.add_argument("--export_vdj", action="store_true",
                    help="If set, export any obs columns that look like VDJ/TCR/BCR fields.")
    ap.add_argument("--vdj_max_rows", type=int, default=200000,
                    help="Cap rows for VDJ export (default 200000)")

    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    lineage_name = _infer_lineage_name_from_path(args.h5ad)

    print("[INFO] Reading:", args.h5ad)
    adata = sc.read_h5ad(args.h5ad)

    # basic checks
    if args.cluster_key not in adata.obs.columns:
        raise KeyError(f"Missing cluster_key '{args.cluster_key}' in adata.obs. Available keys include: "
                       f"{list(adata.obs.columns)[:50]} ...")

    obs_cols = list(adata.obs.columns)

    # resolve celltypist label column
    if args.celltypist_label_key.lower() == "auto":
        ct_label = _guess_celltypist_label_key(obs_cols)
    else:
        ct_label = args.celltypist_label_key

    if ct_label is None or ct_label not in adata.obs.columns:
        print("[WARN] Could not resolve CellTypist label column.")
        print("[WARN] obs columns that contain 'celltypist':",
              [c for c in obs_cols if "celltypist" in c.lower()])
        raise RuntimeError(
            "No CellTypist label column found. "
            "Re-run with --celltypist_label_key <your_column_name>."
        )

    ct_conf = args.celltypist_conf_key if args.celltypist_conf_key in obs_cols else None
    if ct_conf is None:
        print(f"[WARN] celltypist_conf_key '{args.celltypist_conf_key}' not found; confidence QC will be skipped.")

    print(f"[INFO] Using cluster_key='{args.cluster_key}'")
    print(f"[INFO] Using celltypist_label_key='{ct_label}'")
    if ct_conf:
        print(f"[INFO] Using celltypist_conf_key='{ct_conf}'")

    # -----------------------------
    # 1) crosstab cluster vs celltypist
    # -----------------------------
    tab, tab_prop = _safe_crosstab(adata.obs[args.cluster_key], adata.obs[ct_label])

    out_counts = os.path.join(args.out_dir, f"{lineage_name}_crosstab_{args.cluster_key}_vs_{ct_label}_counts.csv")
    out_props = os.path.join(args.out_dir, f"{lineage_name}_crosstab_{args.cluster_key}_vs_{ct_label}_rowprops.csv")
    tab.to_csv(out_counts)
    tab_prop.to_csv(out_props)
    print("[INFO] Wrote:", out_counts)
    print("[INFO] Wrote:", out_props)

    # -----------------------------
    # 2) confidence summary per cluster
    # -----------------------------
    if ct_conf:
        conf = (
            adata.obs
            .groupby(adata.obs[args.cluster_key].astype(str))[ct_conf]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
            .rename(columns={args.cluster_key: "cluster"})
        )
        out_conf = os.path.join(args.out_dir, f"{lineage_name}_celltypist_conf_by_{args.cluster_key}.csv")
        conf.to_csv(out_conf, index=False)
        print("[INFO] Wrote:", out_conf)

    # -----------------------------
    # 3) marker genes per cluster (Scanpy)
    # -----------------------------
    print(f"[INFO] rank_genes_groups: method={args.marker_method}, groupby={args.cluster_key}")
    sc.tl.rank_genes_groups(adata, groupby=args.cluster_key, method=args.marker_method)

    rg = adata.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names

    rows = []
    topn = int(args.top_n_markers)

    for g in groups:
        names = rg["names"][g][:topn]
        pvals = rg["pvals_adj"][g][:topn] if "pvals_adj" in rg else [np.nan] * topn
        scores = rg["scores"][g][:topn] if "scores" in rg else [np.nan] * topn
        logfc = rg["logfoldchanges"][g][:topn] if "logfoldchanges" in rg else [np.nan] * topn

        for i in range(len(names)):
            rows.append({
                "cluster": str(g),
                "rank": i + 1,
                "gene": str(names[i]),
                "score": float(scores[i]) if scores[i] == scores[i] else np.nan,
                "logFC": float(logfc[i]) if logfc[i] == logfc[i] else np.nan,
                "pvals_adj": float(pvals[i]) if pvals[i] == pvals[i] else np.nan,
            })

    markers = pd.DataFrame(rows)
    out_markers = os.path.join(args.out_dir, f"{lineage_name}_markers_top{topn}_by_{args.cluster_key}.csv")
    markers.to_csv(out_markers, index=False)
    print("[INFO] Wrote:", out_markers)

    # -----------------------------
    # 4) optional VDJ export
    # -----------------------------
    if args.export_vdj:
        vdj_cols = _detect_vdj_cols(obs_cols)
        if len(vdj_cols) == 0:
            print("[WARN] No VDJ-like columns detected in adata.obs; skipping VDJ export.")
        else:
            keep = [args.cluster_key, ct_label]
            if ct_conf:
                keep.append(ct_conf)
            # also keep sample/patient if present
            for k in ["sample", "patient", "donor", "subject", "timepoint", "time_month"]:
                if k in obs_cols and k not in keep:
                    keep.append(k)
            keep += [c for c in vdj_cols if c not in keep]

            df = adata.obs[keep].copy()
            if df.shape[0] > args.vdj_max_rows:
                df = df.sample(args.vdj_max_rows, random_state=0)

            out_vdj = os.path.join(args.out_dir, f"{lineage_name}_vdj_fields_subset.csv")
            df.to_csv(out_vdj, index=False)
            print("[INFO] Wrote:", out_vdj)
            print("[INFO] VDJ-like columns exported:", vdj_cols)

    # quick human-readable summary
    summary_txt = os.path.join(args.out_dir, f"{lineage_name}_qc_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Lineage: {lineage_name}\n")
        f.write(f"Input: {args.h5ad}\n")
        f.write(f"cluster_key: {args.cluster_key}\n")
        f.write(f"celltypist_label_key: {ct_label}\n")
        f.write(f"celltypist_conf_key: {ct_conf or 'NOT_FOUND'}\n")
        f.write(f"Markers: method={args.marker_method}, top_n={topn}\n\n")
        f.write("Crosstab outputs:\n")
        f.write(f" - {out_counts}\n")
        f.write(f" - {out_props}\n")
        if ct_conf:
            f.write(f" - {os.path.join(args.out_dir, f'{lineage_name}_celltypist_conf_by_{args.cluster_key}.csv')}\n")
        f.write(f"Markers output:\n - {out_markers}\n")
    print("[INFO] Wrote:", summary_txt)

    print("[INFO] DONE.")


if __name__ == "__main__":
    main()
