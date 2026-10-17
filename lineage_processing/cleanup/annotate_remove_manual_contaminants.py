#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import scanpy as sc
import pandas as pd


def parse_labels(s: str):
    return [x.strip() for x in s.split(";") if x.strip()]


def parse_mapping(s: str):
    """
    Parse mapping like:
    'old1=new1;old2=new2'
    """
    mapping = {}
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad relabel item (missing '='): {item}")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def relabel_and_remove_one(h5ad_path: str, out_h5ad_path: str, label_col: str, relabel_map, remove_labels):
    print(f"[INFO] Reading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    if label_col not in adata.obs.columns:
        raise KeyError(f"Missing label column '{label_col}' in {h5ad_path}")

    labels = adata.obs[label_col].astype(str).copy()

    print(f"[INFO] Label column: {label_col}")

    # --- relabel first ---
    if relabel_map:
        print("[INFO] Relabel map:")
        for old, new in relabel_map.items():
            print(f"    {old}  -->  {new}")
        labels = labels.replace(relabel_map)

    # write relabeled labels back
    adata.obs[label_col] = labels

    # if display column exists, keep it aligned too
    if "manual_cell_type_display" in adata.obs.columns:
        disp = adata.obs["manual_cell_type_display"].astype(str).copy()
        if relabel_map:
            disp = disp.replace(relabel_map)
        adata.obs["manual_cell_type_display"] = disp

    # --- remove contaminants ---
    remove_mask = adata.obs[label_col].astype(str).isin(remove_labels)
    before = adata.n_obs
    kept = adata[~remove_mask].copy()

    print(f"[INFO] Remove labels: {remove_labels}")
    print(f"[INFO] Removed cells: {int(remove_mask.sum()):,}")
    print(f"[INFO] Cells before: {before:,}")
    print(f"[INFO] Cells after:  {kept.n_obs:,}")

    if int(remove_mask.sum()) > 0:
        print("[INFO] Removed label counts:")
        print(adata.obs.loc[remove_mask, label_col].value_counts())

    print("[INFO] Final label counts (top 30):")
    print(kept.obs[label_col].value_counts().head(30))

    os.makedirs(os.path.dirname(out_h5ad_path), exist_ok=True)
    kept.write_h5ad(out_h5ad_path, compression="gzip")
    print(f"[INFO] Wrote: {out_h5ad_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Relabel close-enough manual cell types and remove contaminant labels."
    )
    ap.add_argument("--t_h5ad", required=True)
    ap.add_argument("--b_h5ad", required=True)
    ap.add_argument("--label_col", default="manual_cell_type")
    ap.add_argument("--t_relabel", default="")
    ap.add_argument("--b_relabel", default="")
    ap.add_argument("--t_remove", required=True, help="Semicolon-separated labels to remove from T")
    ap.add_argument("--b_remove", required=True, help="Semicolon-separated labels to remove from B")
    args = ap.parse_args()

    t_relabel = parse_mapping(args.t_relabel) if args.t_relabel.strip() else {}
    b_relabel = parse_mapping(args.b_relabel) if args.b_relabel.strip() else {}

    t_remove = parse_labels(args.t_remove)
    b_remove = parse_labels(args.b_remove)

    t_out = os.path.join(os.path.dirname(args.t_h5ad), "processed_reintegrated_manual_annotated_clean.h5ad")
    b_out = os.path.join(os.path.dirname(args.b_h5ad), "processed_reintegrated_manual_annotated_clean.h5ad")

    relabel_and_remove_one(
        h5ad_path=args.t_h5ad,
        out_h5ad_path=t_out,
        label_col=args.label_col,
        relabel_map=t_relabel,
        remove_labels=t_remove,
    )

    relabel_and_remove_one(
        h5ad_path=args.b_h5ad,
        out_h5ad_path=b_out,
        label_col=args.label_col,
        relabel_map=b_relabel,
        remove_labels=b_remove,
    )

    print("[INFO] DONE.")


if __name__ == "__main__":
    main()