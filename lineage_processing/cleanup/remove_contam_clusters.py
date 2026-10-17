#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import scanpy as sc


def parse_cluster_list(s: str):
    s = s.strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def remove_clusters(adata, cluster_key: str, remove_clusters_list):
    before = adata.n_obs
    mask_keep = ~adata.obs[cluster_key].astype(str).isin(remove_clusters_list)
    adata_clean = adata[mask_keep].copy()
    after = adata_clean.n_obs
    removed = before - after
    return adata_clean, before, after, removed


def main():
    ap = argparse.ArgumentParser(description="Remove specified contaminated subclusters from T/B lineage h5ad files.")
    ap.add_argument("--t_in", required=True, help="Input T lineage h5ad")
    ap.add_argument("--b_in", required=True, help="Input B lineage h5ad")
    ap.add_argument("--t_out", required=True, help="Output cleaned T lineage h5ad")
    ap.add_argument("--b_out", required=True, help="Output cleaned B lineage h5ad")
    ap.add_argument("--cluster_key", default="leiden_sub", help="obs column containing cluster labels")
    ap.add_argument("--t_remove", default="", help="Comma-separated T clusters to remove, e.g. 16,19")
    ap.add_argument("--b_remove", default="", help="Comma-separated B clusters to remove, e.g. 15,17")
    args = ap.parse_args()

    t_remove = parse_cluster_list(args.t_remove)
    b_remove = parse_cluster_list(args.b_remove)

    print("[INFO] Reading T:", args.t_in)
    adata_t = sc.read_h5ad(args.t_in)
    print("[INFO] Reading B:", args.b_in)
    adata_b = sc.read_h5ad(args.b_in)

    if args.cluster_key not in adata_t.obs.columns:
        raise ValueError(f"T lineage missing cluster key '{args.cluster_key}'")
    if args.cluster_key not in adata_b.obs.columns:
        raise ValueError(f"B lineage missing cluster key '{args.cluster_key}'")

    print("[INFO] T clusters to remove:", t_remove)
    adata_t_clean, t_before, t_after, t_removed = remove_clusters(
        adata_t, args.cluster_key, t_remove
    )

    print("[INFO] B clusters to remove:", b_remove)
    adata_b_clean, b_before, b_after, b_removed = remove_clusters(
        adata_b, args.cluster_key, b_remove
    )

    adata_t_clean.write_h5ad(args.t_out, compression="gzip")
    adata_b_clean.write_h5ad(args.b_out, compression="gzip")

    print("\n=== SUMMARY ===")
    print(f"T before:   {t_before}")
    print(f"T removed:  {t_removed}")
    print(f"T after:    {t_after}")
    print(f"T wrote:    {args.t_out}")

    print(f"B before:   {b_before}")
    print(f"B removed:  {b_removed}")
    print(f"B after:    {b_after}")
    print(f"B wrote:    {args.b_out}")


if __name__ == "__main__":
    main()