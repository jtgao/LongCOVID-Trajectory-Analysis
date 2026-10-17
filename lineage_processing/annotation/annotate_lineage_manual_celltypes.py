#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, Optional, List

import scanpy as sc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def first_existing(obs_cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in obs_cols:
            return c
    return None


def parse_mapping(mapping_str: str) -> Dict[str, str]:
    """
    Parse mapping string like:
    "0=naive_B;1=activated_B;11=T_contam;12=platelet_contam"
    """
    out: Dict[str, str] = {}
    if mapping_str is None or mapping_str.strip() == "":
        return out

    items = [x.strip() for x in mapping_str.split(";") if x.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad mapping item '{item}'. Expected format cluster=label")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def prettify_manual_label(label: str) -> str:
    """
    Convert internal manual labels into cleaner display labels for UMAPs.
    """
    if label is None:
        return "Unassigned"

    s = str(label)

    # basic formatting
    s = s.replace("_", " ")

    # common wording cleanup
    s = s.replace("CD8 NKlike", "CD8/NK-like")
    s = s.replace("TMIGD2 like T", "TMIGD2+ T")
    s = s.replace("TRAV8 like T", "TRAV8-like T")
    s = s.replace("naive like T", "naive-like T")
    s = s.replace("naive like T2", "naive-like T")
    s = s.replace("activated Treg or CD4", "activated Treg/CD4")
    s = s.replace("GZMK CD8 effector", "GZMK+ CD8 effector")
    s = s.replace("IFN stim T", "IFN-stimulated T")
    s = s.replace("cytotoxic T variant", "cytotoxic T variant")
    s = s.replace("cycling T", "cycling T")
    s = s.replace("myeloid contam", "myeloid contaminant")
    s = s.replace("RBC contam", "RBC contaminant")
    s = s.replace("platelet contam", "platelet contaminant")
    s = s.replace("T contam", "T contaminant")
    s = s.replace("platelet or myeloid suspect", "platelet/myeloid suspect")
    s = s.replace("nonclassical T suspect", "nonclassical T suspect")

    s = s.replace("naive B IGHV4", "naive B (IGHV4)")
    s = s.replace("B IGHV3 like", "B (IGHV3-like)")
    s = s.replace("B IGLL5 like", "B (IGLL5-like)")
    s = s.replace("light chain high B", "light-chain-high B")
    s = s.replace("activated B2", "activated B")
    s = s.replace("activated B or stress", "activated/stress B")
    s = s.replace("activated B suspect", "activated B suspect")
    s = s.replace("memory like B suspect", "memory-like B suspect")
    s = s.replace("APC like B suspect", "APC-like B suspect")
    s = s.replace("proliferative or regulatory B suspect", "proliferative/regulatory B suspect")

    # nicer capitalization for UNASSIGNED
    if s.upper() == "UNASSIGNED":
        s = "Unassigned"

    return s

def save_umap_obs(
    adata,
    key: str,
    out_png: str,
    title: str,
    on_data: bool = False,
):
    sc.pl.umap(
        adata,
        color=key,
        title=title,
        show=False,
        legend_loc="on data" if on_data else "right margin",
        legend_fontsize=4 if on_data else None,
        legend_fontweight="bold" if on_data else None,
        sort_order=True,
    )

    if on_data:
        ax = plt.gca()
        for txt in ax.texts:
            txt.set_color("white")
            txt.set_bbox(dict(
                facecolor="black",
                alpha=0.4,
                edgecolor="none",
                boxstyle="round,pad=0.15"
            ))

    ensure_dir(os.path.dirname(out_png))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_per_patient_umaps(
    adata,
    patient_key: str,
    out_dir: str,
    lineage_name: str,
    color_key: str = "manual_cell_type",
    max_patients: Optional[int] = None,
):
    if patient_key not in adata.obs.columns:
        print(f"[WARN] patient_key '{patient_key}' not found in {lineage_name}; skipping per-patient UMAPs.")
        return

    patients = adata.obs[patient_key].astype(str)
    uniq = sorted(patients.unique().tolist())
    if max_patients is not None:
        uniq = uniq[:max_patients]

    base = os.path.join(out_dir, lineage_name, "per_patient_manual_cell_type")
    ensure_dir(base)

    for pid in uniq:
        sub = adata[patients == pid].copy()
        if sub.n_obs == 0:
            continue
        if "X_umap" not in sub.obsm:
            print(f"[WARN] {lineage_name} patient={pid} missing X_umap; skipping.")
            continue
        if color_key not in sub.obs.columns:
            print(f"[WARN] {lineage_name} patient={pid} missing '{color_key}'; skipping.")
            continue

        pid_dir = os.path.join(base, f"patient_{pid}")
        ensure_dir(pid_dir)

        out_png = os.path.join(pid_dir, f"{lineage_name}_patient_{pid}_umap_by_{color_key}.png")
        sc.pl.umap(
            sub,
            color=color_key,
            title=f"{lineage_name} — patient {pid} — {color_key}",
            show=False,
            legend_loc="on data",
            legend_fontsize=5,
            legend_fontweight="bold",
            sort_order=True,
        )

        ax = plt.gca()
        for txt in ax.texts:
            txt.set_color("white")
            txt.set_bbox(dict(
                facecolor="black",
                alpha=0.4,
                edgecolor="none",
                boxstyle="round,pad=0.15"
            ))
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", out_png)


def annotate_one(
    h5ad_path: str,
    out_h5ad_path: str,
    out_dir: str,
    lineage_name: str,
    cluster_key: str,
    mapping: Dict[str, str],
    unknown_label: str = "UNASSIGNED",
    make_per_patient: bool = True,
    max_patients: Optional[int] = 20,
):
    print(f"[INFO] Reading {lineage_name}: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"{lineage_name}: missing cluster key '{cluster_key}' in obs")

    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)

    # map cluster -> manual cell type
    adata.obs["manual_cell_type"] = adata.obs[cluster_key].map(mapping).fillna(unknown_label)
    adata.obs["manual_cell_type"] = pd.Categorical(adata.obs["manual_cell_type"])

    # clean display labels for plotting / presentation
    adata.obs["manual_cell_type_display"] = (
        adata.obs["manual_cell_type"]
        .astype(str)
        .map(prettify_manual_label)
    )
    adata.obs["manual_cell_type_display"] = pd.Categorical(adata.obs["manual_cell_type_display"])

    # save updated h5ad
    print(f"[INFO] Writing annotated h5ad: {out_h5ad_path}")
    adata.write_h5ad(out_h5ad_path, compression="gzip")

    # save mapping table
    ensure_dir(out_dir)
    mapping_df = pd.DataFrame(
        [{"cluster": k, "manual_cell_type": v} for k, v in mapping.items()]
    ).sort_values("cluster")
    mapping_csv = os.path.join(out_dir, f"{lineage_name}_manual_cell_type_mapping.csv")
    mapping_df.to_csv(mapping_csv, index=False)
    print("[INFO] Wrote:", mapping_csv)

    # save cluster counts
    counts_df = (
        adata.obs.groupby([cluster_key, "manual_cell_type"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values([cluster_key, "n_cells"], ascending=[True, False])
    )
    counts_csv = os.path.join(out_dir, f"{lineage_name}_manual_cell_type_cluster_counts.csv")
    counts_df.to_csv(counts_csv, index=False)
    print("[INFO] Wrote:", counts_csv)

    lineage_dir = os.path.join(out_dir, lineage_name)
    ensure_dir(lineage_dir)

    # UMAP by manual cell type
    save_umap_obs(
        adata,
        key="manual_cell_type_display",
        out_png=os.path.join(lineage_dir, f"{lineage_name}_umap_by_manual_cell_type.png"),
        title=f"{lineage_name} UMAP — cell type",
        on_data=True,
    )
    print("[INFO] Wrote manual_cell_type UMAP for", lineage_name)

    # UMAP by cluster for reference
    save_umap_obs(
        adata,
        key=cluster_key,
        out_png=os.path.join(lineage_dir, f"{lineage_name}_umap_by_{cluster_key}.png"),
        title=f"{lineage_name} UMAP — {cluster_key}",
        on_data=True,
    )
    print("[INFO] Wrote cluster UMAP for", lineage_name)

    # Repair / create patient column if missing or blank
    needs_patient_fix = False
    if "patient" not in adata.obs.columns:
        needs_patient_fix = True
    else:
        patient_vals = adata.obs["patient"].astype(str).str.strip()
        if patient_vals.nunique() <= 1:
            needs_patient_fix = True

    if needs_patient_fix:
        if "filename_sample_id" in adata.obs.columns:
            adata.obs["patient"] = adata.obs["filename_sample_id"].astype(str)
        elif "sample" in adata.obs.columns:
            adata.obs["patient"] = adata.obs["sample"].astype(str).str.split("-").str[0]

    # per-patient UMAPs
    if make_per_patient:
        patient_key = first_existing(
            list(adata.obs.columns),
            ["patient", "Patient", "patient_id", "PatientID", "filename_sample_id", "donor", "donor_id", "subject", "subject_id"]
        )
        if patient_key is not None:
            save_per_patient_umaps(
                adata=adata,
                patient_key=patient_key,
                out_dir=out_dir,
                lineage_name=lineage_name,
                color_key="manual_cell_type_display",
                max_patients=max_patients,
            )
        else:
            print(f"[WARN] {lineage_name}: no patient-like column found; skipping per-patient UMAPs.")


def main():
    ap = argparse.ArgumentParser(description="Annotate reintegrated T/B lineage objects with manual cell type labels.")
    ap.add_argument("--t_h5ad", required=True)
    ap.add_argument("--b_h5ad", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cluster_key", default="leiden_sub")
    ap.add_argument("--t_mapping", required=True, help='Semicolon-separated cluster=label mapping for T')
    ap.add_argument("--b_mapping", required=True, help='Semicolon-separated cluster=label mapping for B')
    ap.add_argument("--unknown_label", default="UNASSIGNED")
    ap.add_argument("--max_patients", type=int, default=20)
    ap.add_argument("--per-patient-plots", action="store_true",
                    help="Opt in to plots labeled/named with patient metadata; keep outputs private.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    t_mapping = parse_mapping(args.t_mapping)
    b_mapping = parse_mapping(args.b_mapping)

    t_out_h5ad = os.path.join(
        os.path.dirname(args.t_h5ad),
        "processed_reintegrated_manual_annotated.h5ad"
    )
    b_out_h5ad = os.path.join(
        os.path.dirname(args.b_h5ad),
        "processed_reintegrated_manual_annotated.h5ad"
    )

    annotate_one(
        h5ad_path=args.t_h5ad,
        out_h5ad_path=t_out_h5ad,
        out_dir=args.out_dir,
        lineage_name="T_lineage",
        cluster_key=args.cluster_key,
        mapping=t_mapping,
        unknown_label=args.unknown_label,
        make_per_patient=args.per_patient_plots,
        max_patients=args.max_patients,
    )

    annotate_one(
        h5ad_path=args.b_h5ad,
        out_h5ad_path=b_out_h5ad,
        out_dir=args.out_dir,
        lineage_name="B_lineage",
        cluster_key=args.cluster_key,
        mapping=b_mapping,
        unknown_label=args.unknown_label,
        make_per_patient=args.per_patient_plots,
        max_patients=args.max_patients,
    )

    print("[INFO] DONE.")


if __name__ == "__main__":
    main()
