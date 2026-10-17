#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_h5ad", required=True)
    parser.add_argument("--raw_h5_dir", required=True)
    parser.add_argument("--out_h5ad", required=True)
    args = parser.parse_args()

    # 1. Load your clean RNA data
    adata = sc.read_h5ad(args.rna_h5ad)
    print(f"Loaded RNA data with {adata.n_obs} cells.")

    # Get unique sample IDs from your metadata
    samples = adata.obs['sample'].unique()
    print(f"Samples found in RNA: {samples}")

    all_protein_data = []

    for s in samples:
        # Input convention: <sample>_sample_filtered_feature_bc_matrix.h5.
        # Cell IDs must use <sample>_<barcode>; alignment below preserves this convention.
        h5_path = os.path.join(args.raw_h5_dir, f"{s}_sample_filtered_feature_bc_matrix.h5")
        
        if not os.path.exists(h5_path):
            print(f"[WARN] Missing .h5 for sample {s} at {h5_path}")
            continue

        print(f"[INFO] Processing sample {s}...")
        raw_data = sc.read_10x_h5(h5_path, gex_only=False)
        protein_only = raw_data[:, raw_data.var['feature_types'] == 'Antibody Capture'].to_df()
        
        # KEY FIX: Match the prefixing logic
        # If RNA barcodes are 'SampleID_Barcode-1', we prefix the raw protein barcodes to match
        protein_only.index = f"{s}_" + protein_only.index
        
        # Keep only the cells that exist in your clean RNA object
        overlap_cells = list(set(protein_only.index) & set(adata.obs_names))
        all_protein_data.append(protein_only.loc[overlap_cells])

    # 3. Combine and Add to ADATA
    full_protein_df = pd.concat(all_protein_data)
    
    # Fill missing cells with 0 (if any cells were lost during QC)
    full_protein_df = full_protein_df.reindex(adata.obs_names).fillna(0)
    
    # Store in .obsm for easy access in heatmaps
    adata.obsm['protein'] = full_protein_df
    
    # 4. Save
    adata.write_h5ad(args.out_h5ad)
    print(f"[SUCCESS] Saved merged object to {args.out_h5ad}")

if __name__ == "__main__":
    main()
