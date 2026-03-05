#!/usr/bin/env python
"""
preprocess_new_holdout.py  (scArches_env)

t=0.165 holdout + cap 10,000 cells per timepoint 전처리.
이미 X_scVI(dim30) 인코딩이 완료된 h5ad를 재사용한다.

Output:
  results/new_run/train_cap10k_holdout0165.h5ad   - scDiffeq 학습용
  results/new_run/holdout_t0165.h5ad              - 평가용 (t=0.165 cells)
"""

import os
import sys
import numpy as np
import anndata as ad
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
INPUT_H5AD = BASE_DIR / "results" / "Input_fetal_neuron_trainset_after_scVI_hvg_5010_latent_dim30.h5ad"
OUT_DIR    = BASE_DIR / "results" / "new_run"

HOLDOUT_TIMEPOINT = 0.165423   # 정확한 값
CAP_PER_TIMEPOINT = 10_000
SEED = 42

# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 70)
    print("Preprocessing: holdout t=0.165, cap 10k/timepoint")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading: {INPUT_H5AD}")
    adata = ad.read_h5ad(INPUT_H5AD)
    print(f"    Total cells: {adata.shape[0]:,}  genes: {adata.shape[1]:,}")
    print(f"    obsm keys  : {list(adata.obsm.keys())}")

    # X_scVI 확인
    if 'X_scVI' not in adata.obsm:
        print("ERROR: X_scVI not found in obsm. Run scVI encoding first.")
        sys.exit(1)
    print(f"    X_scVI dim : {adata.obsm['X_scVI'].shape}")

    # 타임포인트 분포
    time_col = 'age_time_norm'
    tp_counts = adata.obs[time_col].value_counts().sort_index()
    print(f"\n    Timepoint distribution:")
    for t, n in tp_counts.items():
        marker = "  ← HOLDOUT" if abs(t - HOLDOUT_TIMEPOINT) < 1e-4 else ""
        print(f"      t={t:.6f}  {n:6,} cells{marker}")

    # ------------------------------------------------------------------
    # 2. Split holdout
    # ------------------------------------------------------------------
    print(f"\n[2] Splitting holdout (t={HOLDOUT_TIMEPOINT})")
    mask_holdout = (adata.obs[time_col] - HOLDOUT_TIMEPOINT).abs() < 1e-4
    adata_holdout = adata[mask_holdout].copy()
    adata_train   = adata[~mask_holdout].copy()
    print(f"    Holdout  : {adata_holdout.shape[0]:,} cells")
    print(f"    Remaining: {adata_train.shape[0]:,} cells")

    # ------------------------------------------------------------------
    # 3. Cap 10k per timepoint (train set only)
    # ------------------------------------------------------------------
    print(f"\n[3] Capping at {CAP_PER_TIMEPOINT:,} cells per timepoint")
    keep_idx = []
    for t, grp in adata_train.obs.groupby(time_col, sort=True):
        idx = grp.index.tolist()
        if len(idx) > CAP_PER_TIMEPOINT:
            idx = np.random.choice(idx, CAP_PER_TIMEPOINT, replace=False).tolist()
            print(f"    t={t:.6f}  {len(grp):6,} → {CAP_PER_TIMEPOINT:,} (sampled)")
        else:
            print(f"    t={t:.6f}  {len(grp):6,} (kept all)")
        keep_idx.extend(idx)

    adata_train_cap = adata_train[keep_idx].copy()
    print(f"\n    Train (capped): {adata_train_cap.shape[0]:,} cells")

    # ------------------------------------------------------------------
    # 4. Reset obs_names (scDiffeq requires integer-like unique index)
    # ------------------------------------------------------------------
    print("\n[4] Resetting obs_names for scDiffeq compatibility")
    adata_train_cap.obs['original_barcode'] = adata_train_cap.obs_names.copy()
    adata_train_cap.obs_names = [str(i) for i in range(adata_train_cap.shape[0])]

    adata_holdout.obs['original_barcode'] = adata_holdout.obs_names.copy()
    adata_holdout.obs_names = [str(i) for i in range(adata_holdout.shape[0])]

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n[5] Final dataset summary")
    print(f"    Train (capped)  : {adata_train_cap.shape[0]:,} cells")
    tp_final = adata_train_cap.obs[time_col].value_counts().sort_index()
    for t, n in tp_final.items():
        print(f"      t={t:.6f}  {n:6,}")

    print(f"\n    Holdout (t=0.165): {adata_holdout.shape[0]:,} cells")
    ct_holdout = adata_holdout.obs['cluster_annotated'].value_counts()
    for ct, n in ct_holdout.items():
        print(f"      {ct}: {n}")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    train_out   = OUT_DIR / "train_cap10k_holdout0165.h5ad"
    holdout_out = OUT_DIR / "holdout_t0165.h5ad"

    print(f"\n[6] Saving")
    print(f"    Train  → {train_out}")
    adata_train_cap.write_h5ad(train_out)
    print(f"    Holdout→ {holdout_out}")
    adata_holdout.write_h5ad(holdout_out)

    print("\n[Done] Preprocessing complete.")
    print(f"  Train file : {train_out}")
    print(f"  Holdout file: {holdout_out}")


if __name__ == "__main__":
    main()
