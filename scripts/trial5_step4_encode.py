#!/usr/bin/env python
"""
trial5_step4_encode.py  (scArches_env)

Trial5 scVI dim=30 모델로 퍼터베이션 latent 인코딩.
t=0 (49d, age_time_norm=0.0) 세포에 KO/OE 적용 후 인코딩.

Output: results/trial5/perturb_latents/
"""

import os, warnings, json
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5  = BASE / "results" / "trial5"

SCVI_MODEL = TRIAL5 / "scvi_dim30"
HVG_JSON   = TRIAL5 / "hvg_A.json"
RAW_PATH   = TRIAL5 / "trial5_train_cap10k.h5ad"
TIME_COL   = "age_time_norm"
BATCH_KEY  = "batch"

PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--oe_factor", type=float, default=3.0)
    p.add_argument("--t0",        type=float, default=0.0,  help="start timepoint (age_time_norm)")
    p.add_argument("--t0_tag",    type=str,   default="t0", help="tag for output folder")
    p.add_argument("--cell_type", type=str,   default=None, help="filter start cells by CellType_refine (e.g. RG)")
    p.add_argument("--base_dir",  type=str,   default=str(BASE / "results" / "trial5"),
                   help="base results directory (e.g. .../trial6)")
    return p.parse_args()

def main():
    args = parse_args()
    OE_FACTOR = args.oe_factor
    oe_suffix = f"OE{int(OE_FACTOR)}x"
    T0        = args.t0
    t0_tag    = args.t0_tag
    print("=" * 60)
    print(f"Trial5 Step 4: Perturbation Encoding  (OE={OE_FACTOR}x, t0={T0}, tag={t0_tag})")
    print("=" * 60)

    TRIAL_DIR = Path(args.base_dir)
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = TRIAL_DIR / f"perturb_latents_{t0_tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import scvi
    scvi.settings.seed = 42

    # ── 1. HVG 목록 로드 & 데이터 준비 ───────────────────────────
    print("\n[1] Loading data and HVG list...")
    with open(HVG_JSON) as f:
        hvg_genes = json.load(f)
    print(f"  HVG: {len(hvg_genes):,}  |  IL17RD: {'IL17RD' in hvg_genes}")

    adata_full = ad.read_h5ad(RAW_PATH)
    print(f"  Full data: {adata_full.shape}")

    # raw counts → X
    adata_full.X = adata_full.layers['counts'].copy()
    adata_hvg = adata_full[:, hvg_genes].copy()
    print(f"  HVG subset: {adata_hvg.shape}")

    # ── 2. scVI 로드 ──────────────────────────────────────────────
    print("\n[2] Loading scVI model...")
    scvi.model.SCVI.setup_anndata(adata_hvg, batch_key=BATCH_KEY)
    vae = scvi.model.SCVI.load(str(SCVI_MODEL), adata=adata_hvg)
    print("  scVI model loaded.")

    # ── 3. 시작 시점 세포 추출 ────────────────────────────────────
    print(f"\n[3] Extracting t={T0} cells (tag={t0_tag})...")
    mask_t0 = (adata_hvg.obs[TIME_COL] - T0).abs() < 1e-4
    adata_t0 = adata_hvg[mask_t0].copy()
    print(f"  t={T0} cells (all): {adata_t0.n_obs:,}")
    print(f"  Cell types:\n{adata_t0.obs['CellType_refine'].value_counts().to_string()}")

    if args.cell_type:
        ct_mask  = adata_t0.obs['CellType_refine'] == args.cell_type
        adata_t0 = adata_t0[ct_mask].copy()
        print(f"\n  → Filtered to '{args.cell_type}': {adata_t0.n_obs:,} cells")

    counts = adata_t0.X.toarray() if sp.issparse(adata_t0.X) else np.array(adata_t0.X)
    gene_names = adata_t0.var_names.tolist()
    print(f"  Count matrix: {counts.shape}")

    adata_t0.obs.to_csv(OUT_DIR / "start_cells_obs.csv")
    print("  obs saved.")

    def encode(counts_mat):
        """counts_mat [N, n_hvg] → scVI → latent [N, 30]"""
        adata_q = adata_t0.copy()
        adata_q.X = sp.csr_matrix(counts_mat.astype(np.float32))
        latent = vae.get_latent_representation(adata_q)
        return latent

    # ── 4. Control 인코딩 ─────────────────────────────────────────
    print("\n[4] Encoding control...")
    z_ctrl = encode(counts)
    np.save(OUT_DIR / "z_ctrl.npy", z_ctrl)
    print(f"  z_ctrl: {z_ctrl.shape}  saved.")

    # ── 5. 각 유전자 KO / OE ─────────────────────────────────────
    results = {"ctrl": z_ctrl}

    for gene in PERTURB_GENES:
        if gene not in gene_names:
            print(f"\n  WARNING: {gene} not in HVG, skipping.")
            continue

        gene_idx = gene_names.index(gene)
        orig_mean = counts[:, gene_idx].mean()

        # KO
        print(f"\n[5] {gene} KO...")
        counts_ko = counts.copy(); counts_ko[:, gene_idx] = 0
        z_ko = encode(counts_ko)
        np.save(OUT_DIR / f"z_{gene}_KO.npy",     z_ko)
        shift_ko = np.abs(z_ko - z_ctrl).mean()
        print(f"  z_{gene}_KO: {z_ko.shape}  |  mean expr {orig_mean:.2f} → 0  |  latent shift={shift_ko:.4f}")
        results[f"{gene}_KO"] = z_ko

        # OE
        print(f"\n[5] {gene} OE ({OE_FACTOR}x)...")
        counts_oe = counts.copy(); counts_oe[:, gene_idx] = counts[:, gene_idx] * OE_FACTOR
        z_oe = encode(counts_oe)
        np.save(OUT_DIR / f"z_{gene}_{oe_suffix}.npy", z_oe)
        shift_oe = np.abs(z_oe - z_ctrl).mean()
        print(f"  z_{gene}_OE: {z_oe.shape}  |  mean expr {orig_mean:.2f} → {orig_mean*OE_FACTOR:.2f}  |  latent shift={shift_oe:.4f}")
        results[f"{gene}_OE"] = z_oe

    # ── 6. 요약 ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Latent shift summary (|mean| vs ctrl)")
    print(f"  {'Condition':<20} {'mean_shift':>12} {'max_shift':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
    for cond, z in results.items():
        if cond == 'ctrl': continue
        diff = np.abs(z - z_ctrl)
        print(f"  {cond:<20}  {diff.mean():>12.4f}  {diff.max():>12.4f}")

    print(f"\n  Saved to: {OUT_DIR}")
    print(f"  Files: {[f.name for f in sorted(OUT_DIR.glob('*.npy'))]}")

if __name__ == "__main__":
    main()
