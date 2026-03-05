#!/usr/bin/env python
"""
trial6_step4_encode.py  (scArches_env)

Trial6: RG-only start cells at later timepoints.
  --t0_tag t70d_RG   → t=0.2471 (70d), RG only
  --t0_tag t115d_RG  → t=0.5588 (115d), RG only

Output: results/trial6/perturb_latents_{t0_tag}/
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

BASE      = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5    = BASE / "results" / "trial5"   # scVI model & raw data
TRIAL6    = BASE / "results" / "trial6"
TRIAL6.mkdir(parents=True, exist_ok=True)

SCVI_MODEL    = TRIAL5 / "scvi_dim30"
HVG_JSON      = TRIAL5 / "hvg_A.json"
RAW_PATH      = TRIAL5 / "trial5_train_cap10k.h5ad"
TIME_COL      = "age_time_norm"
CELLTYPE_COL  = "CellType_refine"
BATCH_KEY     = "batch"

PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

T0_OPTIONS = {
    "t70d_RG":  (0.2471, "RG"),
    "t115d_RG": (0.5588, "RG"),
}

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--t0_tag",    type=str,   required=True,
                   choices=list(T0_OPTIONS.keys()),
                   help="which timepoint/celltype config to use")
    p.add_argument("--oe_factor", type=float, default=3.0)
    return p.parse_args()

def main():
    args      = parse_args()
    t0, cell_type = T0_OPTIONS[args.t0_tag]
    OE_FACTOR = args.oe_factor
    oe_suffix = f"OE{int(OE_FACTOR)}x"
    OUT_DIR   = TRIAL6 / f"perturb_latents_{args.t0_tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Trial6 Step 4: Perturbation Encoding")
    print(f"  t0={t0}, cell_type={cell_type}, OE={OE_FACTOR}x")
    print(f"  out: {OUT_DIR}")
    print("=" * 60)

    import scvi
    scvi.settings.seed = 42

    # ── 1. HVG & 데이터 로드 ────────────────────────────────────
    print("\n[1] Loading data and HVG list...")
    with open(HVG_JSON) as f:
        hvg_genes = json.load(f)
    print(f"  HVG: {len(hvg_genes):,}  |  IL17RD in HVG: {'IL17RD' in hvg_genes}")

    adata_full = ad.read_h5ad(RAW_PATH)
    adata_full.X = adata_full.layers['counts'].copy()
    adata_hvg = adata_full[:, hvg_genes].copy()
    print(f"  HVG subset: {adata_hvg.shape}")

    # ── 2. scVI 로드 ─────────────────────────────────────────────
    print("\n[2] Loading scVI model...")
    scvi.model.SCVI.setup_anndata(adata_hvg, batch_key=BATCH_KEY)
    vae = scvi.model.SCVI.load(str(SCVI_MODEL), adata=adata_hvg)
    print("  scVI model loaded.")

    # ── 3. 시작 시점 RG 세포 추출 ───────────────────────────────
    print(f"\n[3] Extracting t={t0} cells (cell_type={cell_type})...")
    mask_t  = (adata_hvg.obs[TIME_COL] - t0).abs() < 1e-4
    adata_t = adata_hvg[mask_t].copy()
    print(f"  All cells at t={t0}: {adata_t.n_obs:,}")
    print(f"  {adata_t.obs[CELLTYPE_COL].value_counts().to_string()}")

    mask_ct  = adata_t.obs[CELLTYPE_COL] == cell_type
    adata_t0 = adata_t[mask_ct].copy()
    print(f"\n  → {cell_type} only: {adata_t0.n_obs:,} cells")

    counts     = adata_t0.X.toarray() if sp.issparse(adata_t0.X) else np.array(adata_t0.X)
    gene_names = adata_t0.var_names.tolist()
    adata_t0.obs.to_csv(OUT_DIR / "start_cells_obs.csv")

    def encode(counts_mat):
        adata_q   = adata_t0.copy()
        adata_q.X = sp.csr_matrix(counts_mat.astype(np.float32))
        return vae.get_latent_representation(adata_q)

    # ── 4. Control ───────────────────────────────────────────────
    print("\n[4] Encoding control...")
    z_ctrl = encode(counts)
    np.save(OUT_DIR / "z_ctrl.npy", z_ctrl)
    print(f"  z_ctrl: {z_ctrl.shape}  saved.")

    # ── 5. KO / OE ───────────────────────────────────────────────
    results = {"ctrl": z_ctrl}
    for gene in PERTURB_GENES:
        if gene not in gene_names:
            print(f"\n  WARNING: {gene} not in HVG, skipping.")
            continue

        gene_idx  = gene_names.index(gene)
        orig_mean = counts[:, gene_idx].mean()
        pct_expr  = (counts[:, gene_idx] > 0).mean()

        print(f"\n[5] {gene}  (mean={orig_mean:.4f}, pct={pct_expr:.3f})")

        counts_ko = counts.copy(); counts_ko[:, gene_idx] = 0
        z_ko      = encode(counts_ko)
        np.save(OUT_DIR / f"z_{gene}_KO.npy", z_ko)
        print(f"  KO shift={np.abs(z_ko - z_ctrl).mean():.4f}")
        results[f"{gene}_KO"] = z_ko

        counts_oe = counts.copy(); counts_oe[:, gene_idx] *= OE_FACTOR
        z_oe      = encode(counts_oe)
        np.save(OUT_DIR / f"z_{gene}_{oe_suffix}.npy", z_oe)
        print(f"  OE({OE_FACTOR}x) shift={np.abs(z_oe - z_ctrl).mean():.4f}")
        results[f"{gene}_OE"] = z_oe

    # ── 6. 요약 ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'Condition':<20} {'mean_shift':>12} {'max_shift':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
    for cond, z in results.items():
        if cond == 'ctrl': continue
        diff = np.abs(z - z_ctrl)
        print(f"  {cond:<20}  {diff.mean():>12.4f}  {diff.max():>12.4f}")
    print(f"\n  Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
