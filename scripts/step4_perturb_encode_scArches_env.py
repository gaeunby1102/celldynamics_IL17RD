#!/usr/bin/env python
"""
step4_perturb_encode_scArches_env.py  (scArches_env)

t=0 세포에 유전자 KO/OE 적용 후 scVI(dim30)로 인코딩.

Perturbation 조건:
  - Control
  - IL17RD KO / OE 3x
  - PAX6 KO / OE 3x        (RG 정체성)
  - NEUROG2 KO / OE 3x     (흥분성 신경 분화)
  - ASCL1 KO / OE 3x       (억제성 신경 분화)
  - DLX2 KO / OE 3x        (Inh interneuron)
  - HES1 KO / OE 3x        (Notch → progenitor 유지)

Output: results/new_run/perturb_latents/
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pathlib import Path

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
RES     = BASE / "results"
OUT_DIR = RES / "new_run" / "perturb_latents"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# scVI로 학습된 데이터 (highly_variable 컬럼 포함, 5010 HVG 서브셋용)
RAW_PATH   = RES / "Input_fetal_neuron_trainset_after_scVI_hvg_5010_latent_dim30.h5ad"
SCVI_MODEL = RES / "scVI_latent" / "dim30"
TIME_COL   = "age_time_norm"
T0         = 0.0  # 출발 타임포인트

# 정답지 유전자 + IL17RD
PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]
OE_FACTOR = 3.0

# =============================================================================

def main():
    print("=" * 60)
    print("Step 4: Perturbation Encoding  (scArches_env)")
    print("=" * 60)

    # ── 1. scVI 로드 ─────────────────────────────────────────────
    print("\n[1] Loading scVI model...")
    import scvi
    scvi.settings.seed = 42

    # Reference 모델 로드
    ref_raw = ad.read_h5ad(RAW_PATH)
    print(f"  Raw data: {ref_raw.shape}")
    print(f"  Layers  : {list(ref_raw.layers.keys())}")

    # scVI 모델은 5010 HVG로 학습됨 → highly_variable 필터링
    if 'highly_variable' in ref_raw.var.columns:
        ref_hvg = ref_raw[:, ref_raw.var['highly_variable']].copy()
        print(f"  HVG subset: {ref_hvg.shape}")
    else:
        ref_hvg = ref_raw
        print("  WARNING: no highly_variable column, using full data")

    vae = scvi.model.SCVI.load(str(SCVI_MODEL), adata=ref_hvg)
    print("  scVI model loaded.")

    # ── 2. t=0 세포 추출 ─────────────────────────────────────────
    print(f"\n[2] Extracting t={T0} cells...")
    mask_t0 = (ref_hvg.obs[TIME_COL] - T0).abs() < 1e-6
    adata_t0 = ref_hvg[mask_t0].copy()
    print(f"  t=0 cells: {adata_t0.shape[0]:,}")
    print(f"  Cell types:\n{adata_t0.obs['CellType_refine'].value_counts().to_string()}")

    # raw count 행렬 (counts layer 사용)
    if 'counts' in adata_t0.layers:
        counts = adata_t0.layers['counts'].toarray() if hasattr(adata_t0.layers['counts'], 'toarray') \
                 else np.array(adata_t0.layers['counts'])
    else:
        counts = adata_t0.X.toarray() if hasattr(adata_t0.X, 'toarray') else np.array(adata_t0.X)

    gene_names = adata_t0.var_names.tolist()
    print(f"  Count matrix: {counts.shape}")

    # obs 저장 (cell type 등 메타데이터)
    adata_t0.obs.to_csv(OUT_DIR / "start_cells_obs.csv")
    print(f"  obs saved.")

    # ── 3. Control 인코딩 ─────────────────────────────────────────
    print("\n[3] Encoding control (no perturbation)...")

    def encode(counts_mat, ref_adata_template):
        """counts_mat [N, 5010] → scVI → latent [N, 30]"""
        import scipy.sparse as sp
        adata_q = ref_adata_template.copy()
        adata_q.layers['counts'] = sp.csr_matrix(counts_mat.astype(np.float32))
        adata_q.X = adata_q.layers['counts'].copy()
        # scVI는 동일 var_names 필요 → ref와 같은 구조 유지
        scvi.model.SCVI.setup_anndata(adata_q, layer='counts',
                                      batch_key='_scvi_batch')
        latent = vae.get_latent_representation(adata_q)
        return latent  # [N, 30]

    z_ctrl = encode(counts, adata_t0)
    np.save(OUT_DIR / "z_ctrl_t0.npy", z_ctrl)
    print(f"  z_ctrl: {z_ctrl.shape}  saved.")

    # ── 4. 각 유전자 KO / OE ─────────────────────────────────────
    results = {"ctrl": z_ctrl}

    for gene in PERTURB_GENES:
        if gene not in gene_names:
            print(f"  WARNING: {gene} not in HVG, skipping.")
            continue

        gene_idx = gene_names.index(gene)
        orig_expr_mean = counts[:, gene_idx].mean()

        # KO: count = 0
        print(f"\n[4] {gene} KO...")
        counts_ko = counts.copy()
        counts_ko[:, gene_idx] = 0
        z_ko = encode(counts_ko, adata_t0)
        np.save(OUT_DIR / f"z_{gene}_KO.npy", z_ko)
        print(f"  z_{gene}_KO: {z_ko.shape}  (orig mean={orig_expr_mean:.2f} → 0)")
        print(f"  Latent shift (KO vs ctrl): {(z_ko - z_ctrl).mean():.4f} ± {(z_ko - z_ctrl).std():.4f}")
        results[f"{gene}_KO"] = z_ko

        # OE: count × 3
        print(f"\n[4] {gene} OE ({OE_FACTOR}x)...")
        counts_oe = counts.copy()
        counts_oe[:, gene_idx] = counts[:, gene_idx] * OE_FACTOR
        z_oe = encode(counts_oe, adata_t0)
        np.save(OUT_DIR / f"z_{gene}_OE3x.npy", z_oe)
        print(f"  z_{gene}_OE: {z_oe.shape}  (→ {orig_expr_mean*OE_FACTOR:.2f})")
        print(f"  Latent shift (OE vs ctrl): {(z_oe - z_ctrl).mean():.4f} ± {(z_oe - z_ctrl).std():.4f}")
        results[f"{gene}_OE"] = z_oe

    # ── 5. Latent shift 요약 ──────────────────────────────────────
    print("\n[5] Latent shift summary (|mean shift| vs ctrl)")
    print(f"  {'Condition':<20} {'mean_shift':>12} {'max_abs_shift':>14}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*14}")
    for cond, z in results.items():
        if cond == 'ctrl': continue
        diff = z - z_ctrl
        print(f"  {cond:<20}  {diff.mean():>12.4f}  {np.abs(diff).max():>14.4f}")

    print(f"\n[Done] Latents saved to: {OUT_DIR}")
    print(f"  Files: {[f.name for f in OUT_DIR.glob('*.npy')]}")


if __name__ == "__main__":
    main()
