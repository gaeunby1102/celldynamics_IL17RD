#!/usr/bin/env python
"""
step7a_encode_topgenes.py  (scArches_env)

step6 scan 결과 top N 유전자를 KO 인코딩 → z_KO.npy 저장.
step7b (scdiffeq_env)에서 시뮬레이션 읽어옴.

Usage:
    python scripts/step7a_encode_topgenes.py --t0_tag t70d_RG --top_n 150
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"

T0_OPTIONS = {"t70d_RG": 0.2471, "t115d_RG": 0.5588}
REF_GENES  = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]
HK_PREFIXES = ('MT-', 'RPL', 'RPS', 'MRPS', 'MRPL', 'HIST', 'HSP', 'HSPA', 'HSPB')
HK_EXACT    = {'MALAT1', 'NEAT1', 'ACTB', 'ACTG1', 'TUBA1A', 'TUBB', 'TMSB4X',
               'VIM', 'GAPDH', 'LDHA', 'ENO1', 'TMSB10', 'B2M', 'FTL', 'FTH1'}

def is_housekeeping(gene):
    return gene in HK_EXACT or any(gene.startswith(p) for p in HK_PREFIXES)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t0_tag", required=True, choices=list(T0_OPTIONS.keys()))
    p.add_argument("--top_n", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    t0_tag = args.t0_tag
    top_n  = args.top_n

    OUT = TRIAL6 / "allgene_scan" / f"topgene_latents_{t0_tag}"
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Step 7a: Encode top genes  [{t0_tag}]  top_n={top_n}")
    print(f"{'='*60}")

    # ── 1. step6 CSV 로드 & 필터 ──────────────────────────────
    scan_csv = TRIAL6 / "allgene_scan" / f"allgene_ko_scan_{t0_tag}.csv"
    df_scan  = pd.read_csv(scan_csv).sort_values('ko_l2_mean', ascending=False)
    df_bio   = df_scan[~df_scan['gene'].apply(is_housekeeping)].reset_index(drop=True)
    df_bio['bio_rank'] = np.arange(1, len(df_bio) + 1)
    print(f"  Biological genes after HK filter: {len(df_bio)} / {len(df_scan)}")

    top_genes = df_bio.head(top_n)['gene'].tolist()
    for g in REF_GENES:
        if g not in top_genes:
            top_genes.append(g)
    print(f"  Total target genes: {len(top_genes)}  (top {top_n} bio + {len(REF_GENES)} ref)")

    # Bio rank 저장
    df_bio[['gene','bio_rank','ko_l2_mean','mean_expr','frac_expr']].to_csv(
        TRIAL6 / "allgene_scan" / f"allgene_ko_scan_{t0_tag}_bio.csv", index=False)

    # ── 2. 모델 & 데이터 로드 ─────────────────────────────────
    import scvi
    print("\n  Loading adata & scVI model...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.n_obs)]

    model = scvi.model.SCVI.load(str(TRIAL5 / "scvi_dim30"), adata=adata)
    model.to_device("cuda:0")
    model.module.eval()
    device = next(model.module.parameters()).device
    print(f"  Model loaded  device={device}")

    # ── 3. Start cells ────────────────────────────────────────
    LAT_DIR = TRIAL6 / f"perturb_latents_{t0_tag}"
    obs_csv = LAT_DIR / "start_cells_obs.csv"
    z_ctrl  = np.load(LAT_DIR / "z_ctrl.npy")

    if obs_csv.exists():
        obs_df = pd.read_csv(obs_csv, index_col=0)
        mask   = adata.obs['original_barcode'].isin(obs_df.index)
    else:
        mask_t  = (adata.obs['age_time_norm'] - T0_OPTIONS[t0_tag]).abs() < 1e-4
        mask_rg = adata.obs['CellType_refine'].str.contains('RG|Radial', case=False, na=False)
        mask    = mask_t & mask_rg

    adata_start = adata[mask].copy()
    print(f"  Start cells: {adata_start.n_obs}")

    counts = adata_start.layers['counts']
    if sp.issparse(counts):
        x_np = counts.toarray().astype(np.float32)
    else:
        x_np = counts.astype(np.float32)
    x_base = torch.tensor(x_np, dtype=torch.float32)

    cats      = model.adata.obs['batch'].cat.categories
    batch_map = {b: i for i, b in enumerate(cats)}
    batch_idx = torch.tensor(
        [batch_map[b] for b in adata_start.obs['batch']], dtype=torch.long
    ).unsqueeze(1)

    gene_list = list(adata_start.var_names)
    gene_to_gi = {g: i for i, g in enumerate(gene_list)}

    # ── 4. 인코딩 ─────────────────────────────────────────────
    # ctrl 저장 (재확인)
    np.save(OUT / "z_ctrl.npy", z_ctrl)

    skipped = []
    for i, gene in enumerate(top_genes):
        out_path = OUT / f"z_{gene}_KO.npy"
        if out_path.exists():
            print(f"  [{i+1:3d}/{len(top_genes)}] {gene:15s}  [cached]", flush=True)
            continue

        if gene not in gene_to_gi:
            print(f"  [{i+1:3d}/{len(top_genes)}] {gene:15s}  [SKIP: not in HVG]")
            skipped.append(gene)
            continue

        gi   = gene_to_gi[gene]
        x_ko = x_base.clone()
        x_ko[:, gi] = 0.0

        with torch.no_grad():
            out = model.module.inference(x_ko.to(device), batch_idx.to(device))
        z_ko = out['qz'].loc.cpu().numpy()
        l2   = np.linalg.norm(z_ko - z_ctrl, axis=1).mean()

        np.save(out_path, z_ko)
        print(f"  [{i+1:3d}/{len(top_genes)}] {gene:15s}  L2={l2:.5f}", flush=True)

    print(f"\n✓ Saved {len(top_genes) - len(skipped)} KO latents to: {OUT}")
    if skipped:
        print(f"  Skipped ({len(skipped)}): {skipped}")

    # 유전자 목록 저장 (step7b에서 읽음)
    pd.DataFrame({'gene': top_genes}).to_csv(OUT / "target_genes.csv", index=False)
    print(f"  target_genes.csv saved.")


if __name__ == "__main__":
    main()
