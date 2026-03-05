#!/usr/bin/env python
"""
step7b_simulate_topgenes.py  (scdiffeq_env)

step7a에서 저장한 z_KO.npy 로드 → scDiffeq trajectory simulation → L2 계산.
scvi import 없음 — scdiffeq_env에서 실행.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/step7b_simulate_topgenes.py \
        --t0_tag t70d_RG
"""

import torch
_orig = torch.load
def _p(*a, **k): k['weights_only'] = False; return _orig(*a, **k)
torch.load = _p
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"
SCAN_DIR = TRIAL6 / "allgene_scan"

CKPT5 = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

T0_OPTIONS = {"t70d_RG": 0.2471, "t115d_RG": 0.5588}
N_SIM_STEPS = 100
REF_GENES   = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

# 색상 (ref 유전자 강조)
COLORS_REF = {
    'IL17RD': '#d73027', 'PAX6': '#fc8d59', 'NEUROG2': '#fee090',
    'ASCL1': '#91bfdb', 'DLX2': '#4575b4', 'HES1': '#313695',
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t0_tag", required=True, choices=list(T0_OPTIONS.keys()))
    return p.parse_args()


def load_scdiffeq(adata_ref):
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq
    diffeq  = load_diffeq(ckpt_path=str(CKPT5))
    hparams = dict(diffeq.hparams)
    hparams['time_key'] = 'age_time_norm'
    hparams['use_key']  = 'X_scVI'
    model = sdq.scDiffEq(adata=adata_ref, **hparams)
    model.configure_data(adata=adata_ref)
    model.configure_model(diffeq, loading_existing=True)
    model.DiffEq.eval()
    return model


def simulate_endpoint(sde_model, z_init, t_start):
    device = next(sde_model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device, requires_grad=True)
    t_grid = torch.linspace(t_start, 1.0, N_SIM_STEPS + 1).to(device)
    traj   = sde_model.DiffEq.forward(X0, t_grid)
    return traj[-1].detach().cpu().numpy()


def paired_l2(z1, z2):
    return np.linalg.norm(z1 - z2, axis=1).mean()


def main():
    args   = parse_args()
    t0_tag = args.t0_tag
    t0     = T0_OPTIONS[t0_tag]
    LAT_DIR = SCAN_DIR / f"topgene_latents_{t0_tag}"

    if not LAT_DIR.exists():
        raise FileNotFoundError(f"Run step7a first: {LAT_DIR}")

    target_csv = LAT_DIR / "target_genes.csv"
    top_genes  = pd.read_csv(target_csv)['gene'].tolist()
    print(f"{'='*65}")
    print(f"Step 7b: Simulate top genes  [{t0_tag}]  N={len(top_genes)}")
    print(f"{'='*65}")

    # ── 모델 로드 ─────────────────────────────────────────────
    print("\n[1] Loading scDiffeq model...")
    adata_ref = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata_ref.obs['original_barcode'] = adata_ref.obs_names.copy()
    adata_ref.obs_names = [str(i) for i in range(adata_ref.n_obs)]
    sde_model = load_scdiffeq(adata_ref)
    print("  Model loaded.")

    # ── ctrl endpoint ─────────────────────────────────────────
    z_ctrl = np.load(LAT_DIR / "z_ctrl.npy")
    print(f"\n[2] Simulating ctrl endpoint  (t={t0} → 1.0, {N_SIM_STEPS} steps)...")
    z_ctrl_ep = simulate_endpoint(sde_model, z_ctrl, t0)
    print(f"  z_ctrl_ep: {z_ctrl_ep.shape}")

    # ── scan 결과 로드 ────────────────────────────────────────
    bio_csv  = SCAN_DIR / f"allgene_ko_scan_{t0_tag}_bio.csv"
    df_scan  = pd.read_csv(bio_csv).set_index('gene')

    # ── 각 유전자 시뮬레이션 ──────────────────────────────────
    print(f"\n[3] Simulating {len(top_genes)} genes...")
    rows = []
    for i, gene in enumerate(top_genes):
        z_ko_path = LAT_DIR / f"z_{gene}_KO.npy"
        if not z_ko_path.exists():
            print(f"  [{i+1:3d}] {gene:15s}  [SKIP: no KO file]")
            continue

        z_ko    = np.load(z_ko_path)
        l2_start = paired_l2(z_ctrl, z_ko)
        z_ko_ep  = simulate_endpoint(sde_model, z_ko, t0)
        l2_end   = paired_l2(z_ctrl_ep, z_ko_ep)

        scan_info = df_scan.loc[gene] if gene in df_scan.index else None
        rows.append({
            't0_tag': t0_tag,
            'gene': gene,
            'sim_l2_end': l2_end,
            'sim_l2_start': l2_start,
            'scan_l2': scan_info['ko_l2_mean'] if scan_info is not None else np.nan,
            'bio_rank': int(scan_info['bio_rank']) if scan_info is not None else 9999,
            'mean_expr': scan_info['mean_expr'] if scan_info is not None else np.nan,
            'frac_expr': scan_info['frac_expr'] if scan_info is not None else np.nan,
            'is_ref': gene in REF_GENES,
        })

        flag = " ← REF" if gene in REF_GENES else ""
        print(f"  [{i+1:3d}/{len(top_genes)}] {gene:15s}  "
              f"L2_end={l2_end:.5f}  bio_rank={rows[-1]['bio_rank']:4d}{flag}", flush=True)

    # ── 결과 정리 ─────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values('sim_l2_end', ascending=False).reset_index(drop=True)
    df['sim_rank'] = np.arange(1, len(df) + 1)
    out_csv = SCAN_DIR / f"topgene_sim_{t0_tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved: {out_csv}")

    # ── 상관 분석 ─────────────────────────────────────────────
    from scipy.stats import spearmanr
    df_both = df.dropna(subset=['scan_l2'])
    r, p = spearmanr(df_both['scan_l2'], df_both['sim_l2_end'])
    print(f"\n  Scan L2 vs Sim L2 Spearman r={r:.4f}  p={p:.2e}")
    if r > 0.6:
        print("  ✅ Scan (t=0) predicts trajectory divergence well")
    else:
        print("  ⚠ Weak correlation — simulation adds info beyond encoding shift")

    # ── Summary: ref 유전자 ───────────────────────────────────
    print(f"\n{'='*65}")
    print(f"SUMMARY [{t0_tag}]  — Reference genes")
    print(f"  {'Gene':12s}  {'SimRank':>8}  {'BioRank':>8}  {'L2_end':>9}  {'L2_start':>10}")
    for _, r in df[df['is_ref']].sort_values('sim_rank').iterrows():
        print(f"  {r['gene']:12s}  {int(r['sim_rank']):8d}  "
              f"{int(r['bio_rank']) if r['bio_rank'] < 9999 else '?':>8}  "
              f"{r['sim_l2_end']:9.5f}  {r['sim_l2_start']:10.5f}")

    # ── Top 20 출력 ───────────────────────────────────────────
    print(f"\nTop 20 by sim_l2_end:")
    print(f"  {'Gene':15s}  {'SimRank':>8}  {'L2_end':>9}")
    for _, r in df.head(20).iterrows():
        flag = " ←" if r['gene'] in REF_GENES else ""
        print(f"  {r['gene']:15s}  {int(r['sim_rank']):8d}  {r['sim_l2_end']:9.5f}{flag}")

    # ── Figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'All-gene Trajectory Landscape  [{t0_tag}]', fontweight='bold')

    # Rank plot
    ax = axes[0]
    others = df[~df['gene'].isin(REF_GENES)]
    ax.scatter(others['sim_rank'], others['sim_l2_end'],
               s=6, alpha=0.4, color='#aaaaaa')
    for gene, col in COLORS_REF.items():
        row = df[df['gene'] == gene]
        if len(row):
            ax.scatter(row['sim_rank'], row['sim_l2_end'],
                       s=80, color=col, zorder=5, label=gene)
            ax.annotate(gene, (row['sim_rank'].values[0], row['sim_l2_end'].values[0]),
                        xytext=(4, 2), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Simulation rank')
    ax.set_ylabel('Paired L2 at t=1.0')
    ax.legend(fontsize=8, markerscale=1.5)
    ax.set_title('Trajectory divergence (KO)')

    # Scan vs Sim
    ax2 = axes[1]
    df_plot = df.dropna(subset=['scan_l2'])
    ax2.scatter(df_plot['scan_l2'], df_plot['sim_l2_end'],
                s=6, alpha=0.3, color='#555555')
    for gene, col in COLORS_REF.items():
        row = df_plot[df_plot['gene'] == gene]
        if len(row):
            ax2.scatter(row['scan_l2'], row['sim_l2_end'],
                        s=80, color=col, zorder=5, label=gene)
            ax2.annotate(gene, (row['scan_l2'].values[0], row['sim_l2_end'].values[0]),
                         xytext=(3, 2), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('KO L2 at t=0 (encoding scan)')
    ax2.set_ylabel('KO L2 at t=1.0 (simulation)')
    ax2.set_title(f'Encoding scan vs Simulation\nSpearman r={r:.3f}', fontweight='bold')
    ax2.legend(fontsize=8, markerscale=1.5)

    plt.tight_layout()
    fig.savefig(SCAN_DIR / f"fig_topgene_sim_{t0_tag}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Fig saved: fig_topgene_sim_{t0_tag}.png")


if __name__ == "__main__":
    main()
