#!/usr/bin/env python
"""
step7_topgene_simulation.py  (scdiffeq_env)

step6 all-gene KO scan 결과에서 top N 유전자를 선택,
scDiffeq로 trajectory simulation → L2 at t=1.0 계산.

방법:
  1. step6 CSV에서 top N 유전자 선택 (latent shift 기준)
  2. 각 유전자: KO counts → scVI 재인코딩 (내부적으로 scArches 방식)
  3. scDiffeq Trial5 enforce1 → endpoint(t=1.0) 시뮬레이션
  4. Paired L2 계산 → step6 scan 결과와 비교

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/step7_topgene_simulation.py \
        --t0_tag t70d_RG --top_n 100

Output: results/trial6/allgene_scan/
    topgene_sim_{t0_tag}_top{N}.csv
    fig_topgene_sim_{t0_tag}.png
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
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"
OUT    = TRIAL6 / "allgene_scan"
OUT.mkdir(parents=True, exist_ok=True)

CKPT5 = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}
N_SIM_STEPS = 100

# 기존 6개 유전자 (반드시 포함)
REF_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

# Housekeeping 필터 — 고발현으로 library size effect 큰 유전자들 제외
HK_PREFIXES = ('MT-', 'RPL', 'RPS', 'MRPS', 'MRPL', 'HIST', 'HSP', 'HSPA', 'HSPB')
HK_EXACT    = {'MALAT1', 'NEAT1', 'ACTB', 'ACTG1', 'TUBA1A', 'TUBB', 'TMSB4X',
               'VIM', 'GAPDH', 'LDHA', 'ENO1', 'TMSB10', 'B2M', 'FTL', 'FTH1'}

def is_housekeeping(gene):
    return gene in HK_EXACT or any(gene.startswith(p) for p in HK_PREFIXES)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t0_tag", type=str, required=True,
                   choices=list(T0_OPTIONS.keys()))
    p.add_argument("--top_n", type=int, default=100,
                   help="Number of top genes to simulate")
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


def load_scvi(adata):
    import scvi
    m = scvi.model.SCVI.load(str(TRIAL5 / "scvi_dim30"), adata=adata)
    m.to_device("cuda:0")
    m.module.eval()
    return m


def encode_ko(scvi_model, adata_start, gene_idx):
    """gene_idx KO → scVI 재인코딩 → z_ko [N, 30]"""
    device = next(scvi_model.module.parameters()).device
    counts = adata_start.layers['counts']
    if sp.issparse(counts):
        x_np = counts.toarray().astype(np.float32)
    else:
        x_np = counts.astype(np.float32)

    x_ko = torch.tensor(x_np, dtype=torch.float32)
    x_ko[:, gene_idx] = 0.0

    cats = scvi_model.adata.obs['batch'].cat.categories
    batch_map = {b: i for i, b in enumerate(cats)}
    batch_idx = torch.tensor(
        [batch_map[b] for b in adata_start.obs['batch']],
        dtype=torch.long
    ).unsqueeze(1)

    with torch.no_grad():
        out = scvi_model.module.inference(x_ko.to(device), batch_idx.to(device))
    return out['qm'].cpu().numpy()


def simulate_endpoint(sde_model, z_init, t_start):
    device = next(sde_model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device, requires_grad=True)
    t_grid = torch.linspace(t_start, 1.0, N_SIM_STEPS + 1).to(device)
    traj   = sde_model.DiffEq.forward(X0, t_grid)
    return traj[-1].detach().cpu().numpy()


def paired_l2(z1, z2):
    return np.linalg.norm(z1 - z2, axis=1).mean()


def main():
    args = parse_args()
    t0_tag = args.t0_tag
    top_n  = args.top_n
    t0     = T0_OPTIONS[t0_tag]

    print("=" * 65)
    print(f"Step 7: Top-gene Trajectory Simulation  [{t0_tag}]  top_n={top_n}")
    print("=" * 65)

    # ── 1. step6 결과 로드 → 대상 유전자 선정 ─────────────────
    scan_csv = OUT / f"allgene_ko_scan_{t0_tag}.csv"
    if not scan_csv.exists():
        raise FileNotFoundError(f"Run step6 first: {scan_csv}")

    df_scan = pd.read_csv(scan_csv)
    df_scan = df_scan.sort_values('ko_l2_mean', ascending=False).reset_index(drop=True)
    df_scan['scan_rank_raw'] = np.arange(1, len(df_scan) + 1)

    # Housekeeping 필터링 후 재순위
    df_scan['is_hk'] = df_scan['gene'].apply(is_housekeeping)
    df_bio = df_scan[~df_scan['is_hk']].copy().reset_index(drop=True)
    df_bio['scan_rank'] = np.arange(1, len(df_bio) + 1)
    n_hk = df_scan['is_hk'].sum()
    print(f"\n  Housekeeping filtered: {n_hk} genes  → {len(df_bio)} biological genes")

    # top_n (bio 기준) + 기존 6개 유전자 반드시 포함
    top_genes = df_bio.head(top_n)['gene'].tolist()
    for g in REF_GENES:
        if g not in top_genes:
            top_genes.append(g)

    # df_scan에 bio rank 붙이기 (ref 포함)
    df_scan = df_scan.merge(df_bio[['gene','scan_rank']], on='gene', how='left')

    gene_to_idx = {g: i for i, g in enumerate(df_scan['gene'].tolist())}
    print(f"\n  Target genes: {len(top_genes)} (top {top_n} + {len(REF_GENES)} ref)")
    print(f"  IL17RD scan rank: {df_scan[df_scan['gene']=='IL17RD']['scan_rank'].values}")

    # ── 2. 데이터 로드 ─────────────────────────────────────────
    print("\n[2] Loading adata & models...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.n_obs)]

    scvi_model = load_scvi(adata)
    sde_model  = load_scdiffeq(adata)
    print("  Both models loaded.")

    # ── 3. Start cells & ctrl endpoint ─────────────────────────
    LAT_DIR = TRIAL6 / f"perturb_latents_{t0_tag}"
    obs_csv = LAT_DIR / "start_cells_obs.csv"
    z_ctrl  = np.load(LAT_DIR / "z_ctrl.npy")

    if obs_csv.exists():
        obs_df = pd.read_csv(obs_csv, index_col=0)
        mask   = adata.obs['original_barcode'].isin(obs_df.index)
    else:
        mask_t  = (adata.obs['age_time_norm'] - t0).abs() < 1e-4
        mask_rg = adata.obs['CellType_refine'].str.contains('RG|Radial', case=False, na=False)
        mask    = mask_t & mask_rg

    adata_start = adata[mask].copy()
    print(f"  Start cells: {adata_start.n_obs}  z_ctrl: {z_ctrl.shape}")

    print(f"\n[3] Simulating ctrl endpoint (t={t0} → 1.0, {N_SIM_STEPS} steps)...")
    z_ctrl_ep = simulate_endpoint(sde_model, z_ctrl, t0)
    print(f"  z_ctrl_ep: {z_ctrl_ep.shape}  L2 range=[{z_ctrl_ep.min():.3f}, {z_ctrl_ep.max():.3f}]")

    # ── 4. 각 유전자 KO encode → simulate → L2 ────────────────
    print(f"\n[4] Running {len(top_genes)} gene KO simulations...")
    rows = []
    scan_gene_idx = {g: df_scan[df_scan['gene']==g].index[0] for g in top_genes if g in gene_to_idx}

    for i, gene in enumerate(top_genes):
        if gene not in gene_to_idx:
            print(f"  [SKIP] {gene} not in HVG")
            continue

        gi = gene_to_idx[gene]

        # scVI KO encoding
        z_ko = encode_ko(scvi_model, adata_start, gi)
        l2_start = paired_l2(z_ctrl, z_ko)

        # scDiffeq simulation
        z_ko_ep = simulate_endpoint(sde_model, z_ko, t0)
        l2_end  = paired_l2(z_ctrl_ep, z_ko_ep)

        # step6 scan 결과 참조
        scan_row = df_scan[df_scan['gene'] == gene].iloc[0]

        ref_flag = gene in REF_GENES
        rows.append({
            't0_tag': t0_tag,
            'gene': gene,
            'sim_l2_end': l2_end,
            'sim_l2_start': l2_start,
            'scan_l2': scan_row['ko_l2_mean'],
            'scan_rank': int(scan_row['scan_rank']),
            'mean_expr': scan_row['mean_expr'],
            'frac_expr': scan_row['frac_expr'],
            'is_ref_gene': ref_flag,
        })

        flag = " ← REF" if ref_flag else ""
        print(f"  [{i+1:3d}/{len(top_genes)}] {gene:15s}  "
              f"L2_start={l2_start:.5f}  L2_end={l2_end:.5f}{flag}", flush=True)

    # ── 5. 결과 저장 ────────────────────────────────────────────
    df_sim = pd.DataFrame(rows).sort_values('sim_l2_end', ascending=False).reset_index(drop=True)
    df_sim['sim_rank'] = np.arange(1, len(df_sim) + 1)

    csv_out = OUT / f"topgene_sim_{t0_tag}_top{top_n}.csv"
    df_sim.to_csv(csv_out, index=False)
    print(f"\n✓ Saved: {csv_out}")

    # ── 6. 상관 분석: scan_l2 vs sim_l2_end ────────────────────
    from scipy.stats import spearmanr, pearsonr
    r_sp, p_sp = spearmanr(df_sim['scan_l2'], df_sim['sim_l2_end'])
    r_pe, p_pe = pearsonr(df_sim['scan_l2'], df_sim['sim_l2_end'])
    print(f"\n  Scan (t=0) vs Simulation (t=1) correlation:")
    print(f"  Spearman r={r_sp:.4f} p={p_sp:.2e}   Pearson r={r_pe:.4f} p={p_pe:.2e}")
    print(f"  → {'✅ Scan L2 predicts trajectory divergence' if r_sp > 0.5 else '⚠ Weak correlation — simulation needed'}")

    # ── 7. Figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Top-{top_n} Gene Simulation  [{t0_tag}]', fontweight='bold')

    colors_ref = {
        'IL17RD': '#d73027', 'PAX6': '#fc8d59', 'NEUROG2': '#fee090',
        'ASCL1': '#91bfdb', 'DLX2': '#4575b4', 'HES1': '#313695',
    }

    # Rank plot
    ax = axes[0]
    others = df_sim[~df_sim['gene'].isin(REF_GENES)]
    ax.scatter(others['sim_rank'], others['sim_l2_end'],
               s=8, alpha=0.4, color='#aaaaaa', label='non-ref')
    for gene, col in colors_ref.items():
        row = df_sim[df_sim['gene'] == gene]
        if len(row):
            ax.scatter(row['sim_rank'], row['sim_l2_end'],
                       s=80, color=col, zorder=5, label=gene)
            ax.annotate(gene, (row['sim_rank'].values[0], row['sim_l2_end'].values[0]),
                        xytext=(3, 1), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Simulation rank')
    ax.set_ylabel('Paired L2 at t=1.0')
    ax.legend(fontsize=8, markerscale=1.5)
    ax.set_title('Trajectory divergence (simulated)')

    # Scan vs Sim scatter
    ax2 = axes[1]
    ax2.scatter(df_sim['scan_l2'], df_sim['sim_l2_end'],
                s=8, alpha=0.3, color='#555555')
    for gene, col in colors_ref.items():
        row = df_sim[df_sim['gene'] == gene]
        if len(row):
            ax2.scatter(row['scan_l2'], row['sim_l2_end'],
                        s=80, color=col, zorder=5, label=gene)
            ax2.annotate(gene, (row['scan_l2'].values[0], row['sim_l2_end'].values[0]),
                         xytext=(2, 1), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('KO L2 at t=0 (step6 scan)')
    ax2.set_ylabel('KO L2 at t=1.0 (simulated)')
    ax2.set_title(f'Scan vs Simulation\n(Spearman r={r_sp:.3f})', fontweight='bold')
    ax2.legend(fontsize=8, markerscale=1.5)

    plt.tight_layout()
    fig_path = OUT / f"fig_topgene_sim_{t0_tag}_top{top_n}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig saved: {fig_path.name}")

    # ── 8. 요약 출력 ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"SUMMARY  [{t0_tag}]")
    print(f"  {'Gene':15s}  {'SimRank':>8}  {'ScanRank':>9}  {'L2_end':>8}  {'L2_scan':>9}")
    for _, r in df_sim[df_sim['is_ref_gene']].sort_values('sim_rank').iterrows():
        print(f"  {r['gene']:15s}  {int(r['sim_rank']):8d}  "
              f"{int(r['scan_rank']):9d}  {r['sim_l2_end']:8.5f}  {r['scan_l2']:9.5f}")


if __name__ == "__main__":
    main()
