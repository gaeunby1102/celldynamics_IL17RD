#!/usr/bin/env python
"""
trial6_interp_quasi.py  (scdiffeq_env)  —  GPU 1

방법 1: 기존 enforce=1.0 모델로 quasi interpolation 검증
  - t=0.2471(70d) 시작 → t=0.5588(116d) 예측
  - 실제 116d 세포 분포와 SWD 비교
  - UMAP 시각화 (predicted vs observed)

Output: results/trial6/interp_quasi/
"""

import torch
_orig = torch.load
def _p(*a, **k): k['weights_only'] = False; return _orig(*a, **k)
torch.load = _p
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"
OUT_DIR = TRIAL6 / "interp_quasi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_ENFORCE1 = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

T_START = 0.2471   # 70d
T_END   = 0.5588   # 116d (hold-out quasi)
N_SIM   = 50       # simulation steps
N_PROJ  = 100      # SWD random projections
N_SIM_REPEATS = 5  # SDE: multiple runs for stochastic average

TIME_KEY     = "age_time_norm"
USE_KEY      = "X_scVI"
CELLTYPE_COL = "CellType_refine"


def load_model(ckpt_path, adata_ref):
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq
    diffeq  = load_diffeq(ckpt_path=str(ckpt_path))
    hparams = dict(diffeq.hparams)
    hparams['time_key'] = TIME_KEY
    hparams['use_key']  = USE_KEY
    model = sdq.scDiffEq(adata=adata_ref, **hparams)
    model.configure_data(adata=adata_ref)
    model.configure_model(diffeq, loading_existing=True)
    model.DiffEq.eval()
    return model


def simulate_to_endpoint(model, z_init, t_start, t_end, n_steps, stochastic=True):
    """t_start → t_end 시뮬레이션. stochastic=False → ODE"""
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, t_end, n_steps + 1).to(device)

    if not stochastic:
        sde = model.DiffEq.DiffEq
        orig_g = sde.g
        sde.g = lambda t, y: torch.zeros_like(orig_g(t, y))
        try:
            traj = model.DiffEq.forward(X0, t_grid)
        finally:
            sde.g = orig_g
    else:
        traj = model.DiffEq.forward(X0, t_grid)

    return traj.detach().cpu().numpy()  # [T+1, N, D]


def sliced_wasserstein(a, b, n_proj=100, seed=0):
    """a, b: [N, D]"""
    rng  = np.random.default_rng(seed)
    D    = a.shape[1]
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa   = a @ dirs.T   # [N, n_proj]
    pb   = b @ dirs.T
    return float(np.mean([wasserstein_distance(pa[:, k], pb[:, k])
                          for k in range(n_proj)]))


def latent_to_umap(latent, atlas_latent, atlas_umap, k=5):
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn.fit(atlas_latent)
    _, idx = nn.kneighbors(latent)
    return atlas_umap[idx].mean(axis=1)


def main():
    print("=" * 65)
    print("Trial6 Interpolation Quasi-Validation")
    print(f"  enforce=1.0 model  |  t={T_START}(70d) → t={T_END}(116d)")
    print("=" * 65)

    # ── 1. 모델 로드 ────────────────────────────────────────────────────
    print("\n[1] Loading enforce=1.0 model...")
    adata_full = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata_ref  = adata_full.copy()
    adata_ref.obs['original_barcode'] = adata_ref.obs_names.copy()
    adata_ref.obs_names = [str(i) for i in range(adata_ref.n_obs)]

    model = load_model(CKPT_ENFORCE1, adata_ref)
    print(f"  Checkpoint: {CKPT_ENFORCE1.name}")

    # ── 2. 시작 세포 (70d RG) ───────────────────────────────────────────
    print(f"\n[2] Extracting start cells at t={T_START} (70d, RG only)...")
    # 기존 trial6 perturb_latents_t70d_RG 에서 z_ctrl 사용
    lat_dir = TRIAL6 / "perturb_latents_t70d_RG"
    if (lat_dir / "z_ctrl.npy").exists():
        z_start = np.load(lat_dir / "z_ctrl.npy")
        print(f"  Loaded z_start from {lat_dir.name}: {z_start.shape}")
    else:
        # fallback: 직접 추출
        mask_70d = (adata_full.obs[TIME_KEY] - T_START).abs() < 1e-4
        mask_rg  = adata_full.obs.get(CELLTYPE_COL, pd.Series("", index=adata_full.obs_names)) == "RG"
        mask     = mask_70d & mask_rg
        z_start  = adata_full[mask].obsm[USE_KEY]
        print(f"  Extracted {z_start.shape[0]} RG cells at t={T_START} directly")

    # ── 3. 실제 116d 세포 추출 ──────────────────────────────────────────
    print(f"\n[3] Extracting observed cells at t={T_END} (116d)...")
    mask_116 = (adata_full.obs[TIME_KEY] - T_END).abs() < 1e-4
    z_obs    = adata_full[mask_116].obsm[USE_KEY]
    ct_obs   = adata_full[mask_116].obs.get(CELLTYPE_COL, pd.Series("Unknown")).values
    print(f"  116d observed: {z_obs.shape[0]} cells")
    if len(np.unique(ct_obs)) > 1:
        for ct, n in zip(*np.unique(ct_obs, return_counts=True)):
            print(f"    {ct}: {n}")

    # ── 4. 시뮬레이션 ───────────────────────────────────────────────────
    print(f"\n[4] Simulating t={T_START} → t={T_END}  ({N_SIM} steps)...")

    # ODE (deterministic)
    print("  ODE simulation...")
    traj_ode = simulate_to_endpoint(model, z_start, T_START, T_END, N_SIM, stochastic=False)
    z_pred_ode = traj_ode[-1]   # endpoint [N, D]

    # SDE (stochastic, multiple runs)
    print(f"  SDE simulation ({N_SIM_REPEATS} repeats)...")
    sde_endpoints = []
    for r in range(N_SIM_REPEATS):
        traj_r = simulate_to_endpoint(model, z_start, T_START, T_END, N_SIM, stochastic=True)
        sde_endpoints.append(traj_r[-1])
    z_pred_sde_all = np.concatenate(sde_endpoints, axis=0)  # [N*R, D]
    z_pred_sde_mean = np.stack(sde_endpoints).mean(axis=0)  # [N, D]

    # ── 5. SWD 계산 ─────────────────────────────────────────────────────
    print(f"\n[5] SWD (n_proj={N_PROJ})...")

    swd_ode = sliced_wasserstein(z_pred_ode, z_obs, N_PROJ)
    swd_sde = sliced_wasserstein(z_pred_sde_all, z_obs, N_PROJ)

    # Baseline: 70d → 70d (같은 시작점 vs 116d)
    swd_base_70_116 = sliced_wasserstein(z_start, z_obs, N_PROJ)

    # Baseline: 116d 내부 random split SWD
    idx = np.random.default_rng(0).permutation(len(z_obs))
    half = len(z_obs) // 2
    swd_self = sliced_wasserstein(z_obs[idx[:half]], z_obs[idx[half:half*2]], N_PROJ)

    print(f"\n  {'Comparison':<35} {'SWD':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    print(f"  {'ODE pred vs observed 116d':<35} {swd_ode:>8.4f}")
    print(f"  {'SDE pred (all runs) vs observed':<35} {swd_sde:>8.4f}")
    print(f"  {'Baseline: 70d start vs observed 116d':<35} {swd_base_70_116:>8.4f}")
    print(f"  {'Self-split: 116d vs 116d':<35} {swd_self:>8.4f}")

    improvement_ode = (swd_base_70_116 - swd_ode) / swd_base_70_116 * 100
    improvement_sde = (swd_base_70_116 - swd_sde) / swd_base_70_116 * 100
    print(f"\n  ODE improvement over baseline: {improvement_ode:+.1f}%")
    print(f"  SDE improvement over baseline: {improvement_sde:+.1f}%")

    # CSV 저장
    df_swd = pd.DataFrame({
        'comparison': ['ODE_pred_vs_obs', 'SDE_pred_vs_obs',
                       'baseline_70d_vs_obs', 'self_split_116d'],
        'SWD': [swd_ode, swd_sde, swd_base_70_116, swd_self],
    })
    df_swd.to_csv(OUT_DIR / "swd_results.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'swd_results.csv'}")

    # ── 6. UMAP 시각화 ──────────────────────────────────────────────────
    print("\n[6] UMAP visualization...")

    # atlas UMAP 좌표 추출
    if 'X_umap' in adata_full.obsm:
        atlas_umap   = adata_full.obsm['X_umap']
        atlas_latent = adata_full.obsm[USE_KEY]
        print(f"  Atlas: {atlas_latent.shape[0]} cells, UMAP available")

        umap_start   = latent_to_umap(z_start,        atlas_latent, atlas_umap)
        umap_obs     = latent_to_umap(z_obs,           atlas_latent, atlas_umap)
        umap_ode     = latent_to_umap(z_pred_ode,      atlas_latent, atlas_umap)
        umap_sde_all = latent_to_umap(z_pred_sde_all,  atlas_latent, atlas_umap)

        # Figure A: 분포 비교 scatter
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Quasi Interpolation: t={T_START}(70d) → t={T_END}(116d)\n'
                     f'enforce=1.0 | SWD(ODE)={swd_ode:.4f}  SWD(SDE)={swd_sde:.4f}',
                     fontsize=12, fontweight='bold')

        kw_atlas = dict(s=2, alpha=0.2, color='#cccccc', rasterized=True)
        axes[0].scatter(atlas_umap[:, 0], atlas_umap[:, 1], **kw_atlas)
        axes[0].scatter(umap_start[:, 0], umap_start[:, 1],
                        s=8, alpha=0.6, color='#2196F3', label=f'start 70d (n={len(umap_start)})')
        axes[0].scatter(umap_obs[:, 0], umap_obs[:, 1],
                        s=8, alpha=0.6, color='#FF5722', label=f'observed 116d (n={len(umap_obs)})')
        axes[0].set_title('Start vs Observed'); axes[0].legend(fontsize=7)
        axes[0].set_xlabel('UMAP1'); axes[0].set_ylabel('UMAP2')

        axes[1].scatter(atlas_umap[:, 0], atlas_umap[:, 1], **kw_atlas)
        axes[1].scatter(umap_obs[:, 0], umap_obs[:, 1],
                        s=6, alpha=0.4, color='#FF5722', label='observed 116d')
        axes[1].scatter(umap_ode[:, 0], umap_ode[:, 1],
                        s=8, alpha=0.6, color='#4CAF50', label=f'ODE pred (n={len(umap_ode)})')
        axes[1].set_title(f'ODE Predicted vs Observed\nSWD={swd_ode:.4f}')
        axes[1].legend(fontsize=7)
        axes[1].set_xlabel('UMAP1'); axes[1].set_ylabel('UMAP2')

        axes[2].scatter(atlas_umap[:, 0], atlas_umap[:, 1], **kw_atlas)
        axes[2].scatter(umap_obs[:, 0], umap_obs[:, 1],
                        s=6, alpha=0.4, color='#FF5722', label='observed 116d')
        axes[2].scatter(umap_sde_all[:, 0], umap_sde_all[:, 1],
                        s=4, alpha=0.3, color='#9C27B0',
                        label=f'SDE pred (n={len(umap_sde_all)})')
        axes[2].set_title(f'SDE Predicted vs Observed\nSWD={swd_sde:.4f}')
        axes[2].legend(fontsize=7)
        axes[2].set_xlabel('UMAP1'); axes[2].set_ylabel('UMAP2')

        plt.tight_layout()
        fig.savefig(OUT_DIR / "figA_umap_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  figA_umap_comparison.png saved")

        # Figure B: SWD 요약 bar chart
        fig, ax = plt.subplots(figsize=(7, 4))
        labels  = ['Self-split\n116d', 'ODE pred\nvs 116d', 'SDE pred\nvs 116d',
                   'Baseline\n70d vs 116d']
        values  = [swd_self, swd_ode, swd_sde, swd_base_70_116]
        colors  = ['#66BB6A', '#42A5F5', '#AB47BC', '#EF5350']
        bars    = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.axhline(swd_self, color='#66BB6A', lw=1.5, ls='--', alpha=0.6, label='self-split 116d')
        ax.set_ylabel('Sliced Wasserstein Distance')
        ax.set_title('Quasi-Interpolation Performance (enforce=1.0)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "figB_swd_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  figB_swd_summary.png saved")

    else:
        print("  WARNING: X_umap not in adata, skipping UMAP visualization")

    # ── 7. 타임포인트별 SWD 궤적 ───────────────────────────────────────
    print("\n[7] Time-series SWD along trajectory (70d → 116d)...")
    t_grid_np = np.linspace(T_START, T_END, N_SIM + 1)

    traj_ode_full = simulate_to_endpoint(model, z_start, T_START, T_END, N_SIM, stochastic=False)
    # SDE: 대표 1회
    traj_sde_full = simulate_to_endpoint(model, z_start, T_START, T_END, N_SIM, stochastic=True)

    swd_ode_ts = []
    swd_sde_ts = []
    for step_i in range(0, N_SIM + 1, max(1, N_SIM // 10)):
        z_t_ode = traj_ode_full[step_i]
        z_t_sde = traj_sde_full[step_i]
        swd_ode_ts.append((t_grid_np[step_i],
                           sliced_wasserstein(z_t_ode, z_obs, N_PROJ // 2)))
        swd_sde_ts.append((t_grid_np[step_i],
                           sliced_wasserstein(z_t_sde, z_obs, N_PROJ // 2)))

    df_ts = pd.DataFrame({'t':      [x[0] for x in swd_ode_ts],
                          'SWD_ODE': [x[1] for x in swd_ode_ts],
                          'SWD_SDE': [x[1] for x in swd_sde_ts]})
    df_ts.to_csv(OUT_DIR / "swd_timeseries.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_ts['t'], df_ts['SWD_ODE'], 'b-o', lw=2, label='ODE')
    ax.plot(df_ts['t'], df_ts['SWD_SDE'], 'm-s', lw=2, label='SDE (1 run)')
    ax.axvline(T_END, color='red', lw=1.5, ls='--', alpha=0.7, label=f't={T_END} (116d)')
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('SWD vs observed 116d')
    ax.set_title('SWD to Target (116d) Along Trajectory\n(lower = closer to target)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figC_swd_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  figC_swd_timeseries.png saved")

    print(f"\n{'='*65}")
    print(f"[DONE] Output: {OUT_DIR}")
    print(f"  figA_umap_comparison.png")
    print(f"  figB_swd_summary.png")
    print(f"  figC_swd_timeseries.png")
    print(f"  swd_results.csv")
    print(f"  swd_timeseries.csv")
    print(f"\n  Key metrics:")
    print(f"    ODE SWD  : {swd_ode:.4f}")
    print(f"    SDE SWD  : {swd_sde:.4f}")
    print(f"    Baseline : {swd_base_70_116:.4f}")
    print(f"    Self     : {swd_self:.4f}")
    print(f"    Improvement (ODE): {improvement_ode:+.1f}%")
    print(f"    Improvement (SDE): {improvement_sde:+.1f}%")


if __name__ == "__main__":
    main()
