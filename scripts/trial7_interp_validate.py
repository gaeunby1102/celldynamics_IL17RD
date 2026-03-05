#!/usr/bin/env python
"""
trial7_interp_validate.py  (scdiffeq_env)  —  GPU 0

방법 2: trial7 holdout 모델로 진짜 interpolation 검증
  - 학습 타임포인트: 49d / 60d / 70d / 168d  (116d 제외)
  - 검증: t=70d 시작 → t=116d 예측 vs 실제 116d 관측 SWD
  - trial6 quasi 결과와 비교 (동일 메트릭)

사용법:
  python scripts/trial7_interp_validate.py --ckpt <path_to_last.ckpt>
  또는 --auto_ckpt (results/trial7/train/에서 자동 탐색)

Output: results/trial7/interp_validate/
"""

import torch
_orig = torch.load
def _p(*a, **k): k['weights_only'] = False; return _orig(*a, **k)
torch.load = _p
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5  = BASE / "results" / "trial5"
TRIAL6  = BASE / "results" / "trial6"
TRIAL7  = BASE / "results" / "trial7"
OUT_DIR = TRIAL7 / "interp_validate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_START  = 0.2471   # 70d (training set에 포함)
T_END    = 0.5588   # 116d (trial7 hold-out)
N_SIM    = 50
N_PROJ   = 100
N_SIM_REPEATS = 5

TIME_KEY     = "age_time_norm"
USE_KEY      = "X_scVI"
CELLTYPE_COL = "CellType_refine"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str, default=None,
                   help="Path to trial7 last.ckpt")
    p.add_argument("--auto_ckpt", action="store_true",
                   help="Auto-find latest checkpoint in results/trial7/train/")
    return p.parse_args()


def find_latest_ckpt():
    """results/trial7/train/ 에서 가장 최신 last.ckpt 탐색"""
    train_dir = TRIAL7 / "train"
    ckpts = sorted(train_dir.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No last.ckpt found under {train_dir}")
    return ckpts[-1]


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
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, t_end, n_steps + 1).to(device)
    if not stochastic:
        sde = model.DiffEq.DiffEq
        orig_g = sde.g
        sde.g  = lambda t, y: torch.zeros_like(orig_g(t, y))
        try:
            traj = model.DiffEq.forward(X0, t_grid)
        finally:
            sde.g = orig_g
    else:
        traj = model.DiffEq.forward(X0, t_grid)
    return traj.detach().cpu().numpy()   # [T+1, N, D]


def sliced_wasserstein(a, b, n_proj=100, seed=0):
    rng  = np.random.default_rng(seed)
    D    = a.shape[1]
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa   = a @ dirs.T
    pb   = b @ dirs.T
    return float(np.mean([wasserstein_distance(pa[:, k], pb[:, k])
                          for k in range(n_proj)]))


def latent_to_umap(latent, atlas_latent, atlas_umap, k=5):
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn.fit(atlas_latent)
    _, idx = nn.kneighbors(latent)
    return atlas_umap[idx].mean(axis=1)


def main():
    args = parse_args()

    if args.auto_ckpt or args.ckpt is None:
        ckpt_path = find_latest_ckpt()
        print(f"  Auto-found checkpoint: {ckpt_path}")
    else:
        ckpt_path = Path(args.ckpt)

    print("=" * 65)
    print("Trial7 Interpolation Validation (Holdout 116d)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  t_start={T_START}(70d)  →  t_end={T_END}(116d hold-out)")
    print("=" * 65)

    # ── 1. adata 로드 (full — 116d 포함) ────────────────────────────────
    print("\n[1] Loading full adata (including holdout 116d)...")
    adata_full = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    print(f"  Full: {adata_full.n_obs:,} cells")
    tps = sorted(adata_full.obs[TIME_KEY].unique())
    print(f"  Timepoints: {tps}")

    # trial7 학습에 쓰인 adata (116d 제외) — model config용
    mask_holdout = (adata_full.obs[TIME_KEY] - T_END).abs() < 1e-4
    adata_train  = adata_full[~mask_holdout].copy()
    adata_train.obs['original_barcode'] = adata_train.obs_names.copy()
    adata_train.obs_names = [str(i) for i in range(adata_train.n_obs)]
    print(f"  Train (without 116d): {adata_train.n_obs:,} cells")

    # ── 2. 모델 로드 ─────────────────────────────────────────────────────
    print("\n[2] Loading trial7 model...")
    model = load_model(ckpt_path, adata_train)
    print("  Model loaded.")

    # ── 3. 시작 세포 (70d RG) ────────────────────────────────────────────
    print(f"\n[3] Extracting start cells at t={T_START} (70d, RG only)...")
    lat_dir_t6 = TRIAL6 / "perturb_latents_t70d_RG"
    if (lat_dir_t6 / "z_ctrl.npy").exists():
        z_start = np.load(lat_dir_t6 / "z_ctrl.npy")
        print(f"  Loaded z_start from trial6 perturb_latents_t70d_RG: {z_start.shape}")
    else:
        mask_70d = (adata_full.obs[TIME_KEY] - T_START).abs() < 1e-4
        mask_rg  = adata_full.obs.get(CELLTYPE_COL, pd.Series("", index=adata_full.obs_names)) == "RG"
        mask     = mask_70d & mask_rg
        z_start  = adata_full[mask].obsm[USE_KEY]
        print(f"  Extracted {z_start.shape[0]} RG cells at t={T_START} directly")

    # ── 4. 실제 116d 세포 ────────────────────────────────────────────────
    print(f"\n[4] Extracting holdout observed cells at t={T_END} (116d)...")
    z_obs  = adata_full[mask_holdout].obsm[USE_KEY]
    ct_obs = adata_full[mask_holdout].obs.get(CELLTYPE_COL,
             pd.Series("Unknown")).values
    print(f"  116d observed: {z_obs.shape[0]} cells")
    for ct, n in zip(*np.unique(ct_obs, return_counts=True)):
        print(f"    {ct}: {n}")

    # ── 5. 시뮬레이션 ────────────────────────────────────────────────────
    print(f"\n[5] Simulating t={T_START} → t={T_END}  ({N_SIM} steps)...")

    print("  ODE...")
    traj_ode   = simulate_to_endpoint(model, z_start, T_START, T_END, N_SIM, stochastic=False)
    z_pred_ode = traj_ode[-1]

    print(f"  SDE ({N_SIM_REPEATS} runs)...")
    sde_ends = []
    for r in range(N_SIM_REPEATS):
        tr = simulate_to_endpoint(model, z_start, T_START, T_END, N_SIM, stochastic=True)
        sde_ends.append(tr[-1])
    z_pred_sde_all  = np.concatenate(sde_ends, axis=0)
    z_pred_sde_mean = np.stack(sde_ends).mean(axis=0)

    # ── 6. SWD 계산 ──────────────────────────────────────────────────────
    print(f"\n[6] SWD (n_proj={N_PROJ})...")

    swd_ode          = sliced_wasserstein(z_pred_ode,      z_obs, N_PROJ)
    swd_sde          = sliced_wasserstein(z_pred_sde_all,  z_obs, N_PROJ)
    swd_base_70_116  = sliced_wasserstein(z_start,         z_obs, N_PROJ)
    idx = np.random.default_rng(0).permutation(len(z_obs))
    half = len(z_obs) // 2
    swd_self = sliced_wasserstein(z_obs[idx[:half]], z_obs[idx[half:half*2]], N_PROJ)

    # trial6 quasi 결과 로드 (비교용)
    quasi_csv = TRIAL6 / "interp_quasi" / "swd_results.csv"
    quasi_swd = {}
    if quasi_csv.exists():
        df_quasi = pd.read_csv(quasi_csv)
        quasi_swd = dict(zip(df_quasi['comparison'], df_quasi['SWD']))

    print(f"\n  {'Comparison':<38} {'Trial7':>8}  {'Trial6 quasi':>12}")
    print(f"  {'-'*38}  {'-'*8}  {'-'*12}")

    def qv(key):
        return f"{quasi_swd.get(key, float('nan')):>12.4f}" if quasi_swd else "           N/A"

    print(f"  {'ODE pred vs observed 116d':<38} {swd_ode:>8.4f}  {qv('ODE_pred_vs_obs')}")
    print(f"  {'SDE pred (all) vs observed':<38} {swd_sde:>8.4f}  {qv('SDE_pred_vs_obs')}")
    print(f"  {'Baseline: 70d vs 116d':<38} {swd_base_70_116:>8.4f}  {qv('baseline_70d_vs_obs')}")
    print(f"  {'Self-split: 116d vs 116d':<38} {swd_self:>8.4f}  {qv('self_split_116d')}")

    imp_ode = (swd_base_70_116 - swd_ode) / swd_base_70_116 * 100
    imp_sde = (swd_base_70_116 - swd_sde) / swd_base_70_116 * 100
    print(f"\n  ODE improvement over baseline: {imp_ode:+.1f}%")
    print(f"  SDE improvement over baseline: {imp_sde:+.1f}%")

    df_swd = pd.DataFrame({
        'comparison': ['ODE_pred_vs_obs', 'SDE_pred_vs_obs',
                       'baseline_70d_vs_obs', 'self_split_116d'],
        'SWD_trial7': [swd_ode, swd_sde, swd_base_70_116, swd_self],
    })
    if quasi_swd:
        df_swd['SWD_trial6_quasi'] = [quasi_swd.get(k, np.nan)
                                       for k in df_swd['comparison']]
    df_swd.to_csv(OUT_DIR / "swd_results.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'swd_results.csv'}")

    # ── 7. UMAP 시각화 ───────────────────────────────────────────────────
    print("\n[7] UMAP visualization...")
    if 'X_umap' in adata_full.obsm:
        atlas_umap   = adata_full.obsm['X_umap']
        atlas_latent = adata_full.obsm[USE_KEY]

        umap_start   = latent_to_umap(z_start,       atlas_latent, atlas_umap)
        umap_obs     = latent_to_umap(z_obs,          atlas_latent, atlas_umap)
        umap_ode     = latent_to_umap(z_pred_ode,     atlas_latent, atlas_umap)
        umap_sde_all = latent_to_umap(z_pred_sde_all, atlas_latent, atlas_umap)

        # Figure A: 3-panel scatter
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Trial7 Hold-out Interpolation: t={T_START}(70d) → t={T_END}(116d)\n'
                     f'SWD(ODE)={swd_ode:.4f}  SWD(SDE)={swd_sde:.4f}  '
                     f'Baseline={swd_base_70_116:.4f}  Self={swd_self:.4f}',
                     fontsize=11, fontweight='bold')

        kw_atlas = dict(s=2, alpha=0.15, color='#cccccc', rasterized=True)
        axes[0].scatter(atlas_umap[:, 0], atlas_umap[:, 1], **kw_atlas)
        axes[0].scatter(umap_start[:, 0], umap_start[:, 1],
                        s=8, alpha=0.6, color='#2196F3', label=f'start 70d RG (n={len(umap_start)})')
        axes[0].scatter(umap_obs[:, 0], umap_obs[:, 1],
                        s=8, alpha=0.6, color='#FF5722', label=f'hold-out 116d (n={len(umap_obs)})')
        axes[0].set_title('Start vs Hold-out Observed')
        axes[0].legend(fontsize=7); axes[0].set_xlabel('UMAP1'); axes[0].set_ylabel('UMAP2')

        axes[1].scatter(atlas_umap[:, 0], atlas_umap[:, 1], **kw_atlas)
        axes[1].scatter(umap_obs[:, 0], umap_obs[:, 1],
                        s=6, alpha=0.4, color='#FF5722', label='hold-out 116d')
        axes[1].scatter(umap_ode[:, 0], umap_ode[:, 1],
                        s=8, alpha=0.6, color='#4CAF50', label=f'ODE pred (n={len(umap_ode)})')
        axes[1].set_title(f'ODE Predicted vs Hold-out\nSWD={swd_ode:.4f}')
        axes[1].legend(fontsize=7); axes[1].set_xlabel('UMAP1')

        axes[2].scatter(atlas_umap[:, 0], atlas_umap[:, 1], **kw_atlas)
        axes[2].scatter(umap_obs[:, 0], umap_obs[:, 1],
                        s=6, alpha=0.4, color='#FF5722', label='hold-out 116d')
        axes[2].scatter(umap_sde_all[:, 0], umap_sde_all[:, 1],
                        s=4, alpha=0.3, color='#9C27B0',
                        label=f'SDE pred (n={len(umap_sde_all)})')
        axes[2].set_title(f'SDE Predicted vs Hold-out\nSWD={swd_sde:.4f}')
        axes[2].legend(fontsize=7); axes[2].set_xlabel('UMAP1')

        plt.tight_layout()
        fig.savefig(OUT_DIR / "figA_umap_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  figA_umap_comparison.png saved")

        # Figure B: Trial6 quasi vs Trial7 비교 bar
        if quasi_swd:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            labels   = ['Self-split\n116d', 'ODE pred\nvs 116d',
                        'SDE pred\nvs 116d', 'Baseline\n70d vs 116d']
            t7_vals  = [swd_self, swd_ode, swd_sde, swd_base_70_116]
            t6_vals  = [quasi_swd.get(k, np.nan) for k in
                        ['self_split_116d', 'ODE_pred_vs_obs',
                         'SDE_pred_vs_obs', 'baseline_70d_vs_obs']]
            x = np.arange(len(labels))
            w = 0.35
            bars1 = ax.bar(x - w/2, t7_vals, w, label='Trial7 (holdout 116d)', color='#42A5F5', alpha=0.85)
            bars2 = ax.bar(x + w/2, t6_vals, w, label='Trial6 quasi (all 116d seen)', color='#EF5350', alpha=0.85)
            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if not np.isnan(h):
                        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0005,
                                f'{h:.4f}', ha='center', va='bottom', fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylabel('Sliced Wasserstein Distance')
            ax.set_title('Trial7 (True Hold-out) vs Trial6 Quasi Interpolation')
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(OUT_DIR / "figB_trial7_vs_trial6.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  figB_trial7_vs_trial6.png saved")
    else:
        print("  WARNING: X_umap not in adata, skipping UMAP visualization")

    # ── 8. 요약 출력 & config 저장 ───────────────────────────────────────
    with open(OUT_DIR / "summary.txt", 'w') as f:
        f.write(f"Trial7 Interpolation Validation\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"t_start={T_START}(70d)  t_end={T_END}(116d hold-out)\n")
        f.write(f"N_start_cells={z_start.shape[0]}  N_obs_cells={z_obs.shape[0]}\n\n")
        f.write(f"ODE SWD        : {swd_ode:.4f}\n")
        f.write(f"SDE SWD (all)  : {swd_sde:.4f}\n")
        f.write(f"Baseline 70d   : {swd_base_70_116:.4f}\n")
        f.write(f"Self-split 116d: {swd_self:.4f}\n")
        f.write(f"ODE improvement: {imp_ode:+.1f}%\n")
        f.write(f"SDE improvement: {imp_sde:+.1f}%\n")
        if quasi_swd:
            f.write(f"\nTrial6 quasi comparison:\n")
            f.write(f"  Trial6 ODE SWD: {quasi_swd.get('ODE_pred_vs_obs', np.nan):.4f}\n")
            f.write(f"  Trial6 SDE SWD: {quasi_swd.get('SDE_pred_vs_obs', np.nan):.4f}\n")

    print(f"\n{'='*65}")
    print(f"[DONE] Output: {OUT_DIR}")
    print(f"  figA_umap_comparison.png")
    print(f"  figB_trial7_vs_trial6.png  (if trial6 quasi done)")
    print(f"  swd_results.csv")
    print(f"  summary.txt")


if __name__ == "__main__":
    main()
