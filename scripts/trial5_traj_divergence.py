#!/usr/bin/env python
"""
trial5_traj_divergence.py  (scdiffeq_env)

매 타임스텝마다 ctrl vs perturbation 궤적 거리를 측정.
같은 세포에서 출발하므로 paired L2 distance 사용:
  divergence(t) = mean_i ||z_ctrl_i(t) - z_pert_i(t)||

Output: results/trial5/traj_divergence/
"""

import torch
_orig = torch.load
def _p(*a, **k): k['weights_only']=False; return _orig(*a, **k)
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
from pathlib import Path
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

_BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq/results")
TRIAL5  = _BASE / "trial5"

N_SIM_STEPS   = 100
PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

COLORS = {
    'IL17RD_KO': '#1a5276', 'IL17RD_OE': '#5dade2',
    'PAX6_KO':   '#922b21', 'PAX6_OE':   '#f1948a',
    'NEUROG2_KO':'#7d6608', 'NEUROG2_OE':'#d4ac0d',
    'ASCL1_KO':  '#1e8449', 'ASCL1_OE':  '#82e0aa',
    'DLX2_KO':   '#6c3483', 'DLX2_OE':   '#d2b4de',
    'HES1_KO':   '#784212', 'HES1_OE':   '#f0b27a',
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str,   default=None)
    p.add_argument("--tag",       type=str,   default="enforce0")
    p.add_argument("--oe_factor", type=float, default=3.0)
    p.add_argument("--t0",        type=float, default=0.0,  help="simulation start time")
    p.add_argument("--t0_tag",    type=str,   default="t0", help="latent folder tag")
    p.add_argument("--base_dir",  type=str,   default=str(_BASE / "trial5"),
                   help="base results directory (e.g. .../trial6)")
    return p.parse_args()

def load_model(ckpt_path, adata_ref):
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq
    diffeq  = load_diffeq(ckpt_path=str(ckpt_path))
    hparams = dict(diffeq.hparams)
    hparams['time_key'] = 'age_time_norm'
    hparams['use_key']  = 'X_scVI'
    model = sdq.scDiffEq(adata=adata_ref, **hparams)
    model.configure_data(adata=adata_ref)
    model.configure_model(diffeq, loading_existing=True)
    model.DiffEq.eval()
    return model

def simulate(model, z_init, n_steps=100, t_start=0.0):
    """t_start→1 시뮬레이션. returns [T+1, N, D]"""
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, 1.0, n_steps + 1).to(device)
    traj   = model.DiffEq.forward(X0, t_grid)
    return traj.detach().cpu().numpy()   # [T+1, N, D]

def paired_l2(traj_ctrl, traj_pert):
    """각 타임스텝 t에서 mean_i ||ctrl_i(t) - pert_i(t)||"""
    diff = traj_ctrl - traj_pert          # [T+1, N, D]
    dist = np.linalg.norm(diff, axis=2)   # [T+1, N]
    return dist.mean(axis=1)             # [T+1]

def sliced_wasserstein(traj_ctrl, traj_pert, n_proj=50, seed=0):
    """각 타임스텝 t에서 Sliced Wasserstein Distance (1D 투영 평균)"""
    rng = np.random.default_rng(seed)
    T, N, D = traj_ctrl.shape
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    swd = np.zeros(T)
    for t in range(T):
        c = traj_ctrl[t]   # [N, D]
        p = traj_pert[t]   # [N, D]
        proj_c = c @ dirs.T   # [N, n_proj]
        proj_p = p @ dirs.T
        swd[t] = np.mean([wasserstein_distance(proj_c[:,k], proj_p[:,k])
                          for k in range(n_proj)])
    return swd

def main():
    args = parse_args()

    oe_suffix  = f"OE{int(args.oe_factor)}x"
    t0         = args.t0
    t0_tag     = args.t0_tag
    TRIAL_DIR  = Path(args.base_dir)
    LAT_DIR    = TRIAL_DIR / f"perturb_latents_{t0_tag}"
    OUT_DIR    = TRIAL_DIR / "traj_divergence"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print(f"Trajectory Divergence  (tag={args.tag}, OE={args.oe_factor}x, t0={t0})")
    print(f"  base_dir: {TRIAL_DIR}")
    print("=" * 65)

    # ── 1. 모델 로드 ──────────────────────────────────────────────
    print("\n[1] Loading model...")
    adata_train = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata_train.obs['original_barcode'] = adata_train.obs_names.copy()
    adata_train.obs_names = [str(i) for i in range(adata_train.n_obs)]

    if args.ckpt:
        CKPT = Path(args.ckpt)
    else:
        # enforce0 (original) checkpoint
        ckpts = sorted(TRIAL5.rglob("train/trial5_SDE_2*/**.ckpt"),
                       key=lambda p: p.stat().st_mtime)
        CKPT = next(p for p in reversed(ckpts) if p.name == "last.ckpt")
    print(f"  Checkpoint: {CKPT.parent.parent.parent.parent.name}")

    model = load_model(CKPT, adata_train)
    print("  Model loaded.")

    # ── 2. Perturbation latent 로드 ──────────────────────────────
    print("\n[2] Loading perturbation latents...")
    z_ctrl = np.load(LAT_DIR / "z_ctrl.npy")
    print(f"  z_ctrl: {z_ctrl.shape}  (from {LAT_DIR.name})")

    conditions = {}
    for gene in PERTURB_GENES:
        for ctype, fname in [('KO', f"z_{gene}_KO.npy"), ('OE', f"z_{gene}_{oe_suffix}.npy")]:
            path = LAT_DIR / fname
            if path.exists():
                conditions[f"{gene}_{ctype}"] = np.load(path)

    # ── 3. 시뮬레이션 & Divergence 계산 ─────────────────────────
    print(f"\n[3] Simulating ({N_SIM_STEPS} steps, t={t0}→1.0) + computing divergence...")
    print("  (ctrl simulation...)", flush=True)
    traj_ctrl = simulate(model, z_ctrl, N_SIM_STEPS, t_start=t0)   # [T+1, N, D]
    t_axis    = np.linspace(t0, 1.0, N_SIM_STEPS + 1)

    results_l2  = {}
    results_swd = {}

    for cond, z_pert in conditions.items():
        print(f"  {cond}...", flush=True)
        traj_pert = simulate(model, z_pert, N_SIM_STEPS, t_start=t0)

        results_l2[cond]  = paired_l2(traj_ctrl, traj_pert)
        results_swd[cond] = sliced_wasserstein(traj_ctrl, traj_pert, n_proj=50)

        final_l2  = results_l2[cond][-1]
        final_swd = results_swd[cond][-1]
        max_l2    = results_l2[cond].max()
        print(f"    L2(t=1)={final_l2:.4f}  SWD(t=1)={final_swd:.4f}  "
              f"max_L2={max_l2:.4f}", flush=True)

    # ── 4. CSV 저장 ───────────────────────────────────────────────
    df_l2  = pd.DataFrame(results_l2,  index=t_axis)
    df_swd = pd.DataFrame(results_swd, index=t_axis)
    df_l2.index.name  = 't'
    df_swd.index.name = 't'
    df_l2.to_csv(OUT_DIR / f"divergence_l2_{args.tag}.csv")
    df_swd.to_csv(OUT_DIR / f"divergence_swd_{args.tag}.csv")

    # ── 5. 요약 출력 ─────────────────────────────────────────────
    print(f"\n  {'Condition':<18} {'L2(t=1)':>10} {'SWD(t=1)':>10} {'max_L2':>10}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}")
    summary = sorted(conditions.keys(),
                     key=lambda c: results_l2[c][-1], reverse=True)
    for cond in summary:
        print(f"  {cond:<18}  {results_l2[cond][-1]:>10.4f}  "
              f"{results_swd[cond][-1]:>10.4f}  {results_l2[cond].max():>10.4f}")

    # ── 6. Figure A: 전체 divergence 곡선 ───────────────────────
    print("\n[4] Drawing figures...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Trajectory Divergence vs Control  [{args.tag}]',
                 fontsize=13, fontweight='bold')

    for ax, (df, title) in zip(axes, [
        (df_l2,  'Paired L2 Distance (mean per cell)'),
        (df_swd, 'Sliced Wasserstein Distance'),
    ]):
        for cond in summary:
            color = COLORS.get(cond, 'gray')
            lw    = 2.5 if 'IL17RD' in cond else 1.2
            alpha = 1.0 if 'IL17RD' in cond else 0.7
            ls    = '-' if 'KO' in cond else '--'
            ax.plot(t_axis, df[cond], color=color, lw=lw,
                    alpha=alpha, ls=ls, label=cond)
        # 타임포인트 세로선
        for tp in [0.112, 0.247, 0.559, 1.0]:
            ax.axvline(tp, color='#cccccc', lw=0.8, ls=':')
        ax.set_xlabel('Developmental time (t)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlim(t0, 1)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"figA_divergence_curves_{args.tag}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure B: IL17RD 하이라이트 ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'IL17RD vs Reference Genes — Trajectory Divergence  [{args.tag}]',
                 fontsize=12, fontweight='bold')

    highlight = ['IL17RD_KO', 'IL17RD_OE', 'PAX6_KO', 'ASCL1_KO', 'HES1_KO']
    highlight = [c for c in highlight if c in df_l2.columns]

    for ax, (df, title) in zip(axes, [
        (df_l2,  'Paired L2 Distance'),
        (df_swd, 'Sliced Wasserstein Distance'),
    ]):
        # 비하이라이트 조건은 연하게
        for cond in conditions:
            if cond not in highlight:
                ax.plot(t_axis, df[cond], color='#dddddd', lw=0.8, alpha=0.5)
        # 하이라이트
        for cond in highlight:
            color = COLORS.get(cond, 'gray')
            lw    = 3.0 if 'IL17RD' in cond else 1.8
            ls    = '-' if 'KO' in cond else '--'
            ax.plot(t_axis, df[cond], color=color, lw=lw, ls=ls, label=cond)
        for tp in [0.112, 0.247, 0.559, 1.0]:
            ax.axvline(tp, color='#bbbbbb', lw=0.8, ls=':',
                       label='timepoint' if tp == 0.112 else '')
        ax.set_xlabel('Developmental time (t)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlim(t0, 1)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"figB_IL17RD_highlight_{args.tag}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure C: t=1 기준 bar chart (ranking) ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Final Divergence at t=1.0  [{args.tag}]',
                 fontsize=12, fontweight='bold')

    for ax, (df, title) in zip(axes, [
        (df_l2,  'L2 Distance at t=1'),
        (df_swd, 'Sliced Wasserstein at t=1'),
    ]):
        cond_sorted = df.iloc[-1].sort_values(ascending=False)
        colors_bar  = [COLORS.get(c, '#aaaaaa') for c in cond_sorted.index]
        bars = ax.bar(range(len(cond_sorted)), cond_sorted.values,
                      color=colors_bar, alpha=0.85)
        ax.set_xticks(range(len(cond_sorted)))
        ax.set_xticklabels(cond_sorted.index, rotation=45, ha='right', fontsize=8)
        # IL17RD 강조
        for i, cond in enumerate(cond_sorted.index):
            if 'IL17RD' in cond:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
        ax.set_ylabel(title)
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"figC_final_ranking_{args.tag}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Output: {OUT_DIR}")
    print(f"  figA_divergence_curves_{args.tag}.png")
    print(f"  figB_IL17RD_highlight_{args.tag}.png")
    print(f"  figC_final_ranking_{args.tag}.png")
    print("Done!")

if __name__ == "__main__":
    main()
