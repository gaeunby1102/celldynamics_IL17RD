#!/usr/bin/env python
"""
trial6_traj_divergence.py  (scdiffeq_env)

Trial6: Trajectory divergence from RG-only start cells at later timepoints.
  --t0_tag t70d_RG   → simulate t=0.2471 → 1.0
  --t0_tag t115d_RG  → simulate t=0.5588 → 1.0

Output: results/trial6/traj_divergence/
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
from pathlib import Path
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5  = BASE / "results" / "trial5"
TRIAL6  = BASE / "results" / "trial6"
OUT_DIR = TRIAL6 / "traj_divergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIM_STEPS   = 100
PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}

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
    p.add_argument("--ckpt",      type=str,   required=True,   help="scDiffeq checkpoint path")
    p.add_argument("--tag",       type=str,   default="enforce1_t70d_RG")
    p.add_argument("--t0_tag",    type=str,   required=True,   choices=list(T0_OPTIONS.keys()))
    p.add_argument("--oe_factor", type=float, default=3.0)
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

def simulate(model, z_init, n_steps, t_start):
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, 1.0, n_steps + 1).to(device)
    traj   = model.DiffEq.forward(X0, t_grid)
    return traj.detach().cpu().numpy()   # [T+1, N, D]

def paired_l2(traj_ctrl, traj_pert):
    diff = traj_ctrl - traj_pert
    dist = np.linalg.norm(diff, axis=2)
    return dist.mean(axis=1)

def sliced_wasserstein(traj_ctrl, traj_pert, n_proj=50, seed=0):
    rng  = np.random.default_rng(seed)
    T, N, D = traj_ctrl.shape
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    swd  = np.zeros(T)
    for t in range(T):
        proj_c = traj_ctrl[t] @ dirs.T
        proj_p = traj_pert[t] @ dirs.T
        swd[t] = np.mean([wasserstein_distance(proj_c[:, k], proj_p[:, k])
                          for k in range(n_proj)])
    return swd

def main():
    args      = parse_args()
    t0        = T0_OPTIONS[args.t0_tag]
    oe_suffix = f"OE{int(args.oe_factor)}x"
    LAT_DIR   = TRIAL6 / f"perturb_latents_{args.t0_tag}"

    print("=" * 65)
    print(f"Trial6 Trajectory Divergence")
    print(f"  tag={args.tag}  t0_tag={args.t0_tag}  t0={t0}  OE={args.oe_factor}x")
    print("=" * 65)

    # ── 1. 모델 로드 ────────────────────────────────────────────
    print("\n[1] Loading model...")
    adata_train = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata_train.obs['original_barcode'] = adata_train.obs_names.copy()
    adata_train.obs_names = [str(i) for i in range(adata_train.n_obs)]

    CKPT  = Path(args.ckpt)
    model = load_model(CKPT, adata_train)
    print(f"  Checkpoint: {CKPT.parent.parent.parent.parent.name}")
    print("  Model loaded.")

    # ── 2. Latent 로드 ──────────────────────────────────────────
    print(f"\n[2] Loading perturbation latents from {LAT_DIR.name}...")
    z_ctrl = np.load(LAT_DIR / "z_ctrl.npy")
    print(f"  z_ctrl: {z_ctrl.shape}")

    conditions = {}
    for gene in PERTURB_GENES:
        for ctype, fname in [('KO', f"z_{gene}_KO.npy"),
                             ('OE', f"z_{gene}_{oe_suffix}.npy")]:
            path = LAT_DIR / fname
            if path.exists():
                conditions[f"{gene}_{ctype}"] = np.load(path)

    # ── 3. 시뮬레이션 & Divergence ──────────────────────────────
    print(f"\n[3] Simulating ({N_SIM_STEPS} steps, t={t0:.4f}→1.0)...")
    traj_ctrl = simulate(model, z_ctrl, N_SIM_STEPS, t0)
    t_axis    = np.linspace(t0, 1.0, N_SIM_STEPS + 1)

    results_l2  = {}
    results_swd = {}

    for cond, z_pert in conditions.items():
        print(f"  {cond}...", flush=True)
        traj_pert = simulate(model, z_pert, N_SIM_STEPS, t0)
        results_l2[cond]  = paired_l2(traj_ctrl, traj_pert)
        results_swd[cond] = sliced_wasserstein(traj_ctrl, traj_pert)
        print(f"    L2(t=1)={results_l2[cond][-1]:.4f}  "
              f"SWD(t=1)={results_swd[cond][-1]:.4f}", flush=True)

    # ── 4. CSV 저장 ─────────────────────────────────────────────
    df_l2  = pd.DataFrame(results_l2,  index=t_axis)
    df_swd = pd.DataFrame(results_swd, index=t_axis)
    df_l2.index.name  = 't'
    df_swd.index.name = 't'
    df_l2.to_csv(OUT_DIR / f"divergence_l2_{args.tag}.csv")
    df_swd.to_csv(OUT_DIR / f"divergence_swd_{args.tag}.csv")

    # ── 5. 요약 출력 ────────────────────────────────────────────
    summary = sorted(conditions.keys(), key=lambda c: results_l2[c][-1], reverse=True)
    print(f"\n  {'Condition':<18} {'L2(t=1)':>10} {'SWD(t=1)':>10}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}")
    for cond in summary:
        print(f"  {cond:<18}  {results_l2[cond][-1]:>10.4f}  "
              f"{results_swd[cond][-1]:>10.4f}")

    # ── 6. Figure A: 전체 divergence 곡선 ──────────────────────
    print("\n[4] Drawing figures...")
    timepoints = [tp for tp in [0.112, 0.247, 0.559, 1.0] if tp > t0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Trajectory Divergence vs Control  [{args.tag}]  (RG-only start)',
                 fontsize=13, fontweight='bold')
    for ax, (df, title) in zip(axes, [(df_l2, 'Paired L2'), (df_swd, 'Sliced Wasserstein')]):
        for cond in summary:
            ax.plot(t_axis, df[cond], color=COLORS.get(cond, 'gray'),
                    lw=2.5 if 'IL17RD' in cond else 1.2,
                    alpha=1.0 if 'IL17RD' in cond else 0.7,
                    ls='-' if 'KO' in cond else '--', label=cond)
        for tp in timepoints:
            ax.axvline(tp, color='#cccccc', lw=0.8, ls=':')
        ax.set_xlabel('Developmental time (t)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlim(t0, 1.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"figA_divergence_curves_{args.tag}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure B: IL17RD 하이라이트 ─────────────────────────────
    highlight = [c for c in ['IL17RD_KO','IL17RD_OE','PAX6_KO','ASCL1_KO','HES1_KO']
                 if c in df_l2.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'IL17RD vs Reference Genes  [{args.tag}]  (RG-only start)',
                 fontsize=12, fontweight='bold')
    for ax, (df, title) in zip(axes, [(df_l2, 'Paired L2'), (df_swd, 'Sliced Wasserstein')]):
        for cond in conditions:
            if cond not in highlight:
                ax.plot(t_axis, df[cond], color='#dddddd', lw=0.8, alpha=0.5)
        for cond in highlight:
            ax.plot(t_axis, df[cond], color=COLORS.get(cond, 'gray'),
                    lw=3.0 if 'IL17RD' in cond else 1.8,
                    ls='-' if 'KO' in cond else '--', label=cond)
        for tp in timepoints:
            ax.axvline(tp, color='#bbbbbb', lw=0.8, ls=':')
        ax.set_xlabel('Developmental time (t)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlim(t0, 1.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"figB_IL17RD_highlight_{args.tag}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure C: ranking bar chart ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Final Divergence at t=1.0  [{args.tag}]  (RG-only start)',
                 fontsize=12, fontweight='bold')
    for ax, (df, title) in zip(axes, [(df_l2, 'L2 at t=1'), (df_swd, 'SWD at t=1')]):
        cond_sorted = df.iloc[-1].sort_values(ascending=False)
        colors_bar  = [COLORS.get(c, '#aaaaaa') for c in cond_sorted.index]
        bars = ax.bar(range(len(cond_sorted)), cond_sorted.values, color=colors_bar, alpha=0.85)
        ax.set_xticks(range(len(cond_sorted)))
        ax.set_xticklabels(cond_sorted.index, rotation=45, ha='right', fontsize=8)
        for i, cond in enumerate(cond_sorted.index):
            if 'IL17RD' in cond:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
        ax.set_ylabel(title)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"figC_final_ranking_{args.tag}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Output: {OUT_DIR}")
    print(f"  figA_divergence_curves_{args.tag}.png")
    print(f"  figB_IL17RD_highlight_{args.tag}.png")
    print(f"  figC_final_ranking_{args.tag}.png")
    print("Done!")

if __name__ == "__main__":
    main()
