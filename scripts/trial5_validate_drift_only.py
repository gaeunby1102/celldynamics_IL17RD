#!/usr/bin/env python
"""방법 2만 재실행: drift field 비교 (trial5 vs trial7)"""

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
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL7 = BASE / "results" / "trial7"
OUT_DIR = TRIAL5 / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT5 = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")
CKPT7 = (TRIAL7 / "train" /
    "trial7_SDE_holdout116d_20260302_183835" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

TIME_KEY = "age_time_norm"
USE_KEY  = "X_scVI"
N_CELLS  = 500
TP_LABELS = {0.0:"49d", 0.1118:"60d", 0.2471:"70d", 0.5588:"116d", 1.0:"168d"}


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


def get_drift(model, z, t_val):
    device = next(model.DiffEq.parameters()).device
    X = torch.tensor(z.astype(np.float32), device=device, requires_grad=True)
    t = torch.tensor(t_val, dtype=torch.float32, device=device)
    mu = model.DiffEq.DiffEq.f(t, X)
    return mu.detach().cpu().numpy()


def main():
    print("=" * 60)
    print("[방법 2] Drift field comparison: trial5 vs trial7")
    print("=" * 60)

    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    tps   = sorted(adata.obs[TIME_KEY].unique())

    adata_ref = adata.copy()
    adata_ref.obs_names = [str(i) for i in range(adata_ref.n_obs)]

    print("\nLoading models...")
    model5 = load_model(CKPT5, adata_ref)
    model7 = load_model(CKPT7, adata_ref)
    print("  Both models loaded.")

    rows = []
    for t_val in tps:
        lbl  = TP_LABELS.get(round(t_val, 4), f"{t_val:.4f}")
        mask = (adata.obs[TIME_KEY] - t_val).abs() < 1e-4
        z_tp = adata[mask].obsm[USE_KEY]

        rng = np.random.default_rng(42)
        idx = rng.choice(len(z_tp), min(N_CELLS, len(z_tp)), replace=False)
        z_s = z_tp[idx]

        mu5 = get_drift(model5, z_s, t_val)
        mu7 = get_drift(model7, z_s, t_val)

        cos_sims = np.array([
            1 - cosine(mu5[i], mu7[i])
            for i in range(len(z_s))
            if np.linalg.norm(mu5[i]) > 1e-8 and np.linalg.norm(mu7[i]) > 1e-8
        ])
        corr, _ = pearsonr(mu5.flatten(), mu7.flatten())
        mag5 = np.linalg.norm(mu5, axis=1).mean()
        mag7 = np.linalg.norm(mu7, axis=1).mean()

        print(f"\n  [{lbl}] t={t_val:.4f}")
        print(f"    Cosine sim : {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
        print(f"    Pearson r  : {corr:.4f}")
        print(f"    |mu|  t5={mag5:.4f}  t7={mag7:.4f}")

        rows.append({'timepoint': lbl, 't': t_val,
                     'cosine_sim_mean': cos_sims.mean(),
                     'cosine_sim_std':  cos_sims.std(),
                     'pearson_corr': corr,
                     'mag_trial5': mag5, 'mag_trial7': mag7})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "method2_drift_comparison.csv", index=False)

    # 그림 A: cosine sim + pearson
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Trial5 vs Trial7: Drift Field Comparison\n'
                 '(Low similarity at 116d → trial5 memorized 116d)',
                 fontsize=12, fontweight='bold')
    colors = ['#EF5350' if abs(r['t'] - 0.5588) < 1e-4 else '#42A5F5' for r in rows]

    axes[0].bar([r['timepoint'] for r in rows],
                [r['cosine_sim_mean'] for r in rows],
                yerr=[r['cosine_sim_std'] for r in rows],
                color=colors, alpha=0.85, capsize=4)
    axes[0].axhline(0.9, color='gray', lw=1.5, ls='--', alpha=0.6, label='cos=0.9')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Drift Direction Agreement\n(red=116d, blue=others)')
    axes[0].legend(fontsize=8)

    axes[1].bar([r['timepoint'] for r in rows],
                [r['pearson_corr'] for r in rows],
                color=colors, alpha=0.85)
    axes[1].axhline(0.9, color='gray', lw=1.5, ls='--', alpha=0.6, label='r=0.9')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Drift Magnitude Correlation\n(red=116d, blue=others)')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "method2_drift_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 그림 B: magnitude
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(rows)); w = 0.35
    ax.bar(x-w/2, [r['mag_trial5'] for r in rows], w, label='trial5', color='#42A5F5', alpha=0.85)
    ax.bar(x+w/2, [r['mag_trial7'] for r in rows], w, label='trial7', color='#AB47BC', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([r['timepoint'] for r in rows])
    ax.set_ylabel('Mean ||mu||'); ax.set_title('Drift Magnitude: trial5 vs trial7')
    ax.legend(); plt.tight_layout()
    fig.savefig(OUT_DIR / "method2_drift_magnitude.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("SUMMARY — 방법 2")
    cos_116  = df[df['t'].round(4) == 0.5588]['cosine_sim_mean'].values[0]
    cos_rest = df[df['t'].round(4) != 0.5588]['cosine_sim_mean'].mean()
    print(f"  116d cosine sim : {cos_116:.4f}")
    print(f"  타 구간 평균    : {cos_rest:.4f}")
    if cos_116 < cos_rest - 0.1:
        print("  ⚠  116d에서 drift 방향 차이 → memorization 가능성")
    else:
        print("  ✓  116d 특이 패턴 없음 → trial5 dynamics 신뢰 가능")
    print(f"\n  Saved: {OUT_DIR}/method2_drift_comparison.png")


if __name__ == "__main__":
    main()
