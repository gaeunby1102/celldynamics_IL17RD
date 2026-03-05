#!/usr/bin/env python
"""
trial8_validate_model.py  (scdiffeq_env)  —  GPU 1

Trial8 모델 검증 (trial5_validate_model.py와 동일 방법)
방법 1: Pseudo-holdout SWD (전 구간)
방법 2: Trial5 vs Trial8 drift cosine similarity

Output: results/trial8/validation/
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
from scipy.stats import wasserstein_distance, pearsonr, spearmanr
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL8 = BASE / "results" / "trial8"
OUT_DIR = TRIAL8 / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT5 = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

CKPT8 = (TRIAL8 / "train" /
    "trial8_SDE_20260303_170943" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

TIME_KEY  = "age_time_norm"
USE_KEY   = "X_scVI"
N_PROJ    = 100
N_SIM     = 50
N_CELLS_DRIFT = 500

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


def simulate_ode(model, z_init, t_start, t_end, n_steps=50):
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, t_end, n_steps + 1).to(device)
    sde    = model.DiffEq.DiffEq
    orig_g = sde.g
    sde.g  = lambda t, y: torch.zeros_like(orig_g(t, y))
    try:
        traj = model.DiffEq.forward(X0, t_grid)
    finally:
        sde.g = orig_g
    return traj[-1].detach().cpu().numpy()


def sliced_wasserstein(a, b, n_proj=100, seed=0):
    rng  = np.random.default_rng(seed)
    D    = a.shape[1]
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa, pb = a @ dirs.T, b @ dirs.T
    return float(np.mean([wasserstein_distance(pa[:,k], pb[:,k]) for k in range(n_proj)]))


def get_drift(model, z, t_val):
    device = next(model.DiffEq.parameters()).device
    X = torch.tensor(z.astype(np.float32), device=device, requires_grad=True)
    t = torch.tensor(t_val, dtype=torch.float32, device=device)
    mu = model.DiffEq.DiffEq.f(t, X)
    return mu.detach().cpu().numpy()


def main():
    print("=" * 65)
    print("Trial8 Model Validation")
    print(f"  CKPT8: {CKPT8.parent.parent.parent.parent.name}")
    print("=" * 65)

    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    tps   = sorted(adata.obs[TIME_KEY].unique())

    adata_ref = adata.copy()
    adata_ref.obs_names = [str(i) for i in range(adata_ref.n_obs)]

    print("\nLoading models...")
    model5 = load_model(CKPT5, adata_ref)
    model8 = load_model(CKPT8, adata_ref)
    print("  Both loaded.")

    # ── 방법 1: Pseudo-holdout SWD ─────────────────────────────────────
    print("\n" + "─"*65)
    print("[방법 1] Pseudo-holdout SWD")
    print("─"*65)

    intervals = [(tps[i], tps[i+1]) for i in range(len(tps)-1)]
    rows5, rows8 = [], []

    for t_start, t_end in intervals:
        lbl = f"{TP_LABELS.get(round(t_start,4),'?')}→{TP_LABELS.get(round(t_end,4),'?')}"
        z_start = adata[(adata.obs[TIME_KEY]-t_start).abs()<1e-4].obsm[USE_KEY]
        z_end   = adata[(adata.obs[TIME_KEY]-t_end  ).abs()<1e-4].obsm[USE_KEY]

        z_pred5 = simulate_ode(model5, z_start, t_start, t_end, N_SIM)
        z_pred8 = simulate_ode(model8, z_start, t_start, t_end, N_SIM)

        swd_base = sliced_wasserstein(z_start, z_end, N_PROJ)
        swd5     = sliced_wasserstein(z_pred5, z_end, N_PROJ)
        swd8     = sliced_wasserstein(z_pred8, z_end, N_PROJ)

        idx = np.random.default_rng(0).permutation(len(z_end))
        h   = len(z_end)//2
        swd_self = sliced_wasserstein(z_end[idx[:h]], z_end[idx[h:h*2]], N_PROJ)

        imp5 = (swd_base - swd5) / swd_base * 100
        imp8 = (swd_base - swd8) / swd_base * 100

        print(f"\n  [{lbl}]")
        print(f"    Trial5: {swd5:.4f} ({imp5:+.1f}%)  "
              f"Trial8: {swd8:.4f} ({imp8:+.1f}%)  "
              f"Baseline: {swd_base:.4f}")

        rows5.append({'interval':lbl,'t_start':t_start,'SWD_pred':swd5,
                      'SWD_baseline':swd_base,'SWD_self':swd_self,'improvement':imp5})
        rows8.append({'interval':lbl,'t_start':t_start,'SWD_pred':swd8,
                      'SWD_baseline':swd_base,'SWD_self':swd_self,'improvement':imp8})

    df5 = pd.DataFrame(rows5)
    df8 = pd.DataFrame(rows8)
    df5.to_csv(OUT_DIR / "method1_swd_trial5.csv", index=False)
    df8.to_csv(OUT_DIR / "method1_swd_trial8.csv", index=False)

    # 그림: trial5 vs trial8 비교
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows5)); w = 0.25
    ax.bar(x-w,   [r['SWD_pred'] for r in rows5],     w, label='Trial5', color='#42A5F5', alpha=0.85)
    ax.bar(x,     [r['SWD_pred'] for r in rows8],     w, label='Trial8 (tuned)', color='#66BB6A', alpha=0.85)
    ax.bar(x+w,   [r['SWD_baseline'] for r in rows5], w, label='Baseline', color='#EF5350', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r['interval'] for r in rows5], fontsize=10)
    ax.set_ylabel('Sliced Wasserstein Distance')
    ax.set_title('Pseudo-holdout SWD: Trial5 vs Trial8\n(lower = better interpolation)')
    ax.legend(fontsize=9)
    for i, (r5, r8) in enumerate(zip(rows5, rows8)):
        ax.text(i-w, r5['SWD_pred']+0.002, f"{r5['improvement']:+.1f}%",
                ha='center', fontsize=7, color='#1565C0')
        ax.text(i,   r8['SWD_pred']+0.002, f"{r8['improvement']:+.1f}%",
                ha='center', fontsize=7, color='#2E7D32')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "method1_swd_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: method1_swd_comparison.png")

    # ── 방법 2: Drift 비교 (trial5 vs trial8) ─────────────────────────
    print("\n" + "─"*65)
    print("[방법 2] Drift field: Trial5 vs Trial8")
    print("─"*65)

    drift_rows = []
    for t_val in tps:
        lbl  = TP_LABELS.get(round(t_val,4), f"{t_val:.4f}")
        mask = (adata.obs[TIME_KEY]-t_val).abs()<1e-4
        z_tp = adata[mask].obsm[USE_KEY]
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(z_tp), min(N_CELLS_DRIFT, len(z_tp)), replace=False)
        z_s  = z_tp[idx]

        mu5 = get_drift(model5, z_s, t_val)
        mu8 = get_drift(model8, z_s, t_val)

        cos_sims = np.array([
            1 - cosine(mu5[i], mu8[i])
            for i in range(len(z_s))
            if np.linalg.norm(mu5[i])>1e-8 and np.linalg.norm(mu8[i])>1e-8
        ])
        corr, _ = pearsonr(mu5.flatten(), mu8.flatten())
        mag5 = np.linalg.norm(mu5, axis=1).mean()
        mag8 = np.linalg.norm(mu8, axis=1).mean()

        print(f"\n  [{lbl}] cos={cos_sims.mean():.4f}±{cos_sims.std():.4f}  "
              f"pearson={corr:.4f}  |mu| t5={mag5:.3f} t8={mag8:.3f}")

        drift_rows.append({'timepoint':lbl, 't':t_val,
                           'cosine_sim_mean':cos_sims.mean(),
                           'cosine_sim_std': cos_sims.std(),
                           'pearson_corr':   corr,
                           'mag_trial5':mag5, 'mag_trial8':mag8})

    df_drift = pd.DataFrame(drift_rows)
    df_drift.to_csv(OUT_DIR / "method2_drift_trial5_vs_trial8.csv", index=False)

    # 그림: cosine sim trial5→trial7 vs trial5→trial8
    try:
        df_t7 = pd.read_csv(TRIAL5 / "validation" / "method2_drift_comparison.csv")
        has_t7 = True
    except:
        has_t7 = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Drift Field Agreement: Trial5 vs Trial7/Trial8\n'
                 '(higher = more stable drift field)', fontsize=12, fontweight='bold')

    tp_labels_list = [r['timepoint'] for r in drift_rows]
    cos8 = [r['cosine_sim_mean'] for r in drift_rows]
    axes[0].bar(tp_labels_list, cos8, color='#66BB6A', alpha=0.85, label='Trial5 vs Trial8')
    if has_t7:
        cos7 = df_t7['cosine_sim_mean'].tolist()
        axes[0].plot(tp_labels_list, cos7, 'o--', color='#AB47BC', lw=2,
                     label='Trial5 vs Trial7', markersize=8)
    axes[0].axhline(0.8, color='gray', lw=1.5, ls='--', alpha=0.6, label='cos=0.8')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Drift Direction Agreement')
    axes[0].legend(fontsize=8)

    axes[1].bar(tp_labels_list,
                [r['mag_trial8'] for r in drift_rows],
                color='#66BB6A', alpha=0.85, label='Trial8')
    axes[1].plot(tp_labels_list,
                 [r['mag_trial5'] for r in drift_rows],
                 'o--', color='#42A5F5', lw=2, label='Trial5', markersize=8)
    axes[1].set_ylabel('Mean ||mu||')
    axes[1].set_title('Drift Magnitude')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "method2_drift_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: method2_drift_comparison.png")

    # ── 종합 요약 ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)

    print("\n[방법 1] SWD improvement (Trial5 → Trial8):")
    print(f"  {'구간':<12} {'Trial5':>8} {'Trial8':>8} {'개선':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r5, r8 in zip(rows5, rows8):
        delta = r8['improvement'] - r5['improvement']
        flag  = "✓" if r8['improvement'] > r5['improvement'] else "✗"
        print(f"  {r8['interval']:<12} {r5['improvement']:>+7.1f}% {r8['improvement']:>+7.1f}%"
              f" {delta:>+7.1f}% {flag}")

    print("\n[방법 2] Drift cosine sim (Trial5 vs Trial8):")
    for r in drift_rows:
        flag = "✓" if r['cosine_sim_mean'] >= 0.8 else "⚠"
        print(f"  {r['timepoint']:6}: {r['cosine_sim_mean']:.4f} {flag}")

    # 판정
    improved_intervals = sum(1 for r5,r8 in zip(rows5,rows8) if r8['improvement']>r5['improvement'])
    cos_mean = np.mean([r['cosine_sim_mean'] for r in drift_rows])
    print(f"\n[판정]")
    print(f"  SWD 개선 구간: {improved_intervals}/{len(rows5)}")
    print(f"  Drift cos 평균: {cos_mean:.4f}")
    if improved_intervals >= 3 and cos_mean >= 0.7:
        print("  ✓ Trial8이 trial5보다 개선됨 → perturbation pipeline 진행 권장")
    elif improved_intervals >= 2:
        print("  △ 부분 개선 → perturbation 결과 trial5/7/8 비교 필요")
    else:
        print("  ✗ 개선 미미 → 추가 튜닝 필요")

    with open(OUT_DIR / "summary.txt", 'w') as f:
        f.write("Trial8 Validation Summary\n\n")
        f.write("[방법1] SWD improvement:\n")
        for r5,r8 in zip(rows5, rows8):
            f.write(f"  {r8['interval']}: trial5={r5['improvement']:+.1f}%  "
                    f"trial8={r8['improvement']:+.1f}%\n")
        f.write("\n[방법2] Drift cosine sim (trial5 vs trial8):\n")
        for r in drift_rows:
            f.write(f"  {r['timepoint']}: {r['cosine_sim_mean']:.4f}\n")
        f.write(f"\nSWD improved: {improved_intervals}/{len(rows5)}\n")
        f.write(f"Drift cos mean: {cos_mean:.4f}\n")

    print(f"\n[DONE] {OUT_DIR}")


if __name__ == "__main__":
    main()
