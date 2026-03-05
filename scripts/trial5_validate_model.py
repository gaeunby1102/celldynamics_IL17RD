#!/usr/bin/env python
"""
trial5_validate_model.py  (scdiffeq_env)  —  GPU 1

Trial5 enforce=1.0 모델 검증 (재학습 없이)

[방법 1] Pseudo-holdout SWD  — 인접 타임포인트 interpolation
  t=0.0   → t=0.1118  (SWD vs actual)
  t=0.1118 → t=0.2471  (SWD vs actual)
  t=0.2471 → t=0.5588  (SWD vs actual, trial6 quasi와 동일)
  t=0.5588 → t=1.0    (SWD vs actual)
  → 116d 구간만 유독 잘 된다면 memorization; 전 구간 고르면 진짜 학습

[방법 2] Drift field 비교  — trial5 vs trial7
  같은 세포에서 두 모델의 drift(mu) 벡터 비교
  → cosine similarity, correlation per timepoint
  → 116d 부근만 다르면 trial5가 116d를 특별 취급(memorize)

Output: results/trial5/validation/
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
from scipy.stats import wasserstein_distance, pearsonr
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
N_PROJ   = 100
N_SIM    = 50
N_CELLS_DRIFT = 500   # drift 비교용 샘플 (속도)


# ── 유틸 ────────────────────────────────────────────────────────────────

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
    """ODE (sigma=0) で t_start → t_end シミュレーション, endpoint 返却"""
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
    return traj[-1].detach().cpu().numpy()   # [N, D]


def sliced_wasserstein(a, b, n_proj=100, seed=0):
    rng  = np.random.default_rng(seed)
    D    = a.shape[1]
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa   = a @ dirs.T
    pb   = b @ dirs.T
    return float(np.mean([wasserstein_distance(pa[:, k], pb[:, k])
                          for k in range(n_proj)]))


def get_drift(model, z, t_val):
    """batch of cells z at time t → drift vectors [N, D]
    potential SDE drift = -grad(psi) → requires grad 필요
    """
    device = next(model.DiffEq.parameters()).device
    X = torch.tensor(z.astype(np.float32), device=device, requires_grad=True)
    t = torch.tensor(t_val, dtype=torch.float32, device=device)
    sde = model.DiffEq.DiffEq
    mu = sde.f(t, X)
    return mu.detach().cpu().numpy()   # [N, D]


# ── 메인 ────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Trial5 Model Validation (no retraining)")
    print("=" * 65)

    # ── 데이터 로드 ──────────────────────────────────────────────────────
    print("\n[0] Loading data...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    tps   = sorted(adata.obs[TIME_KEY].unique())
    print(f"  Cells: {adata.n_obs:,}  |  Timepoints: {[round(t,4) for t in tps]}")

    # model 로드용 ref adata (index reset)
    adata_ref = adata.copy()
    adata_ref.obs_names = [str(i) for i in range(adata_ref.n_obs)]

    # ── 모델 로드 ────────────────────────────────────────────────────────
    print("\n[1] Loading trial5 enforce1 model...")
    model5 = load_model(CKPT5, adata_ref)
    print("  Trial5 loaded.")

    print("\n[2] Loading trial7 holdout model...")
    model7 = load_model(CKPT7, adata_ref)
    print("  Trial7 loaded.")

    # ═══════════════════════════════════════════════════════════════════
    # 방법 1: Pseudo-holdout SWD (전 구간)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("[방법 1] Pseudo-holdout SWD across all intervals")
    print("─" * 65)

    intervals = [(tps[i], tps[i+1]) for i in range(len(tps)-1)]
    # 구간 라벨
    tp_labels = {
        0.0:    "49d",
        0.1118: "60d",
        0.2471: "70d",
        0.5588: "116d",
        1.0:    "168d",
    }

    rows = []
    for t_start, t_end in intervals:
        lbl_s = tp_labels.get(round(t_start, 4), f"{t_start:.4f}")
        lbl_e = tp_labels.get(round(t_end, 4),   f"{t_end:.4f}")
        label = f"{lbl_s}→{lbl_e}"
        print(f"\n  [{label}]  t={t_start:.4f} → t={t_end:.4f}")

        # 시작 / 끝 세포
        z_start = adata[(adata.obs[TIME_KEY] - t_start).abs() < 1e-4].obsm[USE_KEY]
        z_end   = adata[(adata.obs[TIME_KEY] - t_end  ).abs() < 1e-4].obsm[USE_KEY]

        # ODE 예측
        z_pred = simulate_ode(model5, z_start, t_start, t_end, N_SIM)

        # SWD 계산
        swd_pred     = sliced_wasserstein(z_pred,   z_end, N_PROJ)
        swd_baseline = sliced_wasserstein(z_start,  z_end, N_PROJ)

        # self-split (이론적 최소)
        idx  = np.random.default_rng(0).permutation(len(z_end))
        half = len(z_end) // 2
        swd_self = sliced_wasserstein(z_end[idx[:half]], z_end[idx[half:half*2]], N_PROJ)

        improvement = (swd_baseline - swd_pred) / swd_baseline * 100
        print(f"    ODE pred SWD : {swd_pred:.4f}")
        print(f"    Baseline SWD : {swd_baseline:.4f}")
        print(f"    Self-split   : {swd_self:.4f}")
        print(f"    Improvement  : {improvement:+.1f}%")

        rows.append({
            'interval':    label,
            't_start':     t_start,
            't_end':       t_end,
            'SWD_pred':    swd_pred,
            'SWD_baseline':swd_baseline,
            'SWD_self':    swd_self,
            'improvement': improvement,
        })

    df_swd = pd.DataFrame(rows)
    df_swd.to_csv(OUT_DIR / "method1_pseudo_holdout_swd.csv", index=False)

    # 방법 1 그림
    fig, ax = plt.subplots(figsize=(10, 5))
    x   = np.arange(len(rows))
    w   = 0.28
    ax.bar(x - w,   df_swd['SWD_pred'],     w, label='ODE Predicted', color='#42A5F5', alpha=0.85)
    ax.bar(x,       df_swd['SWD_baseline'], w, label='Baseline (start)',color='#EF5350', alpha=0.85)
    ax.bar(x + w,   df_swd['SWD_self'],     w, label='Self-split (min)', color='#66BB6A', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r['interval'] for r in rows], fontsize=10)
    ax.set_ylabel('Sliced Wasserstein Distance')
    ax.set_title('Trial5 Pseudo-holdout Interpolation (ODE)\n'
                 'Uniform improvement = true dynamics; 116d-only spike = memorization')
    ax.legend(fontsize=9)
    for i, r in enumerate(rows):
        ax.text(i - w, r['SWD_pred'] + 0.002,
                f"{r['improvement']:+.1f}%", ha='center', fontsize=8, color='#1565C0')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "method1_pseudo_holdout_swd.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: method1_pseudo_holdout_swd.csv / .png")

    # ═══════════════════════════════════════════════════════════════════
    # 방법 2: Drift field 비교  (trial5 vs trial7)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("[방법 2] Drift field comparison: trial5 vs trial7")
    print("─" * 65)

    drift_rows = []
    all_mu5, all_mu7 = [], []

    for t_val in tps:
        lbl = tp_labels.get(round(t_val, 4), f"{t_val:.4f}")
        mask = (adata.obs[TIME_KEY] - t_val).abs() < 1e-4
        z_tp = adata[mask].obsm[USE_KEY]

        # 샘플링 (속도)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(z_tp), min(N_CELLS_DRIFT, len(z_tp)), replace=False)
        z_s = z_tp[idx]

        mu5 = get_drift(model5, z_s, t_val)   # [N, D]
        mu7 = get_drift(model7, z_s, t_val)   # [N, D]

        # per-cell cosine similarity
        cos_sims = np.array([
            1 - cosine(mu5[i], mu7[i])
            for i in range(len(z_s))
            if np.linalg.norm(mu5[i]) > 1e-8 and np.linalg.norm(mu7[i]) > 1e-8
        ])

        # 전체 flattened correlation
        flat5 = mu5.flatten()
        flat7 = mu7.flatten()
        corr, _ = pearsonr(flat5, flat7)

        # magnitude
        mag5 = np.linalg.norm(mu5, axis=1).mean()
        mag7 = np.linalg.norm(mu7, axis=1).mean()

        print(f"\n  [{lbl}]  t={t_val:.4f}")
        print(f"    Cosine sim (mean±std) : {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
        print(f"    Pearson corr (flattened): {corr:.4f}")
        print(f"    |mu| trial5={mag5:.4f}  trial7={mag7:.4f}")

        drift_rows.append({
            'timepoint':       lbl,
            't':               t_val,
            'cosine_sim_mean': cos_sims.mean(),
            'cosine_sim_std':  cos_sims.std(),
            'pearson_corr':    corr,
            'mag_trial5':      mag5,
            'mag_trial7':      mag7,
        })
        all_mu5.append(mu5)
        all_mu7.append(mu7)

    df_drift = pd.DataFrame(drift_rows)
    df_drift.to_csv(OUT_DIR / "method2_drift_comparison.csv", index=False)

    # 방법 2 그림 A: cosine similarity per timepoint
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Trial5 vs Trial7: Drift Field Comparison\n'
                 '(Low similarity at 116d → trial5 memorized 116d)',
                 fontsize=12, fontweight='bold')

    colors = ['#EF5350' if abs(r['t'] - 0.5588) < 1e-4 else '#42A5F5'
              for r in drift_rows]

    axes[0].bar([r['timepoint'] for r in drift_rows],
                [r['cosine_sim_mean'] for r in drift_rows],
                yerr=[r['cosine_sim_std'] for r in drift_rows],
                color=colors, alpha=0.85, capsize=4)
    axes[0].axhline(0.9, color='gray', lw=1, ls='--', alpha=0.6, label='cos=0.9')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Cosine Similarity (trial5 vs trial7)')
    axes[0].set_title('Drift Direction Agreement\n(red = 116d, blue = others)')
    axes[0].legend(fontsize=8)

    axes[1].bar([r['timepoint'] for r in drift_rows],
                [r['pearson_corr'] for r in drift_rows],
                color=colors, alpha=0.85)
    axes[1].axhline(0.9, color='gray', lw=1, ls='--', alpha=0.6, label='r=0.9')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Drift Magnitude Correlation\n(red = 116d, blue = others)')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "method2_drift_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 방법 2 그림 B: magnitude 비교
    fig, ax = plt.subplots(figsize=(9, 4))
    x  = np.arange(len(drift_rows))
    w  = 0.35
    ax.bar(x - w/2, [r['mag_trial5'] for r in drift_rows], w,
           label='trial5 (all tp)', color='#42A5F5', alpha=0.85)
    ax.bar(x + w/2, [r['mag_trial7'] for r in drift_rows], w,
           label='trial7 (no 116d)', color='#AB47BC', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r['timepoint'] for r in drift_rows])
    ax.set_ylabel('Mean ||mu|| (drift magnitude)')
    ax.set_title('Drift Magnitude: trial5 vs trial7')
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "method2_drift_magnitude.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved: method2_drift_comparison.csv / .png / magnitude.png")

    # ═══════════════════════════════════════════════════════════════════
    # 종합 요약
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print("\n[방법 1] Pseudo-holdout SWD improvement over baseline:")
    for r in rows:
        flag = ""
        if r['improvement'] < 0:
            flag = "  ← 악화 (worse than baseline)"
        elif r['improvement'] > 5:
            flag = "  ← 좋음"
        print(f"  {r['interval']:12s}: {r['improvement']:+5.1f}%{flag}")

    print("\n[방법 2] Drift cosine similarity (trial5 vs trial7):")
    for r in drift_rows:
        flag = "  ← 주의 (different)" if r['cosine_sim_mean'] < 0.8 else ""
        print(f"  {r['timepoint']:6s}: {r['cosine_sim_mean']:.4f}{flag}")

    # 최종 판정
    print("\n[판정]")
    swd_116_improvement = df_swd[df_swd['t_start'].round(4) == 0.2471]['improvement'].values
    other_improvements  = df_swd[df_swd['t_start'].round(4) != 0.2471]['improvement'].values
    drift_116_cos       = df_drift[df_drift['t'].round(4) == 0.5588]['cosine_sim_mean'].values
    drift_other_cos     = df_drift[df_drift['t'].round(4) != 0.5588]['cosine_sim_mean'].values

    print(f"  방법1 — 116d 구간 개선: {swd_116_improvement[0]:+.1f}%  |  "
          f"타 구간 평균: {other_improvements.mean():+.1f}%")
    print(f"  방법2 — 116d drift cos: {drift_116_cos[0]:.4f}  |  "
          f"타 구간 평균: {drift_other_cos.mean():.4f}")

    if (swd_116_improvement[0] - other_improvements.mean() > 3 or
            drift_116_cos[0] < drift_other_cos.mean() - 0.1):
        print("\n  ⚠  116d 구간 특이 패턴 감지 → 일부 memorization 가능성")
        print("     → perturbation 결과 해석 시 116d 타임포인트 주의 필요")
    else:
        print("\n  ✓  전 구간 균일한 패턴 → 진짜 dynamics 학습 가능성 높음")
        print("     → trial5 enforce1 모델 perturbation pipeline 진행 신뢰 가능")

    with open(OUT_DIR / "validation_summary.txt", 'w') as f:
        f.write("Trial5 Model Validation\n\n")
        f.write("[방법1] Pseudo-holdout SWD:\n")
        for r in rows:
            f.write(f"  {r['interval']}: pred={r['SWD_pred']:.4f}  "
                    f"base={r['SWD_baseline']:.4f}  imp={r['improvement']:+.1f}%\n")
        f.write("\n[방법2] Drift cosine similarity (trial5 vs trial7):\n")
        for r in drift_rows:
            f.write(f"  {r['timepoint']}: cos={r['cosine_sim_mean']:.4f}  "
                    f"pearson={r['pearson_corr']:.4f}\n")

    print(f"\n[DONE] Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
