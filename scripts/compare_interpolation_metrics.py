#!/usr/bin/env python
"""
compare_interpolation_metrics.py  (scdiffeq_env)

EMD(Wasserstein-1) vs Sinkhorn Divergence 비교
모델별 interpolation 성능 수치 비교:
  - 구 모델 (all timepoints, epoch=999): t=0.122 → t=0.165
  - 신 모델 (holdout t=0.165, epoch=1999): t=0.122 → t=0.165
  - Baseline (시뮬레이션 없이 t=0.122 vs t=0.165)

지표:
  (1) Per-dim W1  : 각 latent 차원별 Wasserstein-1 평균 (빠름, 근사)
  (2) Sinkhorn    : geomloss SamplesLoss (multivariate, 훈련 손실과 동일 계열)
"""

# ── PyTorch 호환 패치 ─────────────────────────────────────────────
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
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import wasserstein_distance
from datetime import datetime

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq/results")
NEW_RUN = BASE / "new_run"
OUT_DIR = NEW_RUN / "evaluate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_START = 0.121891
T_END   = 0.165423
N_SIM_STEPS = 50

# =============================================================================
# Metric functions
# =============================================================================

def per_dim_w1(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Per-dimension Wasserstein-1, mean & per-dim array."""
    emd = np.array([wasserstein_distance(pred[:, d], actual[:, d])
                    for d in range(pred.shape[1])])
    return {'mean': float(emd.mean()), 'per_dim': emd}


def sinkhorn_divergence(pred: np.ndarray, actual: np.ndarray,
                        blur: float = 0.05, n_sample: int = 2000,
                        seed: int = 42) -> float:
    """
    geomloss SamplesLoss("sinkhorn") — debiased Sinkhorn divergence.
    blur: 정규화 강도 (ε). 작을수록 EMD에 가까워짐.
    n_sample: 계산 부하 줄이기 위해 서브샘플링.
    """
    from geomloss import SamplesLoss
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur)

    rng = np.random.default_rng(seed)
    ip = rng.choice(len(pred),   size=min(n_sample, len(pred)),   replace=False)
    ia = rng.choice(len(actual), size=min(n_sample, len(actual)), replace=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xp = torch.tensor(pred[ip],   dtype=torch.float32, device=device)
    xa = torch.tensor(actual[ia], dtype=torch.float32, device=device)

    val = loss_fn(xp, xa).item()
    return float(val)


# =============================================================================
# Model loading & simulation
# =============================================================================

def load_model(ckpt_path: Path, adata: ad.AnnData):
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq
    diffeq  = load_diffeq(ckpt_path=ckpt_path)
    hparams = dict(diffeq.hparams)
    hparams['time_key'] = 'age_time_norm'
    hparams['use_key']  = 'X_scVI'
    model   = sdq.scDiffEq(adata=adata, **hparams)
    model.configure_data(adata=adata)
    model.configure_model(diffeq, loading_existing=True)
    model.DiffEq.eval()
    return model


def simulate(model, z_start: np.ndarray, t0: float, t1: float,
             n_steps: int = 50, n_sims: int = 5) -> np.ndarray:
    """SDE forward. returns [n_sims*N, D]"""
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_start.astype(np.float32), device=device)
    t_grid = torch.linspace(t0, t1, n_steps + 1).to(device)
    endpoints = []
    for i in range(n_sims):
        traj = model.DiffEq.forward(X0, t_grid)
        endpoints.append(traj[-1].detach().cpu().numpy())
    return np.concatenate(endpoints, axis=0)


# =============================================================================
# Main
# =============================================================================

def evaluate_one(name: str, ckpt_path: Path,
                 adata_train: ad.AnnData, z_start: np.ndarray,
                 z_actual: np.ndarray,
                 n_sims: int = 5) -> dict:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"{'='*60}")

    model  = load_model(ckpt_path, adata_train)
    z_pred = simulate(model, z_start, T_START, T_END,
                      n_steps=N_SIM_STEPS, n_sims=n_sims)
    print(f"  Predicted: {z_pred.shape}")

    # 지표 계산
    print("  Computing metrics...")
    w1_pred = per_dim_w1(z_pred,   z_actual)
    w1_base = per_dim_w1(z_start,  z_actual)

    print("  Computing Sinkhorn (blur=0.05)...")
    sk_pred_005  = sinkhorn_divergence(z_pred,  z_actual, blur=0.05)
    sk_base_005  = sinkhorn_divergence(z_start, z_actual, blur=0.05)

    print("  Computing Sinkhorn (blur=0.5)...")
    sk_pred_05  = sinkhorn_divergence(z_pred,  z_actual, blur=0.5)
    sk_base_05  = sinkhorn_divergence(z_start, z_actual, blur=0.5)

    result = {
        'name':           name,
        'ckpt':           ckpt_path.name,
        'n_pred_cells':   z_pred.shape[0],
        # W1
        'w1_pred':        w1_pred['mean'],
        'w1_base':        w1_base['mean'],
        'w1_improve':     w1_base['mean'] - w1_pred['mean'],
        'w1_improve_pct': (w1_base['mean'] - w1_pred['mean']) / w1_base['mean'] * 100,
        # Sinkhorn blur=0.05 (EMD에 가까움)
        'sk005_pred':     sk_pred_005,
        'sk005_base':     sk_base_005,
        'sk005_improve':  sk_base_005 - sk_pred_005,
        'sk005_improve_pct': (sk_base_005 - sk_pred_005) / max(sk_base_005, 1e-8) * 100,
        # Sinkhorn blur=0.5 (부드러운 비교)
        'sk05_pred':      sk_pred_05,
        'sk05_base':      sk_base_05,
        'sk05_improve':   sk_base_05 - sk_pred_05,
        'sk05_improve_pct': (sk_base_05 - sk_pred_05) / max(sk_base_05, 1e-8) * 100,
        # per-dim W1 array (for plotting)
        '_w1_per_dim_pred': w1_pred['per_dim'],
        '_w1_per_dim_base': w1_base['per_dim'],
        '_z_pred':          z_pred,
    }
    return result


def main():
    print("=" * 60)
    print("Interpolation Metric Comparison")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[Data] Loading...")
    # 신 모델용 (holdout 제외)
    adata_new = ad.read_h5ad(NEW_RUN / "train_cap10k_holdout0165.h5ad")
    # 구 모델용 (전체 훈련 데이터)
    adata_old = ad.read_h5ad(BASE / "Input_fetal_neuron_trainset_after_scVI_hvg_5010_latent_dim30.h5ad")
    adata_old.obs_names = [str(i) for i in range(adata_old.shape[0])]

    # holdout (실제 t=0.165)
    adata_holdout = ad.read_h5ad(NEW_RUN / "holdout_t0165.h5ad")
    z_actual = adata_holdout.obsm['X_scVI']

    # 출발 세포 (t=0.122)
    mask_new = (adata_new.obs['age_time_norm'] - T_START).abs() < 1e-4
    z_start  = adata_new[mask_new].obsm['X_scVI']
    print(f"  Start cells (t=0.122) : {z_start.shape[0]:,}")
    print(f"  Actual cells (t=0.165): {z_actual.shape[0]:,}")

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    ckpt_new = next((NEW_RUN / "train").rglob("epoch=1999*.ckpt"))
    ckpt_old = BASE / ("train_scVI/dim_test/dim30/"
                       "LightningSDE-FixedPotential-RegularizedVelocityRatio/"
                       "version_0/checkpoints/epoch=999-step=30000.ckpt")

    experiments = [
        ("Old model\n(all tp, ep=999)",  ckpt_old, adata_old),
        ("New model\n(holdout, ep=1999)", ckpt_new, adata_new),
    ]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = []
    for name, ckpt, adata in experiments:
        r = evaluate_one(name, ckpt, adata, z_start, z_actual, n_sims=5)
        results.append(r)

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Model':<28}  {'W1 pred':>8}  {'W1 base':>8}  {'W1 Δ%':>7}  "
          f"{'SK(0.05)':>10}  {'SK base':>8}  {'SK Δ%':>7}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*7}")
    for r in results:
        name_flat = r['name'].replace('\n', ' ')
        print(f"  {name_flat:<28}  "
              f"{r['w1_pred']:>8.4f}  {r['w1_base']:>8.4f}  {r['w1_improve_pct']:>6.1f}%  "
              f"{r['sk005_pred']:>10.4f}  {r['sk005_base']:>8.4f}  {r['sk005_improve_pct']:>6.1f}%")
    print(f"{'='*70}")

    # Baseline (공통)
    print(f"\n  [Baseline — no simulation]")
    print(f"    W1 (t=0.122 vs t=0.165)       : {results[0]['w1_base']:.4f}")
    print(f"    Sinkhorn blur=0.05             : {results[0]['sk005_base']:.4f}")
    print(f"    Sinkhorn blur=0.5              : {results[0]['sk05_base']:.4f}")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    rows = []
    for r in results:
        rows.append({
            'model':            r['name'].replace('\n', ' '),
            'checkpoint':       r['ckpt'],
            'W1_predicted':     r['w1_pred'],
            'W1_baseline':      r['w1_base'],
            'W1_improvement':   r['w1_improve'],
            'W1_improve_pct':   r['w1_improve_pct'],
            'Sinkhorn_blur005_predicted': r['sk005_pred'],
            'Sinkhorn_blur005_baseline':  r['sk005_base'],
            'Sinkhorn_blur005_improve_pct': r['sk005_improve_pct'],
            'Sinkhorn_blur05_predicted':  r['sk05_pred'],
            'Sinkhorn_blur05_baseline':   r['sk05_base'],
            'Sinkhorn_blur05_improve_pct':r['sk05_improve_pct'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metric_comparison.csv", index=False)
    print(f"\n  CSV saved: {OUT_DIR / 'metric_comparison.csv'}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    print("\n[Figure] Drawing...")

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ['#3498db', '#e74c3c']   # old=blue, new=red
    labels = [r['name'] for r in results]

    # ── (A) Bar: W1 comparison ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(results))
    w = 0.3
    bars_pred = ax.bar(x - w/2, [r['w1_pred'] for r in results], w,
                       color=colors, alpha=0.85, label='Predicted')
    bars_base = ax.bar(x + w/2, [r['w1_base'] for r in results], w,
                       color='gray', alpha=0.5, label='Baseline (no sim)')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results], fontsize=8)
    ax.set_ylabel('Mean Wasserstein-1')
    ax.set_title('(A) Per-dim W1 Comparison')
    ax.legend(fontsize=8)
    for bar, r in zip(bars_pred, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{r['w1_improve_pct']:+.1f}%", ha='center', va='bottom', fontsize=8)

    # ── (B) Bar: Sinkhorn blur=0.05 ───────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    bars_pred2 = ax.bar(x - w/2, [r['sk005_pred'] for r in results], w,
                        color=colors, alpha=0.85, label='Predicted')
    bars_base2 = ax.bar(x + w/2, [r['sk005_base'] for r in results], w,
                        color='gray', alpha=0.5, label='Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results], fontsize=8)
    ax.set_ylabel('Sinkhorn Divergence')
    ax.set_title('(B) Sinkhorn (blur=0.05, ~EMD)')
    ax.legend(fontsize=8)
    for bar, r in zip(bars_pred2, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{r['sk005_improve_pct']:+.1f}%", ha='center', va='bottom', fontsize=8)

    # ── (C) Bar: Sinkhorn blur=0.5 ────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    bars_pred3 = ax.bar(x - w/2, [r['sk05_pred'] for r in results], w,
                        color=colors, alpha=0.85, label='Predicted')
    bars_base3 = ax.bar(x + w/2, [r['sk05_base'] for r in results], w,
                        color='gray', alpha=0.5, label='Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results], fontsize=8)
    ax.set_ylabel('Sinkhorn Divergence')
    ax.set_title('(C) Sinkhorn (blur=0.5, smoother)')
    ax.legend(fontsize=8)

    # ── (D) Per-dim W1: old vs new ────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    dims = np.arange(30)
    for i, (r, color) in enumerate(zip(results, colors)):
        ax.plot(dims, r['_w1_per_dim_pred'], color=color, lw=1.5, alpha=0.9,
                label=f"{r['name'].replace(chr(10),' ')} pred (mean={r['w1_pred']:.4f})")
    ax.plot(dims, results[0]['_w1_per_dim_base'], color='gray', lw=1.2,
            ls='--', alpha=0.7, label=f"Baseline (mean={results[0]['w1_base']:.4f})")
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Wasserstein-1')
    ax.set_title('(D) Per-dimension W1: Old vs New model')
    ax.legend(fontsize=8)
    ax.set_xticks(dims[::2])

    # ── (E) Improvement % summary ────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    metric_names = ['W1', 'Sinkhorn\n(blur=0.05)', 'Sinkhorn\n(blur=0.5)']
    for i, (r, color) in enumerate(zip(results, colors)):
        impr = [r['w1_improve_pct'], r['sk005_improve_pct'], r['sk05_improve_pct']]
        ax.bar(np.arange(3) + i*0.3, impr, 0.28,
               color=color, alpha=0.85,
               label=r['name'].replace('\n', ' '))
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(3) + 0.15)
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylabel('Improvement vs Baseline (%)')
    ax.set_title('(E) Improvement % by metric')
    ax.legend(fontsize=7)

    plt.suptitle(
        'Interpolation Evaluation: Old vs New Model  |  t=0.122 → t=0.165\n'
        'EMD (per-dim W1) vs Sinkhorn Divergence (multivariate)',
        fontsize=12, y=1.01
    )
    fig.savefig(OUT_DIR / "fig5_metric_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {OUT_DIR / 'fig5_metric_comparison.png'}")

    print(f"\n[Done] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
