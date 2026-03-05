#!/usr/bin/env python
"""
evaluate_new_holdout.py  (scdiffeq_env)

학습된 모델 평가: t=0.122 → t=0.165 interpolation
Earth Mover's Distance(EMD/Wasserstein-1) 로 예측 vs 실제 비교.

Usage:
    python evaluate_new_holdout.py --ckpt_dir results/new_run/train/holdout0165_SDE_XXXXX
    python evaluate_new_holdout.py  # 자동으로 최신 exp dir 탐색
"""

# ── PyTorch 2.6+ 호환 패치 ────────────────────────────────────────────────────
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import argparse
import glob
import json
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRAIN_H5AD  = BASE_DIR / "results" / "new_run" / "train_cap10k_holdout0165.h5ad"
HOLDOUT_H5AD= BASE_DIR / "results" / "new_run" / "holdout_t0165.h5ad"
TRAIN_EXP_DIR = BASE_DIR / "results" / "new_run" / "train"
OUT_DIR     = BASE_DIR / "results" / "new_run" / "evaluate"

T_START = 0.121891   # t=0.122 cells → simulate forward
T_END   = 0.165423   # target: t=0.165 holdout

N_SIMS    = 10    # SDE stochastic repeats
N_STEPS   = 100   # ODE/SDE integration steps

# =============================================================================
# Helpers
# =============================================================================

def find_best_ckpt(exp_dir: Path):
    """체크포인트 재귀 탐색 — scDiffeq는 LightningSDE-.../version_0/checkpoints/ 에 저장."""
    # 재귀적으로 모든 .ckpt 파일 탐색
    ckpts = [p for p in exp_dir.rglob("*.ckpt") if p.name != "last.ckpt"]
    if not ckpts:
        # last.ckpt 만 있는 경우
        last_ckpts = list(exp_dir.rglob("last.ckpt"))
        if last_ckpts:
            return last_ckpts[0]
        return None

    # epoch=XXXX-step=YYYY.ckpt 패턴에서 epoch 숫자 추출 → 가장 높은 epoch
    def epoch_of(p):
        try:
            return int(str(p.stem).split('epoch=')[1].split('-')[0])
        except:
            return -1

    best = max(ckpts, key=epoch_of)
    return best


def find_latest_exp_dir(train_dir: Path):
    """가장 최근에 만든 실험 폴더 반환."""
    dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def load_diffeq_model(ckpt_path: Path, adata: ad.AnnData):
    """scDiffeq 모델 로드 (step2 notebook 패턴)."""
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq

    print(f"  Loading checkpoint: {ckpt_path.name}")

    # diffeq 객체 로드 → hparams 추출
    diffeq = load_diffeq(ckpt_path=ckpt_path)
    hparams = dict(diffeq.hparams)
    hparams['time_key'] = 'age_time_norm'
    hparams['use_key']  = 'X_scVI'
    print(f"  latent_dim: {hparams.get('latent_dim', '?')}")

    model = sdq.scDiffEq(adata=adata, **hparams)
    model.configure_data(adata=adata)
    model.configure_model(diffeq, loading_existing=True)
    model.configure_kNN(kNN_key='X_scVI')
    model.DiffEq.eval()
    return model


def simulate_forward(model, z_start: np.ndarray, t_start: float, t_end: float,
                     n_steps: int = 100, n_sims: int = 10):
    """
    z_start [N, D] → SDE forward simulation → endpoint [n_sims*N, D]
    step2 notebook 패턴: model.DiffEq.forward(x0, t_grid)
    """
    device = next(model.DiffEq.parameters()).device
    X0 = torch.tensor(z_start.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, t_end, n_steps + 1).to(device)

    all_endpoints = []
    model.DiffEq.eval()
    for i in range(n_sims):
        print(f"    sim {i+1}/{n_sims} ...", flush=True)
        # PotentialSDE는 autograd 필요 → no_grad 사용 안 함
        traj = model.DiffEq.forward(X0, t_grid)  # [n_steps+1, N, D]
        endpoint = traj[-1].detach().cpu().numpy()  # [N, D]
        all_endpoints.append(endpoint)

    return np.concatenate(all_endpoints, axis=0)  # [n_sims*N, D]


def compute_emd_per_dim(pred: np.ndarray, actual: np.ndarray):
    """각 latent 차원별 Wasserstein-1 거리 계산."""
    D = pred.shape[1]
    emd_per_dim = np.array([
        wasserstein_distance(pred[:, d], actual[:, d])
        for d in range(D)
    ])
    return emd_per_dim


def compute_pca_emd(pred: np.ndarray, actual: np.ndarray, n_components: int = 10):
    """PCA 공간에서 EMD 계산 (모든 세포 함께 PCA fit)."""
    combined = np.vstack([pred, actual])
    pca = PCA(n_components=n_components)
    pca.fit(combined)
    pred_pca   = pca.transform(pred)
    actual_pca = pca.transform(actual)

    emd_pcs = np.array([
        wasserstein_distance(pred_pca[:, i], actual_pca[:, i])
        for i in range(n_components)
    ])
    return emd_pcs, pca


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="Experiment directory (containing 01_checkpoints/)")
    p.add_argument("--n_sims",   type=int, default=N_SIMS)
    p.add_argument("--n_steps",  type=int, default=N_STEPS)
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Interpolation Evaluation: t=0.122 → t=0.165")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. 실험 디렉토리 & 체크포인트 찾기
    # ------------------------------------------------------------------
    if args.ckpt_dir:
        exp_dir = Path(args.ckpt_dir)
    else:
        print(f"\n[1] Auto-detecting latest experiment in: {TRAIN_EXP_DIR}")
        exp_dir = find_latest_exp_dir(TRAIN_EXP_DIR)
        if exp_dir is None:
            print(f"ERROR: No experiment dirs found in {TRAIN_EXP_DIR}")
            sys.exit(1)

    print(f"    Experiment dir: {exp_dir}")

    ckpt_path = find_best_ckpt(exp_dir)
    if ckpt_path is None:
        print(f"ERROR: No checkpoints found in {exp_dir}/01_checkpoints/")
        sys.exit(1)
    print(f"    Checkpoint    : {ckpt_path.name}")

    # ------------------------------------------------------------------
    # 2. 데이터 로드
    # ------------------------------------------------------------------
    print(f"\n[2] Loading data")
    print(f"    Train  : {TRAIN_H5AD}")
    adata_train   = ad.read_h5ad(TRAIN_H5AD)
    print(f"    Holdout: {HOLDOUT_H5AD}")
    adata_holdout = ad.read_h5ad(HOLDOUT_H5AD)

    # t=0.122 cells (start population)
    time_col = 'age_time_norm'
    mask_start = (adata_train.obs[time_col] - T_START).abs() < 1e-4
    adata_start = adata_train[mask_start]
    z_start = adata_start.obsm['X_scVI']
    z_actual = adata_holdout.obsm['X_scVI']

    print(f"    Start cells (t={T_START:.3f}): {z_start.shape[0]:,}")
    print(f"    Actual holdout (t={T_END:.3f}): {z_actual.shape[0]:,}")

    # ------------------------------------------------------------------
    # 3. 모델 로드
    # ------------------------------------------------------------------
    print(f"\n[3] Loading model")
    try:
        model = load_diffeq_model(ckpt_path, adata_train)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. 시뮬레이션
    # ------------------------------------------------------------------
    print(f"\n[4] Simulating t={T_START} → t={T_END}  (n_sims={args.n_sims})")
    z_pred = simulate_forward(
        model, z_start, T_START, T_END,
        n_steps=args.n_steps, n_sims=args.n_sims
    )

    if z_pred is None:
        print("ERROR: Simulation failed.")
        sys.exit(1)

    print(f"    Predicted cells: {z_pred.shape[0]:,}  (shape={z_pred.shape})")

    # ------------------------------------------------------------------
    # 5. EMD 계산
    # ------------------------------------------------------------------
    print(f"\n[5] Computing EMD (Wasserstein-1)")

    # 5a. Per-dimension EMD
    emd_per_dim = compute_emd_per_dim(z_pred, z_actual)
    mean_emd    = emd_per_dim.mean()
    print(f"    Mean EMD (across {z_pred.shape[1]} dims): {mean_emd:.4f}")
    print(f"    Per-dim EMD (min={emd_per_dim.min():.4f}, max={emd_per_dim.max():.4f})")

    # 5b. PCA-space EMD
    emd_pcs, pca = compute_pca_emd(z_pred, z_actual, n_components=10)
    print(f"    PCA-space EMD (PC1~10): {emd_pcs}")
    print(f"    Mean PCA EMD: {emd_pcs.mean():.4f}")

    # Baseline: t=0.122 세포 그 자체와의 EMD (시뮬 없이 가져다 비교)
    emd_baseline = compute_emd_per_dim(z_start, z_actual)
    mean_baseline = emd_baseline.mean()
    print(f"\n    Baseline EMD (t=0.122 vs t=0.165 no simulation): {mean_baseline:.4f}")
    print(f"    Improvement: {mean_baseline - mean_emd:.4f} ({'better' if mean_emd < mean_baseline else 'worse'})")

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    print(f"\n[6] Generating figures → {OUT_DIR}")

    # Figure 1: Per-dim EMD bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    dims = np.arange(len(emd_per_dim))
    ax.bar(dims, emd_per_dim, color='steelblue', alpha=0.8, label='Predicted')
    ax.axhline(emd_per_dim.mean(), color='steelblue', lw=1.5, ls='--', label=f'Mean={mean_emd:.4f}')
    ax.axhline(emd_baseline.mean(), color='gray', lw=1.5, ls='--', label=f'Baseline={mean_baseline:.4f}')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Wasserstein-1 Distance')
    ax.set_title(f'Per-dim EMD: t=0.122→0.165 (predicted vs actual)\nMean={mean_emd:.4f}, Baseline={mean_baseline:.4f}')
    ax.legend()

    # Figure 1b: PCA PC1~10 EMD
    ax2 = axes[1]
    ax2.bar(np.arange(len(emd_pcs)), emd_pcs, color='salmon', alpha=0.8)
    ax2.axhline(emd_pcs.mean(), color='red', ls='--', lw=1.5, label=f'Mean={emd_pcs.mean():.4f}')
    ax2.set_xlabel('PCA Component')
    ax2.set_ylabel('Wasserstein-1 Distance')
    ax2.set_title('PCA-space EMD (PC1~10)')
    ax2.set_xticks(range(len(emd_pcs)))
    ax2.set_xticklabels([f'PC{i+1}' for i in range(len(emd_pcs))], rotation=45)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_emd_per_dim.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: PCA 2D scatter — predicted vs actual vs start
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pca2d = PCA(n_components=2)
    pca2d.fit(np.vstack([z_start, z_pred, z_actual]))
    s_2d = pca2d.transform(z_start)
    p_2d = pca2d.transform(z_pred)
    a_2d = pca2d.transform(z_actual)

    ev = pca2d.explained_variance_ratio_
    xlabel = f'PC1 ({ev[0]*100:.1f}%)'
    ylabel = f'PC2 ({ev[1]*100:.1f}%)'

    def scatter(ax, data, color, label, alpha=0.3, s=5):
        ax.scatter(data[:, 0], data[:, 1], c=color, alpha=alpha, s=s, label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    scatter(axes[0], s_2d, 'blue',  'Start (t=0.122)')
    scatter(axes[0], a_2d, 'orange','Actual (t=0.165)')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Start vs Actual')

    scatter(axes[1], p_2d, 'green', 'Predicted')
    scatter(axes[1], a_2d, 'orange','Actual (t=0.165)')
    axes[1].legend(fontsize=8)
    axes[1].set_title(f'Predicted vs Actual\nEMD={mean_emd:.4f}')

    scatter(axes[2], s_2d, 'blue',  'Start (t=0.122)', alpha=0.2)
    scatter(axes[2], p_2d, 'green', 'Predicted',       alpha=0.3)
    scatter(axes[2], a_2d, 'orange','Actual (t=0.165)',alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[2].set_title('All three')

    plt.suptitle(f'Interpolation: t=0.122 → t=0.165\n(checkpoint: {ckpt_path.name})',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_pca_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Cell type composition comparison (if cell type available)
    if 'cluster_annotated' in adata_holdout.obs.columns:
        # 예측 세포의 cell type → kNN으로 impute (근처 실제 세포의 cell type 가져오기)
        from sklearn.neighbors import KNeighborsClassifier
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        knn_clf.fit(z_actual, adata_holdout.obs['cluster_annotated'].values)
        pred_labels = knn_clf.predict(z_pred)

        actual_ct = adata_holdout.obs['cluster_annotated'].value_counts(normalize=True).sort_index()
        pred_ct_series = pd.Series(pred_labels).value_counts(normalize=True).sort_index()

        all_ct = sorted(set(actual_ct.index) | set(pred_ct_series.index))
        actual_pct = [actual_ct.get(ct, 0) for ct in all_ct]
        pred_pct   = [pred_ct_series.get(ct, 0) for ct in all_ct]

        x = np.arange(len(all_ct))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(all_ct)*0.6 + 2), 5))
        ax.bar(x - w/2, actual_pct, w, label='Actual (t=0.165)', color='orange', alpha=0.8)
        ax.bar(x + w/2, pred_pct,   w, label='Predicted',         color='green',  alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_ct, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Fraction')
        ax.set_title('Cell Type Composition: Predicted vs Actual')
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUT_DIR / "fig3_celltype_composition.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame({
        'latent_dim': np.arange(len(emd_per_dim)),
        'emd_predicted': emd_per_dim,
        'emd_baseline': emd_baseline,
    })
    results_df.to_csv(OUT_DIR / "emd_per_dim.csv", index=False)

    summary = {
        'checkpoint': str(ckpt_path),
        'n_start_cells': z_start.shape[0],
        'n_actual_cells': z_actual.shape[0],
        'n_predicted_cells': z_pred.shape[0],
        'n_sims': args.n_sims,
        'mean_emd_predicted': float(mean_emd),
        'mean_emd_baseline': float(mean_baseline),
        'emd_improvement': float(mean_baseline - mean_emd),
        'mean_pca_emd': float(emd_pcs.mean()),
    }
    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Checkpoint       : {ckpt_path.name}")
    print(f"  Start cells      : {z_start.shape[0]:,}  (t=0.122)")
    print(f"  Actual cells     : {z_actual.shape[0]:,}  (t=0.165)")
    print(f"  Predicted cells  : {z_pred.shape[0]:,}  ({args.n_sims} SDE runs)")
    print(f"  Mean EMD (pred)  : {mean_emd:.4f}")
    print(f"  Mean EMD (base)  : {mean_baseline:.4f}  (no simulation)")
    print(f"  Improvement      : {mean_baseline - mean_emd:.4f}")
    print(f"  Mean PCA EMD     : {emd_pcs.mean():.4f}")
    print(f"\n  Outputs saved to : {OUT_DIR}")
    print(f"    fig1_emd_per_dim.png")
    print(f"    fig2_pca_scatter.png")
    print(f"    fig3_celltype_composition.png")
    print(f"    emd_per_dim.csv")
    print(f"    summary.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
