#!/usr/bin/env python
"""
trial6_validate_sde.py  (scdiffeq_env)

4가지 검증:
  A. ODE(sigma=0) vs SDE 궤적 비교
  B. 동일 세포 다중 샘플링 variance
  C. Velocity ratio (mu/sigma norm) 직접 계산
  D. Cell type fidelity (t=1 endpoint vs real 168d distribution)

Output: results/trial6/validation/
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
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5  = BASE / "results" / "trial5"
TRIAL6  = BASE / "results" / "trial6"
OUT_DIR = TRIAL6 / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS       = 100
N_MULTI_CELLS = 50    # B: 다중 샘플링할 세포 수
N_REPEATS     = 20    # B: 반복 횟수
N_RATIO_CELLS = 500   # C: velocity ratio 계산 세포 수
CKPT_PATH     = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

CELLTYPE_COLORS = {
    'RG': '#e41a1c', 'Neuroblast': '#377eb8', 'Ext': '#4daf4a',
    'Fetal_Ext': '#984ea3', 'Fetal_Inh': '#ff7f00', 'Inh': '#a65628',
}

def load_model(adata_ref):
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq
    diffeq  = load_diffeq(ckpt_path=str(CKPT_PATH))
    hparams = dict(diffeq.hparams)
    hparams['time_key'] = 'age_time_norm'
    hparams['use_key']  = 'X_scVI'
    model = sdq.scDiffEq(adata=adata_ref, **hparams)
    model.configure_data(adata=adata_ref)
    model.configure_model(diffeq, loading_existing=True)
    model.DiffEq.eval()
    return model

def simulate_sde(model, z_init, n_steps=N_STEPS, t_start=0.0):
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, 1.0, n_steps + 1).to(device)
    with torch.no_grad():
        traj = model.DiffEq.forward(X0, t_grid)
    return traj.detach().cpu().numpy()

def simulate_ode(model, z_init, n_steps=N_STEPS, t_start=0.0):
    """sigma를 0으로 패칭하여 ODE(결정론적) 궤적 생성"""
    sde = model.DiffEq.DiffEq
    orig_g = sde.g
    sde.g  = lambda t, y: torch.zeros_like(orig_g(t, y))
    try:
        traj = simulate_sde(model, z_init, n_steps, t_start)
    finally:
        sde.g = orig_g
    return traj

def main():
    print("=" * 65)
    print("Trial6 SDE Validation")
    print("=" * 65)

    # ── 데이터 & 모델 로드 ─────────────────────────────────────
    print("\n[Load] Reading data and model...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    model = load_model(adata)
    print("  Model loaded.")

    # t=0 RG 세포 추출
    mask_t0 = adata.obs['age_time_norm'].abs() < 1e-4
    mask_rg = adata.obs['CellType_refine'] == 'RG'
    z_rg    = adata.obsm['X_scVI'][mask_t0 & mask_rg]
    obs_rg  = adata.obs[mask_t0 & mask_rg]
    print(f"  t=0 RG cells: {z_rg.shape[0]}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # A. ODE vs SDE 궤적 비교 (5개 세포, 5번 SDE)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[A] ODE vs SDE trajectory comparison...")
    np.random.seed(0)
    sel_idx = np.random.choice(len(z_rg), 5, replace=False)
    z_sel   = z_rg[sel_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('A. ODE(deterministic) vs SDE(stochastic) Trajectories\n(5 RG cells, t=0→1, dim 0 vs 1)',
                 fontsize=12, fontweight='bold')

    colors_cell = plt.cm.tab10(np.linspace(0, 1, 5))
    for ci, (z_one, col) in enumerate(zip(z_sel, colors_cell)):
        z_one_batch = z_one[None, :]
        # ODE
        ode_traj = simulate_ode(model, z_one_batch)[:, 0, :]  # [T+1, D]
        for ax, (d1, d2, lbl) in zip(axes, [(0,1,'dim0 vs dim1'), (2,3,'dim2 vs dim3')]):
            ax.plot(ode_traj[:, d1], ode_traj[:, d2],
                    color=col, lw=2, ls='-', alpha=0.9,
                    label=f'cell{ci} ODE' if d1==0 else None)
        # SDE x5
        for ri in range(5):
            sde_traj = simulate_sde(model, z_one_batch)[:, 0, :]
            for ax, (d1, d2, lbl) in zip(axes, [(0,1,''), (2,3,'')]):
                ax.plot(sde_traj[:, d1], sde_traj[:, d2],
                        color=col, lw=0.8, ls='--', alpha=0.4)

    for ax, lbl in zip(axes, ['Latent dim 0 vs 1', 'Latent dim 2 vs 3']):
        ax.set_title(lbl)
        ax.set_xlabel('dim A'); ax.set_ylabel('dim B')
        ax.legend(fontsize=6, ncol=2)

    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0],[0],color='k',lw=2,ls='-',label='ODE (drift only)'),
                    Line2D([0],[0],color='k',lw=0.8,ls='--',alpha=0.5,label='SDE (drift+diffusion)')]
    axes[0].legend(handles=legend_elems, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "A_ode_vs_sde.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → A_ode_vs_sde.png saved.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # B. 다중 샘플링 variance (N_MULTI_CELLS 세포 × N_REPEATS 반복)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n[B] Multi-sampling variance ({N_MULTI_CELLS} cells × {N_REPEATS} repeats)...")
    np.random.seed(1)
    z_multi = z_rg[np.random.choice(len(z_rg), N_MULTI_CELLS, replace=False)]

    # 각 세포를 N_REPEATS번 시뮬레이션
    all_trajs = []  # [N_REPEATS, T+1, N_MULTI_CELLS, D]
    for r in range(N_REPEATS):
        traj = simulate_sde(model, z_multi)   # [T+1, N, D]
        all_trajs.append(traj)
        if (r+1) % 5 == 0:
            print(f"  repeat {r+1}/{N_REPEATS}", flush=True)

    all_trajs = np.stack(all_trajs)   # [R, T+1, N, D]
    t_axis = np.linspace(0, 1, N_STEPS + 1)

    # 타임스텝별 variance (across repeats)
    var_per_t = all_trajs.var(axis=0)  # [T+1, N, D]
    mean_var_per_t = var_per_t.mean(axis=(1,2))  # [T+1]

    # ODE 끝점과 SDE 끝점 bias
    ode_trajs = []
    for ci in range(min(10, N_MULTI_CELLS)):
        ode_t = simulate_ode(model, z_multi[ci:ci+1])
        ode_trajs.append(ode_t[:, 0, :])
    ode_endpoints = np.stack([t[-1] for t in ode_trajs])   # [10, D]
    sde_endpoints = all_trajs[:, -1, :min(10,N_MULTI_CELLS), :]  # [R, 10, D]
    bias = np.linalg.norm(sde_endpoints.mean(0) - ode_endpoints, axis=1).mean()
    print(f"  Mean SDE variance (t=1): {mean_var_per_t[-1]:.6f}")
    print(f"  SDE mean - ODE bias (L2): {bias:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'B. Multi-sampling Variance  ({N_MULTI_CELLS} cells × {N_REPEATS} SDE runs)',
                 fontsize=12, fontweight='bold')

    axes[0].plot(t_axis, mean_var_per_t, color='steelblue', lw=2)
    axes[0].fill_between(t_axis, 0, mean_var_per_t, alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Developmental time (t)')
    axes[0].set_ylabel('Mean variance across repeats')
    axes[0].set_title('Trajectory Variance over Time\n(large = diffusion dominant)')
    axes[0].set_xlim(0, 1)

    # t=1에서 세포별 variance histogram
    var_at_t1 = all_trajs[:, -1, :, :].var(axis=0).mean(axis=1)  # [N]
    axes[1].hist(var_at_t1, bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    axes[1].axvline(var_at_t1.mean(), color='red', lw=2, ls='--',
                    label=f'mean={var_at_t1.mean():.5f}')
    axes[1].set_xlabel('Per-cell variance at t=1')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Cell-level Variance at t=1')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "B_multisampling_variance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → B_multisampling_variance.png saved.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # C. Velocity ratio 직접 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n[C] Velocity ratio (||mu||² / ||sigma||²) on {N_RATIO_CELLS} cells...")
    device = next(model.DiffEq.parameters()).device
    sde    = model.DiffEq.DiffEq

    # 전체 atlas의 랜덤 세포
    np.random.seed(2)
    sel  = np.random.choice(len(adata), N_RATIO_CELLS, replace=False)
    z_all = adata.obsm['X_scVI'][sel]
    ct_all = adata.obs['CellType_refine'].values[sel]
    t_all  = adata.obs['age_time_norm'].values[sel]

    X_t  = torch.tensor(z_all.astype(np.float32), device=device)
    with torch.no_grad():
        mu_out    = sde.mu(X_t)                    # [N, D]
        sigma_out = sde.sigma(X_t)                 # [N, D, 1] or [N, D]
        if sigma_out.dim() == 3:
            sigma_out = sigma_out.squeeze(-1)      # [N, D]

    mu_norm    = mu_out.norm(dim=1).cpu().numpy()      # [N]
    sigma_norm = sigma_out.norm(dim=1).cpu().numpy()   # [N]
    ratio      = (mu_norm ** 2) / (sigma_norm ** 2 + 1e-8)

    print(f"  ||mu|| : mean={mu_norm.mean():.4f}  std={mu_norm.std():.4f}")
    print(f"  ||sigma||: mean={sigma_norm.mean():.4f}  std={sigma_norm.std():.4f}")
    print(f"  ratio (target=2.5): mean={ratio.mean():.4f}  median={np.median(ratio):.4f}")

    df_ratio = pd.DataFrame({
        'mu_norm': mu_norm, 'sigma_norm': sigma_norm,
        'ratio': ratio, 'celltype': ct_all, 'time': t_all
    })
    df_ratio.to_csv(OUT_DIR / "C_velocity_ratio.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'C. Velocity Ratio = ||drift||² / ||diffusion||²  (target=2.5)',
                 fontsize=12, fontweight='bold')

    axes[0].hist(ratio, bins=50, color='darkorange', alpha=0.8, edgecolor='white')
    axes[0].axvline(2.5, color='red', lw=2, ls='--', label='target=2.5')
    axes[0].axvline(ratio.mean(), color='blue', lw=2, ls='-',
                    label=f'mean={ratio.mean():.2f}')
    axes[0].set_xlabel('||mu||² / ||sigma||²')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Overall ratio distribution')
    axes[0].legend()

    # cell type별 ratio
    ct_order = ['RG', 'Neuroblast', 'Fetal_Ext', 'Fetal_Inh', 'Ext', 'Inh']
    ct_data  = [df_ratio[df_ratio.celltype == ct]['ratio'].values for ct in ct_order
                if ct in df_ratio.celltype.values]
    ct_labels = [ct for ct in ct_order if ct in df_ratio.celltype.values]
    bp = axes[1].boxplot(ct_data, patch_artist=True, notch=False)
    for patch, ct in zip(bp['boxes'], ct_labels):
        patch.set_facecolor(CELLTYPE_COLORS.get(ct, '#aaaaaa'))
        patch.set_alpha(0.8)
    axes[1].axhline(2.5, color='red', lw=1.5, ls='--', label='target=2.5')
    axes[1].set_xticklabels(ct_labels, rotation=30, ha='right')
    axes[1].set_ylabel('Ratio')
    axes[1].set_title('Ratio by cell type')
    axes[1].legend()

    # time별 ratio (scatter)
    sc = axes[2].scatter(t_all, ratio, c=ratio, cmap='RdYlGn', s=10,
                         alpha=0.5, vmin=0, vmax=5)
    axes[2].axhline(2.5, color='red', lw=1.5, ls='--', label='target=2.5')
    plt.colorbar(sc, ax=axes[2], label='ratio')
    axes[2].set_xlabel('Developmental time (t)')
    axes[2].set_ylabel('||mu||² / ||sigma||²')
    axes[2].set_title('Ratio over developmental time')
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "C_velocity_ratio.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → C_velocity_ratio.png saved.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # D. Cell type fidelity (t=1 endpoint vs real 168d)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[D] Cell type fidelity at t=1...")
    # real 168d 분포
    mask_t1 = (adata.obs['age_time_norm'] - 1.0).abs() < 1e-4
    real_ct  = adata.obs.loc[mask_t1, 'CellType_refine'].value_counts(normalize=True)
    print("  Real 168d distribution:")
    print(real_ct.to_string())

    # t=0 RG 전체 시뮬레이션 → t=1 endpoint
    z_ctrl_t0 = z_rg[:200]   # 200개로 제한
    traj_ctrl  = simulate_sde(model, z_ctrl_t0)   # [T+1, 200, D]
    endpoints  = traj_ctrl[-1]   # [200, D]

    # KNN으로 atlas에 매핑
    atlas_latent = adata.obsm['X_scVI']
    atlas_ct     = adata.obs['CellType_refine'].values
    nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
    nn.fit(atlas_latent)
    _, nn_idx = nn.kneighbors(endpoints)   # [200, 5]

    from collections import Counter
    assigned = []
    for row in nn_idx:
        votes = Counter(atlas_ct[row])
        assigned.append(votes.most_common(1)[0][0])
    sim_ct = pd.Series(assigned).value_counts(normalize=True)
    print("\n  Simulated t=1 distribution (from t=0 RG):")
    print(sim_ct.to_string())

    # 비교 bar chart
    all_cts = sorted(set(list(real_ct.index) + list(sim_ct.index)))
    real_v  = [real_ct.get(ct, 0) for ct in all_cts]
    sim_v   = [sim_ct.get(ct, 0) for ct in all_cts]

    x = np.arange(len(all_cts))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, real_v, w, label='Real 168d', alpha=0.8,
                   color=[CELLTYPE_COLORS.get(ct,'#aaa') for ct in all_cts])
    bars2 = ax.bar(x + w/2, sim_v,  w, label='Simulated t=1 (from t=0 RG)',
                   alpha=0.5, color=[CELLTYPE_COLORS.get(ct,'#aaa') for ct in all_cts],
                   edgecolor='black', linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(all_cts, rotation=30, ha='right')
    ax.set_ylabel('Proportion')
    ax.set_title('D. Cell Type Fidelity: Real 168d vs Simulated t=1\n(KNN-assigned from atlas latent)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "D_celltype_fidelity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → D_celltype_fidelity.png saved.")

    print(f"\n{'='*65}")
    print(f"  All validation outputs: {OUT_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
