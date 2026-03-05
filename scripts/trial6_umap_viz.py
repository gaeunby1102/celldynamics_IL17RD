#!/usr/bin/env python
"""
trial6_umap_viz.py  (scdiffeq_env)

5가지 UMAP 시각화:
  A. Trajectory overlay — 궤적 선을 atlas UMAP 위에 overlay
  B. Endpoint density — ctrl vs perturbed t=1 분포 비교
  C. Displacement vectors — ctrl→perturbed 변위 화살표
  D. Perturbation sensitivity score — 세포별 L2를 UMAP에 색으로
  E. Time-lapse panel — 5 타임스텝에서 ctrl vs IL17RD_KO 분포

Output: results/trial6/umap_viz/
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
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5  = BASE / "results" / "trial5"
TRIAL6  = BASE / "results" / "trial6"
LAT_DIR = TRIAL6 / "perturb_latents_t0_RG"   # t=0 RG latents (재사용)
OUT_DIR = TRIAL6 / "umap_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# t=0 RG latent는 trial5 perturb_latents에서 사용 (RG만 따로 없으면 전체 사용)
LAT_DIR_T0 = TRIAL5 / "perturb_latents"

N_STEPS  = 50    # 시각화용 (속도를 위해 줄임)
N_VIZ    = 200   # 시각화할 세포 수

CKPT_PATH = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

CELLTYPE_COLORS = {
    'RG': '#e41a1c', 'Neuroblast': '#377eb8', 'Ext': '#4daf4a',
    'Fetal_Ext': '#984ea3', 'Fetal_Inh': '#ff7f00', 'Inh': '#a65628',
}
GENE_COLORS = {
    'IL17RD_KO': '#1a5276', 'IL17RD_OE': '#5dade2',
    'PAX6_KO': '#922b21',   'ASCL1_KO': '#1e8449',
    'HES1_KO': '#784212',
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

def simulate(model, z_init, n_steps=N_STEPS, t_start=0.0):
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t_start, 1.0, n_steps + 1).to(device)
    with torch.no_grad():
        traj = model.DiffEq.forward(X0, t_grid)
    return traj.detach().cpu().numpy()   # [T+1, N, D]

def latent_to_umap(latent, atlas_latent, atlas_umap, k=5):
    """atlas KNN으로 latent → UMAP 좌표 근사 투영"""
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(atlas_latent)
    _, idx = nn.kneighbors(latent)
    return atlas_umap[idx].mean(axis=1)   # [N, 2]

def traj_to_umap(traj, atlas_latent, atlas_umap, k=5, subsample_steps=10):
    """궤적 [T+1, N, D] → UMAP 좌표 [T+1, N, 2]"""
    T, N, D = traj.shape
    step_idx = np.linspace(0, T-1, subsample_steps, dtype=int)
    umap_traj = np.zeros((subsample_steps, N, 2))
    for i, ti in enumerate(step_idx):
        umap_traj[i] = latent_to_umap(traj[ti], atlas_latent, atlas_umap, k)
    return umap_traj, step_idx

def main():
    print("=" * 65)
    print("Trial6 UMAP Visualization")
    print("=" * 65)

    # ── 데이터 & 모델 로드 ─────────────────────────────────────
    print("\n[Load] Reading data and model...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    model = load_model(adata)

    atlas_latent = adata.obsm['X_scVI']      # [N_atlas, 30]
    atlas_umap   = adata.obsm['X_umap']      # [N_atlas, 2]
    atlas_ct     = adata.obs['CellType_refine'].values
    atlas_t      = adata.obs['age_time_norm'].values

    # t=116d RG latents 사용 (가장 IL17RD 발현 높음)
    lat_dir = TRIAL6 / "perturb_latents_t115d_RG"
    if not lat_dir.exists():
        lat_dir = LAT_DIR_T0
        t_start = 0.0
        print(f"  Using t=0 latents from {lat_dir}")
    else:
        t_start = 0.5588
        print(f"  Using t=116d RG latents from {lat_dir}")

    z_ctrl = np.load(lat_dir / "z_ctrl.npy")
    print(f"  z_ctrl: {z_ctrl.shape}")

    # 시각화용 세포 subsample
    np.random.seed(42)
    sel = np.random.choice(len(z_ctrl), min(N_VIZ, len(z_ctrl)), replace=False)
    z_ctrl_sub = z_ctrl[sel]

    # perturb latents
    conds = {}
    for gene, ctype in [('IL17RD','KO'), ('IL17RD','OE'),
                         ('PAX6','KO'), ('ASCL1','KO'), ('HES1','KO')]:
        fname = f"z_{gene}_KO.npy" if ctype == 'KO' else f"z_{gene}_OE3x.npy"
        path  = lat_dir / fname
        if path.exists():
            conds[f"{gene}_{ctype}"] = np.load(path)[sel]

    print(f"  Conditions: {list(conds.keys())}")

    # 배경 UMAP (atlas 전체)
    bg_umap = atlas_umap
    bg_ct   = atlas_ct

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # A. Trajectory overlay UMAP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[A] Trajectory overlay UMAP...")
    # ctrl + IL17RD_KO + PAX6_KO 궤적
    traj_ctrl = simulate(model, z_ctrl_sub, t_start=t_start)
    umap_ctrl, step_idx = traj_to_umap(traj_ctrl, atlas_latent, atlas_umap, subsample_steps=12)
    t_axis_sub = np.linspace(t_start, 1.0, N_STEPS + 1)[step_idx]

    viz_conds = {k: v for k, v in conds.items() if k in ['IL17RD_KO', 'PAX6_KO']}
    traj_perts = {}
    umap_perts = {}
    for cond, z_pert in viz_conds.items():
        tp = simulate(model, z_pert, t_start=t_start)
        traj_perts[cond] = tp
        umap_perts[cond], _ = traj_to_umap(tp, atlas_latent, atlas_umap, subsample_steps=12)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('A. Trajectory Overlay on Atlas UMAP\n(RG start → t=1, 12 steps sampled)',
                 fontsize=12, fontweight='bold')

    plot_configs = [
        ('ctrl', umap_ctrl, 'gray', 'Control'),
        ('IL17RD_KO', umap_perts.get('IL17RD_KO'), '#1a5276', 'IL17RD_KO'),
        ('PAX6_KO',   umap_perts.get('PAX6_KO'),   '#922b21', 'PAX6_KO'),
    ]

    for ax, (tag, umap_traj, col, title) in zip(axes, plot_configs):
        # 배경 atlas
        for ct in np.unique(bg_ct):
            m = bg_ct == ct
            ax.scatter(bg_umap[m, 0], bg_umap[m, 1],
                       c=CELLTYPE_COLORS.get(ct, '#cccccc'), s=1, alpha=0.15)
        if umap_traj is not None:
            # 궤적 선
            cmap = plt.cm.plasma
            for ci in range(umap_traj.shape[1]):
                for ti in range(len(t_axis_sub) - 1):
                    frac = ti / (len(t_axis_sub) - 1)
                    ax.plot(umap_traj[ti:ti+2, ci, 0], umap_traj[ti:ti+2, ci, 1],
                            color=cmap(frac), lw=0.6, alpha=0.5)
            # 끝점
            ax.scatter(umap_traj[-1, :, 0], umap_traj[-1, :, 1],
                       c=col, s=15, alpha=0.8, zorder=5, label='t=1 endpoint')
            # 시작점
            ax.scatter(umap_traj[0, :, 0], umap_traj[0, :, 1],
                       c='black', s=10, alpha=0.6, zorder=5, marker='x', label='t_start')
        ax.set_title(title); ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "A_trajectory_overlay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → A_trajectory_overlay.png saved.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # B. Endpoint density UMAP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[B] Endpoint density UMAP...")
    ctrl_end_umap = latent_to_umap(traj_ctrl[-1], atlas_latent, atlas_umap)

    n_conds = len(conds)
    fig, axes = plt.subplots(2, n_conds + 1, figsize=(5*(n_conds+1), 10))
    fig.suptitle('B. Endpoint Distribution at t=1 (KDE density on atlas UMAP)',
                 fontsize=12, fontweight='bold')

    umap1_range = (bg_umap[:,0].min(), bg_umap[:,0].max())
    umap2_range = (bg_umap[:,1].min(), bg_umap[:,1].max())
    grid_x = np.linspace(*umap1_range, 100)
    grid_y = np.linspace(*umap2_range, 100)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    def kde_density(pts):
        if len(pts) < 3: return np.zeros(grid_pts.shape[1])
        try:
            kde = gaussian_kde(pts.T, bw_method=0.3)
            return kde(grid_pts)
        except: return np.zeros(grid_pts.shape[1])

    ctrl_dens = kde_density(ctrl_end_umap)

    def plot_density_bg(ax, dens, title, cmap='Blues'):
        ax.scatter(bg_umap[:,0], bg_umap[:,1], c='#eeeeee', s=1, alpha=0.1)
        ax.contourf(xx, yy, dens.reshape(100,100), levels=10, cmap=cmap, alpha=0.7)
        ax.set_title(title, fontsize=9); ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

    plot_density_bg(axes[0, 0], ctrl_dens, 'Control endpoint\n(t=1)', 'Greys')

    for ci, (cond, z_pert) in enumerate(conds.items()):
        traj_pert = simulate(model, z_pert, t_start=t_start)
        pert_end_umap = latent_to_umap(traj_pert[-1], atlas_latent, atlas_umap)
        pert_dens = kde_density(pert_end_umap)

        plot_density_bg(axes[0, ci+1], pert_dens, f'{cond} endpoint\n(t=1)', 'Blues')

        # 차이 density (pert - ctrl)
        diff_dens = pert_dens - ctrl_dens
        vmax = np.abs(diff_dens).max()
        axes[1, ci+1].scatter(bg_umap[:,0], bg_umap[:,1], c='#eeeeee', s=1, alpha=0.1)
        axes[1, ci+1].contourf(xx, yy, diff_dens.reshape(100,100), levels=20,
                                cmap='RdBu_r', alpha=0.8, vmin=-vmax, vmax=vmax)
        axes[1, ci+1].set_title(f'Δdensity: {cond} - ctrl', fontsize=9)
        axes[1, ci+1].set_xlabel('UMAP1'); axes[1, ci+1].set_ylabel('UMAP2')

    axes[1, 0].axis('off')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "B_endpoint_density.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → B_endpoint_density.png saved.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # C. Displacement vector map
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[C] Displacement vector map...")
    ctrl_end_umap = latent_to_umap(traj_ctrl[-1], atlas_latent, atlas_umap)
    start_umap    = latent_to_umap(z_ctrl_sub, atlas_latent, atlas_umap)

    n_conds = len(conds)
    fig, axes = plt.subplots(1, n_conds, figsize=(6*n_conds, 6))
    if n_conds == 1: axes = [axes]
    fig.suptitle('C. Displacement Vectors: ctrl endpoint → perturbed endpoint\n(arrows per cell)',
                 fontsize=12, fontweight='bold')

    for ax, (cond, z_pert) in zip(axes, conds.items()):
        traj_pert = simulate(model, z_pert, t_start=t_start)
        pert_end_umap = latent_to_umap(traj_pert[-1], atlas_latent, atlas_umap)

        displace = pert_end_umap - ctrl_end_umap   # [N, 2]
        mag      = np.linalg.norm(displace, axis=1)

        # 배경
        ax.scatter(bg_umap[:,0], bg_umap[:,1], c='#eeeeee', s=1, alpha=0.1)

        # 화살표 (magnitude로 색)
        norm = mcolors.Normalize(vmin=0, vmax=np.percentile(mag, 95))
        cmap = plt.cm.YlOrRd
        q = ax.quiver(ctrl_end_umap[:,0], ctrl_end_umap[:,1],
                      displace[:,0], displace[:,1],
                      mag, cmap=cmap, norm=norm,
                      alpha=0.8, scale=None, scale_units='xy',
                      angles='xy', width=0.003)
        plt.colorbar(q, ax=ax, label='Displacement magnitude')
        ax.set_title(f'{cond}\nmean displacement={mag.mean():.4f}')
        ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

    plt.tight_layout()
    fig.savefig(OUT_DIR / "C_displacement_vectors.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → C_displacement_vectors.png saved.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # D. Perturbation sensitivity score UMAP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[D] Perturbation sensitivity score UMAP...")
    n_conds = len(conds)
    fig, axes = plt.subplots(1, n_conds, figsize=(6*n_conds, 6))
    if n_conds == 1: axes = [axes]
    fig.suptitle('D. Per-cell Perturbation Sensitivity (L2: ctrl vs pert at t=1)\nshown on start-cell UMAP position',
                 fontsize=12, fontweight='bold')

    start_umap = latent_to_umap(z_ctrl_sub, atlas_latent, atlas_umap)

    for ax, (cond, z_pert) in zip(axes, conds.items()):
        traj_pert = simulate(model, z_pert, t_start=t_start)
        # per-cell L2 at t=1
        diff_t1 = traj_ctrl[-1] - traj_pert[-1]  # [N, D]
        l2_t1   = np.linalg.norm(diff_t1, axis=1)  # [N]

        # 배경
        ax.scatter(bg_umap[:,0], bg_umap[:,1], c='#eeeeee', s=1, alpha=0.1)
        sc = ax.scatter(start_umap[:,0], start_umap[:,1], c=l2_t1,
                        cmap='hot_r', s=20, alpha=0.85,
                        vmin=0, vmax=np.percentile(l2_t1, 95))
        plt.colorbar(sc, ax=ax, label='L2(ctrl, pert) at t=1')
        ax.set_title(f'{cond}\nmean={l2_t1.mean():.4f}  max={l2_t1.max():.4f}')
        ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

    plt.tight_layout()
    fig.savefig(OUT_DIR / "D_sensitivity_score.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → D_sensitivity_score.png saved.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # E. Time-lapse panel (ctrl vs IL17RD_KO, 5 timepoints)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[E] Time-lapse panel UMAP...")
    traj_ctrl_full = simulate(model, z_ctrl_sub, N_STEPS, t_start)
    t_axis_full    = np.linspace(t_start, 1.0, N_STEPS + 1)

    target_t = np.linspace(t_start, 1.0, 6)[1:]  # 5 timepoints (exclude start)
    snap_idx = [np.argmin(np.abs(t_axis_full - t)) for t in target_t]

    # IL17RD_KO 궤적
    il17rd_ko = conds.get('IL17RD_KO')
    pax6_ko   = conds.get('PAX6_KO')

    if il17rd_ko is not None and pax6_ko is not None:
        traj_il17_full = simulate(model, il17rd_ko, N_STEPS, t_start)
        traj_pax6_full = simulate(model, pax6_ko,   N_STEPS, t_start)

        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle('E. Time-lapse UMAP: ctrl / IL17RD_KO / PAX6_KO\n(columns = timepoints)',
                     fontsize=13, fontweight='bold')
        row_labels  = ['Control', 'IL17RD_KO', 'PAX6_KO']
        row_trajs   = [traj_ctrl_full, traj_il17_full, traj_pax6_full]
        row_colors  = ['gray', '#1a5276', '#922b21']

        for ri, (label, traj, col) in enumerate(zip(row_labels, row_trajs, row_colors)):
            for ci, (si, t_val) in enumerate(zip(snap_idx, target_t)):
                ax = axes[ri, ci]
                # 배경
                ax.scatter(bg_umap[:,0], bg_umap[:,1], c='#eeeeee', s=0.5, alpha=0.1)
                # 현재 타임스텝 세포 위치
                snap_umap = latent_to_umap(traj[si], atlas_latent, atlas_umap)
                ax.scatter(snap_umap[:,0], snap_umap[:,1],
                           c=col, s=12, alpha=0.7)
                ax.set_title(f'{label}\nt={t_val:.2f}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        fig.savefig(OUT_DIR / "E_timelapse_panel.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  → E_timelapse_panel.png saved.")
    else:
        print("  IL17RD_KO or PAX6_KO latent not found, skipping E.")

    print(f"\n{'='*65}")
    print(f"  All UMAP viz outputs: {OUT_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
