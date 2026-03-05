#!/usr/bin/env python
"""
trial5_step5_simulate.py  (scdiffeq_env)

Trial5 scDiffeq 모델로 퍼터베이션 궤적 시뮬레이션.
t=0 → t=1 전체 발달 궤적 비교.

Output: results/trial5/perturb_results/
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
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

TRIAL5  = Path("/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5")
LAT_DIR = TRIAL5 / "perturb_latents"
_OUT_BASE = TRIAL5 / "perturb_results"

T_SIM_START = 0.0
T_SIM_END   = 1.0
N_SIM_STEPS = 100

PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

COLORS = {
    'ctrl':         '#95a5a6',
    'IL17RD_KO':    '#1a5276', 'IL17RD_OE':    '#5dade2',
    'PAX6_KO':      '#922b21', 'PAX6_OE':      '#f1948a',
    'NEUROG2_KO':   '#7d6608', 'NEUROG2_OE':   '#f9e79f',
    'ASCL1_KO':     '#1e8449', 'ASCL1_OE':     '#82e0aa',
    'DLX2_KO':      '#6c3483', 'DLX2_OE':      '#d2b4de',
    'HES1_KO':      '#784212', 'HES1_OE':      '#f0b27a',
}

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

def simulate(model, z_init, t0=0.0, t1=1.0, n_steps=100):
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device)
    t_grid = torch.linspace(t0, t1, n_steps + 1).to(device)
    traj   = model.DiffEq.forward(X0, t_grid)
    return traj.detach().cpu().numpy()   # [T, N, D]

def assign_celltype(z_query, z_ref, ct_ref, k=10):
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(z_ref)
    _, idx = knn.kneighbors(z_query)
    return [Counter(ct_ref[i].tolist()).most_common(1)[0][0] for i in idx]

def composition(labels):
    return pd.Series(labels).value_counts(normalize=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: auto-find last.ckpt)")
    p.add_argument("--tag",  type=str, default=None, help="output subdirectory tag (e.g. enforce1)")
    return p.parse_args()

def main():
    args = parse_args()

    print("=" * 65)
    print("Trial5 Step 5: Perturbation Simulation")
    print("=" * 65)

    # ── 1. 모델 & 데이터 로드 ────────────────────────────────────
    print("\n[1] Loading model and data...")
    adata_train = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata_train.obs['original_barcode'] = adata_train.obs_names.copy()
    adata_train.obs_names = [str(i) for i in range(adata_train.n_obs)]

    CKPT = Path(args.ckpt) if args.ckpt else next(TRIAL5.rglob("train/**/last.ckpt"))
    tag = args.tag or "enforce0"
    OUT_DIR = _OUT_BASE / tag
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Checkpoint: {CKPT}")
    print(f"  Output tag: {tag}")
    model = load_model(CKPT, adata_train)
    print("  Model loaded.")

    # kNN 참조용
    z_ref    = adata_train.obsm['X_scVI']
    ct_ref   = adata_train.obs['CellType_refine'].values
    umap_ref = adata_train.obsm['X_umap']
    print(f"  Reference: {adata_train.n_obs:,} cells, {len(np.unique(ct_ref))} cell types")

    knn_umap = NearestNeighbors(n_neighbors=10, n_jobs=-1)
    knn_umap.fit(z_ref)
    def proj_umap(z):
        _, idx = knn_umap.kneighbors(z)
        return umap_ref[idx].mean(axis=1)

    # ── 2. Perturbation latent 로드 ──────────────────────────────
    print("\n[2] Loading perturbation latents...")
    z_ctrl = np.load(LAT_DIR / "z_ctrl_t0.npy")
    print(f"  z_ctrl: {z_ctrl.shape}")

    conditions = {"ctrl": z_ctrl}
    for gene in PERTURB_GENES:
        for ctype, fname in [('KO', f"z_{gene}_KO.npy"), ('OE', f"z_{gene}_OE3x.npy")]:
            path = LAT_DIR / fname
            if path.exists():
                conditions[f"{gene}_{ctype}"] = np.load(path)
                print(f"  {gene}_{ctype}: {conditions[f'{gene}_{ctype}'].shape}")

    # ── 3. 시뮬레이션 ────────────────────────────────────────────
    print(f"\n[3] Simulating t={T_SIM_START} → t={T_SIM_END} ({N_SIM_STEPS} steps)...")
    trajectories = {}
    endpoints    = {}
    ct_endpoints = {}

    for cond, z_init in conditions.items():
        print(f"  {cond}...", flush=True)
        traj = simulate(model, z_init, T_SIM_START, T_SIM_END, N_SIM_STEPS)
        trajectories[cond] = traj
        endpoints[cond]    = traj[-1]
        ct_endpoints[cond] = assign_celltype(traj[-1], z_ref, ct_ref)
        print(f"    → endpoint: {traj[-1].shape}")

    # ── 4. Cell type 구성 비교 ───────────────────────────────────
    print("\n[4] Computing cell type compositions...")
    comp_all = {cond: composition(cts) for cond, cts in ct_endpoints.items()}
    comp_df  = pd.DataFrame(comp_all).fillna(0).reindex(sorted(pd.concat(comp_all.values()).index.unique()))
    comp_df.to_csv(OUT_DIR / "endpoint_composition.csv")
    print(comp_df.round(3).to_string())

    delta_df = comp_df.subtract(comp_df['ctrl'], axis=0).drop(columns='ctrl')
    delta_df.to_csv(OUT_DIR / "endpoint_delta_vs_ctrl.csv")

    all_ct    = comp_df.index.tolist()
    cond_list = list(conditions.keys())

    # ── 5. Figures ───────────────────────────────────────────────
    print("\n[5] Drawing figures...")

    # Fig A: 전체 cell type 구성
    fig, ax = plt.subplots(figsize=(max(12, len(all_ct)*0.7), 6))
    n_conds = len(cond_list)
    x = np.arange(len(all_ct))
    w = 0.8 / n_conds
    for i, cond in enumerate(cond_list):
        vals = [comp_df.loc[ct, cond] if ct in comp_df.index else 0 for ct in all_ct]
        ax.bar(x + i*w - (n_conds-1)*w/2, vals, w,
               label=cond, color=COLORS.get(cond, f'C{i}'), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(all_ct, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Proportion at t=1.0')
    ax.set_title('Gene Perturbation: Cell Type Composition at t=1.0')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figA_endpoint_composition.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig B: Delta 히트맵
    fig, ax = plt.subplots(figsize=(max(10, len(delta_df.columns)*1.2), max(6, len(all_ct)*0.4)))
    vmax = np.abs(delta_df.values).max()
    im = ax.imshow(delta_df.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_yticks(range(len(all_ct))); ax.set_yticklabels(all_ct, fontsize=8)
    ax.set_xticks(range(len(delta_df.columns)))
    ax.set_xticklabels(delta_df.columns, rotation=45, ha='right', fontsize=9)
    plt.colorbar(im, ax=ax, label='Δ Proportion vs Control')
    ax.set_title('Gene Perturbation: Δ Cell Type Proportion vs Control')
    for i in range(len(all_ct)):
        for j in range(len(delta_df.columns)):
            val = delta_df.values[i, j]
            if abs(val) > 0.02:
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        fontsize=6, color='white' if abs(val) > vmax*0.6 else 'black')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figB_delta_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig C: Mean trajectory PCA
    from sklearn.decomposition import PCA
    mean_trajs = {cond: trajectories[cond].mean(axis=1) for cond in cond_list}
    pca = PCA(n_components=2)
    pca.fit(np.vstack(list(mean_trajs.values())))

    fig, ax = plt.subplots(figsize=(8, 7))
    for cond, mean_t in mean_trajs.items():
        pts   = pca.transform(mean_t)
        color = COLORS.get(cond, 'gray')
        lw    = 2.5 if cond == 'ctrl' else 1.5
        ax.plot(pts[:,0], pts[:,1], color=color, lw=lw, alpha=0.9 if cond=='ctrl' else 0.75, label=cond)
        ax.scatter(*pts[0],  s=80, color=color, marker='o', zorder=5)
        ax.scatter(*pts[-1], s=120, color=color, marker='*', zorder=5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Mean Trajectory in Latent PCA Space\n(● t=0, ★ t=1)')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figC_trajectory_pca.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig D: Endpoint UMAP 투영
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    umap_ctrl_ep = proj_umap(endpoints['ctrl'])
    for i, cond in enumerate(cond_list[:8]):
        ax = axes[i]
        ax.scatter(umap_ref[:,0], umap_ref[:,1], c='#e8e8e8', s=0.3, alpha=0.2, rasterized=True)
        ax.scatter(umap_ctrl_ep[:,0], umap_ctrl_ep[:,1], c='#bbbbbb', s=3, alpha=0.4, rasterized=True)
        umap_ep = proj_umap(endpoints[cond])
        ax.scatter(umap_ep[:,0], umap_ep[:,1], c=COLORS.get(cond,'red'), s=4, alpha=0.6, rasterized=True)
        ax.set_title(cond, fontsize=10); ax.axis('off')
    for j in range(len(cond_list), len(axes)):
        axes[j].axis('off')
    plt.suptitle('Endpoint at t=1.0 (projected onto reference UMAP)\nGray=Control, Colored=Perturbation', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figD_endpoint_umap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig E: IL17RD 특별 비교
    il17rd_conds = [c for c in ['ctrl','IL17RD_KO','IL17RD_OE'] if c in comp_df.columns or c=='ctrl']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    x = np.arange(len(all_ct)); w = 0.25
    for i, cond in enumerate(il17rd_conds):
        col = [comp_df.loc[ct, cond] if ct in comp_df.index else 0 for ct in all_ct]
        ax.bar(x+(i-1)*w, col, w, label=cond, color=COLORS.get(cond,f'C{i}'), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(all_ct, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Proportion'); ax.set_title('IL17RD: Endpoint Composition'); ax.legend(fontsize=8)

    ax = axes[1]
    il17rd_delta_conds = [c for c in ['IL17RD_KO','IL17RD_OE'] if c in delta_df.columns]
    for i, cond in enumerate(il17rd_delta_conds):
        ds = delta_df[cond].sort_values()
        ax.barh(np.arange(len(ds))+i*0.35, ds.values, 0.33,
                color=COLORS.get(cond,f'C{i}'), alpha=0.85, label=cond)
    ax.set_yticks(np.arange(len(delta_df[il17rd_delta_conds[0]]))+0.175 if il17rd_delta_conds else [])
    if il17rd_delta_conds:
        ax.set_yticklabels(delta_df[il17rd_delta_conds[0]].sort_values().index, fontsize=7)
    ax.axvline(0, color='k', lw=0.8); ax.set_xlabel('Δ Proportion'); ax.set_title('IL17RD: Δ vs Control')
    ax.legend(fontsize=8)

    ax = axes[2]
    il17rd_traj_conds = ['ctrl'] + [c for c in ['IL17RD_KO','IL17RD_OE'] if c in mean_trajs]
    pca2 = PCA(n_components=2)
    pca2.fit(np.vstack([mean_trajs[c] for c in il17rd_traj_conds]))
    for cond in il17rd_traj_conds:
        pts = pca2.transform(mean_trajs[cond])
        ax.plot(pts[:,0], pts[:,1], color=COLORS.get(cond,'gray'), lw=2, label=cond)
        ax.scatter(*pts[0],  s=80, color=COLORS.get(cond,'gray'), marker='o', zorder=5)
        ax.scatter(*pts[-1], s=120, color=COLORS.get(cond,'gray'), marker='*', zorder=5)
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2'); ax.set_title('IL17RD Trajectory PCA\n(● t=0, ★ t=1)'); ax.legend(fontsize=8)

    plt.suptitle('IL17RD Perturbation Analysis', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figE_IL17RD_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── 요약 ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  PERTURBATION SUMMARY  (t=1.0 Δ vs ctrl)")
    print(f"{'='*65}")
    for cond in delta_df.columns:
        top_up   = delta_df[cond].nlargest(2)
        top_down = delta_df[cond].nsmallest(2)
        print(f"\n  [{cond}]")
        print(f"    ↑ : " + ", ".join([f"{ct}({v:+.3f})" for ct,v in top_up.items()]))
        print(f"    ↓ : " + ", ".join([f"{ct}({v:+.3f})" for ct,v in top_down.items()]))

    print(f"\n  Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
