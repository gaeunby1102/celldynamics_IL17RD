#!/usr/bin/env python
"""
trial5_step6_drift.py  (scdiffeq_env)

Trial5 scDiffeq 모델로 Drift / Diffusion 퍼터베이션 분석.

  Drift   μ(X) = sde.f(None, X)  → 세포 이동 방향·속도
  Diffusion σ(X) = sde.g(None, X) → 세포 운명 불확실성

Output: results/trial5/drift_diffusion_results/
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
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

TRIAL5  = Path("/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5")
LAT_DIR = TRIAL5 / "perturb_latents"
_OUT_BASE = TRIAL5 / "drift_diffusion_results"

PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

COLORS = {
    'IL17RD_KO': '#1a5276', 'IL17RD_OE': '#5dade2',
    'PAX6_KO':   '#922b21', 'PAX6_OE':   '#f1948a',
    'NEUROG2_KO':'#7d6608', 'NEUROG2_OE':'#d4ac0d',
    'ASCL1_KO':  '#1e8449', 'ASCL1_OE':  '#82e0aa',
    'DLX2_KO':   '#6c3483', 'DLX2_OE':   '#d2b4de',
    'HES1_KO':   '#784212', 'HES1_OE':   '#f0b27a',
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
    model.DiffEq.DiffEq.eval()
    return model

def compute_drift_diffusion(model, z: np.ndarray):
    sde    = model.DiffEq.DiffEq   # PotentialSDE
    device = next(sde.parameters()).device
    X      = torch.tensor(z.astype(np.float32), device=device, requires_grad=True)
    mu     = sde.f(None, X)   # drift (uses autograd internally)
    sigma  = sde.g(None, X)   # diffusion
    return mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()

def drift_metrics(mu_ctrl, mu_pert):
    mu_c = torch.tensor(mu_ctrl); mu_p = torch.tensor(mu_pert)
    cos_sim     = F.cosine_similarity(mu_c, mu_p, dim=1).numpy()
    angle_deg   = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
    mag_ctrl    = np.linalg.norm(mu_ctrl, axis=1)
    mag_pert    = np.linalg.norm(mu_pert, axis=1)
    speed_ratio = mag_pert / (mag_ctrl + 1e-8)
    return {'cos_sim': cos_sim, 'angle_deg': angle_deg,
            'mag_ctrl': mag_ctrl, 'mag_pert': mag_pert, 'speed_ratio': speed_ratio}

def diffusion_metrics(sig_ctrl, sig_pert):
    # sigma shape: (N, D) or (N, D, 1) → flatten to (N, D)
    sig_ctrl = sig_ctrl.reshape(sig_ctrl.shape[0], -1)
    sig_pert = sig_pert.reshape(sig_pert.shape[0], -1)
    diff_ctrl = np.linalg.norm(sig_ctrl, axis=1)
    diff_pert = np.linalg.norm(sig_pert, axis=1)
    return {'diff_ctrl': diff_ctrl, 'diff_pert': diff_pert,
            'delta_diff': diff_pert - diff_ctrl}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--tag",  type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()

    print("=" * 65)
    print("Trial5 Step 6: Drift / Diffusion Perturbation Analysis")
    print("=" * 65)

    # ── 1. 모델 & 데이터 ─────────────────────────────────────────
    print("\n[1] Loading model...")
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

    start_obs  = pd.read_csv(LAT_DIR / "start_cells_obs.csv", index_col=0)
    cell_types = start_obs['CellType_refine'].values

    # ── 2. Drift/Diffusion 계산 ───────────────────────────────────
    print("\n[2] Computing drift & diffusion...")
    z_ctrl = np.load(LAT_DIR / "z_ctrl_t0.npy")
    mu_ctrl, sig_ctrl = compute_drift_diffusion(model, z_ctrl)
    print(f"  ctrl: mu={mu_ctrl.shape}, sigma={sig_ctrl.shape}")

    conditions = {}
    for gene in PERTURB_GENES:
        for ctype, fname in [('KO', f"z_{gene}_KO.npy"), ('OE', f"z_{gene}_OE3x.npy")]:
            path = LAT_DIR / fname
            if not path.exists(): continue
            key = f"{gene}_{ctype}"
            z   = np.load(path)
            mu, sig = compute_drift_diffusion(model, z)
            dm  = drift_metrics(mu_ctrl, mu)
            dfm = diffusion_metrics(sig_ctrl, sig)
            conditions[key] = {'z': z, 'mu': mu, 'sigma': sig, **dm, **dfm}
            print(f"  {key}: cos_sim={dm['cos_sim'].mean():.4f}  "
                  f"speed={dm['speed_ratio'].mean():.4f}  Δdiff={dfm['delta_diff'].mean():.4f}")

    # ── 3. Summary ────────────────────────────────────────────────
    rows = []
    for cond, m in conditions.items():
        rows.append({
            'condition':        cond,
            'mean_cos_sim':     m['cos_sim'].mean(),
            'mean_angle_deg':   m['angle_deg'].mean(),
            'mean_speed_ratio': m['speed_ratio'].mean(),
            'mean_delta_diff':  m['delta_diff'].mean(),
            'pct_cos_lt09':     (m['cos_sim'] < 0.9).mean() * 100,
        })
    df_sum = pd.DataFrame(rows).sort_values('mean_cos_sim')
    df_sum.to_csv(OUT_DIR / "drift_diffusion_summary.csv", index=False)
    print("\n" + df_sum.to_string(index=False))

    cond_list   = list(conditions.keys())
    colors_list = [COLORS.get(c, 'gray') for c in cond_list]

    # ── 4. Figures ───────────────────────────────────────────────
    print("\n[3] Drawing figures...")

    # Fig A: 조건별 평균 지표 bar
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(cond_list))
    for ax, (key, ylabel, title, hline) in zip(axes, [
        ('cos_sim',    'Mean Cosine Similarity',    '(A) Drift Direction\n(1.0=no change)', 1.0),
        ('speed_ratio','Mean Speed Ratio',           '(B) Drift Speed\n(1.0=same)',         1.0),
        ('delta_diff', 'Mean Δ Diffusion',           '(C) Diffusion Change\n(>0=more uncertain)', 0),
    ]):
        vals = [conditions[c][key].mean() for c in cond_list]
        ax.bar(x, vals, color=colors_list, alpha=0.85)
        ax.axhline(hline, color='k', lw=0.8, ls='--')
        ax.set_xticks(x); ax.set_xticklabels(cond_list, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title)
    plt.suptitle('Perturbation Effect on Drift & Diffusion at t=0', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figA_drift_diffusion_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig B: 2D scatter + cell-type breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for cond, color in zip(cond_list, colors_list):
        m = conditions[cond]
        ax.scatter(m['cos_sim'], m['delta_diff'], c=color, s=8, alpha=0.4, rasterized=True)
        ax.scatter(m['cos_sim'].mean(), m['delta_diff'].mean(),
                   c=color, s=120, marker='*', zorder=5,
                   edgecolors='black', linewidths=0.5, label=cond)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axvline(0.95, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Drift Cosine Similarity'); ax.set_ylabel('Δ Diffusion')
    ax.set_title('(D) 운명 변화 2D 분류\n★=mean, ·=each cell')
    ax.legend(fontsize=7, ncol=2, loc='lower left', framealpha=0.8)

    ax = axes[1]
    unique_ct = np.unique(cell_types)
    for ci, cond in enumerate(['IL17RD_KO','IL17RD_OE']):
        if cond not in conditions: continue
        m = conditions[cond]
        ct_means, ct_labels = [], []
        for ct in unique_ct:
            mask = cell_types == ct
            if mask.sum() < 5: continue
            ct_means.append(m['cos_sim'][mask].mean())
            ct_labels.append(ct)
        ax.barh(np.arange(len(ct_labels))+ci*0.35, ct_means, 0.33,
                color=COLORS.get(cond,'gray'), alpha=0.85, label=cond)
    ax.set_yticks(np.arange(len(ct_labels))+0.175)
    ax.set_yticklabels(ct_labels, fontsize=8)
    ax.axvline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Mean Cosine Similarity')
    ax.set_title('(E) IL17RD: Cell Type별 Drift\n(1.0=변화없음)')
    ax.legend(fontsize=8); ax.set_xlim(0.9, 1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figB_2D_and_celltype.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig C: Quiver (ctrl vs IL17RD KO/OE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pca = PCA(n_components=2); pca.fit(z_ctrl)

    def draw_quiver(ax, z, mu, title, color='steelblue', n_grid=20):
        pos_2d   = pca.transform(z)
        drift_2d = mu @ pca.components_.T
        xr = np.linspace(pos_2d[:,0].min(), pos_2d[:,0].max(), n_grid)
        yr = np.linspace(pos_2d[:,1].min(), pos_2d[:,1].max(), n_grid)
        gx, gy, gu, gv = [], [], [], []
        for xi in xr:
            for yi in yr:
                mask = ((np.abs(pos_2d[:,0]-xi) < (xr[1]-xr[0])) &
                        (np.abs(pos_2d[:,1]-yi) < (yr[1]-yr[0])))
                if mask.sum() < 3: continue
                gx.append(xi); gy.append(yi)
                gu.append(drift_2d[mask,0].mean()); gv.append(drift_2d[mask,1].mean())
        ax.scatter(pos_2d[:,0], pos_2d[:,1], c='#e8e8e8', s=2, alpha=0.3, rasterized=True)
        ax.quiver(gx, gy, gu, gv, color=color, alpha=0.8, scale=None, width=0.003, headwidth=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

    draw_quiver(axes[0], z_ctrl, mu_ctrl, 'Control', color='#555555')
    if 'IL17RD_KO' in conditions:
        draw_quiver(axes[1], conditions['IL17RD_KO']['z'],
                    conditions['IL17RD_KO']['mu'], 'IL17RD KO', color=COLORS['IL17RD_KO'])
    if 'IL17RD_OE' in conditions:
        draw_quiver(axes[2], conditions['IL17RD_OE']['z'],
                    conditions['IL17RD_OE']['mu'], 'IL17RD OE 3x', color=COLORS['IL17RD_OE'])
    plt.suptitle('(F) Velocity Field (Quiver) in Latent PCA Space', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figC_quiver_IL17RD.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig D: Violin plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (key, ylabel, title, hline) in zip(axes, [
        ('cos_sim',    'Cosine Similarity', 'Drift Cosine Similarity Distribution', 1.0),
        ('delta_diff', 'Δ Diffusion',       'Δ Diffusion Distribution',             0),
    ]):
        data = [conditions[c][key] for c in cond_list]
        parts = ax.violinplot(data, positions=np.arange(len(cond_list)),
                              showmedians=True, showextrema=False)
        for pc, color in zip(parts['bodies'], colors_list):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        ax.set_xticks(range(len(cond_list)))
        ax.set_xticklabels(cond_list, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.axhline(hline, color='k', ls='--', lw=0.8)
    plt.suptitle('Drift & Diffusion Change: All Conditions', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figD_violin.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── 최종 요약 ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  DRIFT / DIFFUSION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Condition':<18} {'cos_sim':>9} {'angle(°)':>9} {'speed':>8} {'Δdiff':>10}")
    print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*10}")
    for _, row in df_sum.iterrows():
        print(f"  {row['condition']:<18}  {row['mean_cos_sim']:>9.4f}  "
              f"{row['mean_angle_deg']:>9.2f}  {row['mean_speed_ratio']:>8.4f}  "
              f"{row['mean_delta_diff']:>10.4f}")
    print(f"\n  Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
