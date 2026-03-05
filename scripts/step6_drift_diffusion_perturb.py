#!/usr/bin/env python
"""
step6_drift_diffusion_perturb.py  (scdiffeq_env)

Perturbation 후 Drift / Diffusion 변화 분석.

  Drift   μ(X) = model.DiffEq.f(None, X)  → 세포 이동 방향·속도
  Diffusion σ(X) = model.DiffEq.g(None, X) → 세포 운명 불확실성

비교 전략:
  A. t=0에서 drift cosine similarity  (방향 변화)
  B. drift magnitude ratio            (속도 변화)
  C. Δ diffusion                      (불확실성 변화)
  D. 2D scatter: cos_sim vs Δ diffusion
  E. Cell type별 drift 변화
  F. Quiver plot (PCA 2D velocity field)
"""

import torch
_orig = torch.load
def _p(*a, **k): k['weights_only']=False; return _orig(*a, **k)
torch.load = _p
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

BASE    = Path("/data2/Atlas_Normal/IL17RD_scdiffeq/results")
NEW_RUN = BASE / "new_run"
LAT_DIR = NEW_RUN / "perturb_latents"
OUT_DIR = NEW_RUN / "drift_diffusion_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

COLORS = {
    'IL17RD_KO': '#1a5276', 'IL17RD_OE': '#5dade2',
    'PAX6_KO':   '#922b21', 'PAX6_OE':   '#f1948a',
    'NEUROG2_KO':'#7d6608', 'NEUROG2_OE':'#d4ac0d',
    'ASCL1_KO':  '#1e8449', 'ASCL1_OE':  '#82e0aa',
    'DLX2_KO':   '#6c3483', 'DLX2_OE':   '#d2b4de',
    'HES1_KO':   '#784212', 'HES1_OE':   '#f0b27a',
}

# =============================================================================

def load_model(ckpt_path, adata_ref):
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq
    diffeq  = load_diffeq(ckpt_path=ckpt_path)
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
    """
    Returns:
      mu    [N, D] - drift vector
      sigma [N, D] - diffusion vector (norm = scalar uncertainty)
    """
    sde = model.DiffEq.DiffEq   # PotentialSDE – has .f() and .g()
    device = next(sde.parameters()).device
    # PotentialSDE.drift() computes autograd.grad(ψ, y), so y needs requires_grad=True
    X = torch.tensor(z.astype(np.float32), device=device, requires_grad=True)
    mu    = sde.f(None, X)   # drift  (uses autograd internally)
    sigma = sde.g(None, X)   # diffusion
    return mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()


def drift_metrics(mu_ctrl, mu_pert):
    """drift 방향·속도 비교 지표."""
    mu_c = torch.tensor(mu_ctrl)
    mu_p = torch.tensor(mu_pert)
    cos_sim   = F.cosine_similarity(mu_c, mu_p, dim=1).numpy()        # [N]
    angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))        # [N]
    mag_ctrl  = np.linalg.norm(mu_ctrl, axis=1)                        # [N]
    mag_pert  = np.linalg.norm(mu_pert, axis=1)                        # [N]
    speed_ratio = mag_pert / (mag_ctrl + 1e-8)                         # [N]
    return {
        'cos_sim':    cos_sim,
        'angle_deg':  angle_deg,
        'mag_ctrl':   mag_ctrl,
        'mag_pert':   mag_pert,
        'speed_ratio':speed_ratio,
    }


def diffusion_metrics(sig_ctrl, sig_pert):
    diff_ctrl = np.linalg.norm(sig_ctrl, axis=1)
    diff_pert = np.linalg.norm(sig_pert, axis=1)
    return {
        'diff_ctrl':  diff_ctrl,
        'diff_pert':  diff_pert,
        'delta_diff': diff_pert - diff_ctrl,   # 양수=불확실성↑
    }


# =============================================================================

def main():
    print("=" * 65)
    print("Step 6: Drift / Diffusion Perturbation Analysis")
    print("=" * 65)

    # ── 1. 모델 & 데이터 로드 ────────────────────────────────────
    print("\n[1] Loading model...")
    adata_train = ad.read_h5ad(NEW_RUN / "train_cap10k_holdout0165.h5ad")
    CKPT = next((NEW_RUN / "train").rglob("epoch=1999*.ckpt"))
    model = load_model(CKPT, adata_train)
    print(f"  {CKPT.name}")

    start_obs = pd.read_csv(LAT_DIR / "start_cells_obs.csv", index_col=0)
    cell_types = start_obs['CellType_refine'].values

    # ── 2. Latent 로드 & Drift/Diffusion 계산 ────────────────────
    print("\n[2] Computing drift & diffusion for all conditions...")
    z_ctrl = np.load(LAT_DIR / "z_ctrl_t0.npy")
    mu_ctrl, sig_ctrl = compute_drift_diffusion(model, z_ctrl)
    print(f"  ctrl: mu={mu_ctrl.shape}, sigma={sig_ctrl.shape}")

    conditions = {}
    for gene in PERTURB_GENES:
        for ctype in ['KO', 'OE']:
            key  = f"{gene}_{ctype}"
            path = LAT_DIR / f"z_{gene}_{'KO' if ctype=='KO' else 'OE3x'}.npy"
            if not path.exists():
                continue
            z = np.load(path)
            mu, sig = compute_drift_diffusion(model, z)
            dm = drift_metrics(mu_ctrl, mu)
            dfm = diffusion_metrics(sig_ctrl, sig)
            conditions[key] = {
                'z': z, 'mu': mu, 'sigma': sig,
                **dm, **dfm
            }
            print(f"  {key}: cos_sim={dm['cos_sim'].mean():.4f}  "
                  f"speed_ratio={dm['speed_ratio'].mean():.4f}  "
                  f"Δdiff={dfm['delta_diff'].mean():.4f}")

    # ── 3. Summary DataFrame ──────────────────────────────────────
    rows = []
    for cond, m in conditions.items():
        rows.append({
            'condition':       cond,
            'mean_cos_sim':    m['cos_sim'].mean(),
            'mean_angle_deg':  m['angle_deg'].mean(),
            'mean_speed_ratio':m['speed_ratio'].mean(),
            'mean_delta_diff': m['delta_diff'].mean(),
            'pct_cos_sim_lt09': (m['cos_sim'] < 0.9).mean() * 100,
        })
    df_summary = pd.DataFrame(rows).sort_values('mean_cos_sim')
    df_summary.to_csv(OUT_DIR / "drift_diffusion_summary.csv", index=False)
    print("\n" + df_summary.to_string(index=False))

    # ── 4. Figures ───────────────────────────────────────────────
    print("\n[3] Drawing figures...")

    cond_list  = list(conditions.keys())
    colors_list = [COLORS.get(c, 'gray') for c in cond_list]

    # ── Fig A: 조건별 평균 지표 비교 (bar) ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(cond_list))

    ax = axes[0]
    vals = [conditions[c]['cos_sim'].mean() for c in cond_list]
    bars = ax.bar(x, vals, color=colors_list, alpha=0.85)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(cond_list, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('(A) Drift Direction Change\n(1.0 = no change)')
    ax.set_ylim(0.9, 1.01)

    ax = axes[1]
    vals = [conditions[c]['speed_ratio'].mean() for c in cond_list]
    ax.bar(x, vals, color=colors_list, alpha=0.85)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(cond_list, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Speed Ratio (pert/ctrl)')
    ax.set_title('(B) Drift Speed Change\n(1.0 = same speed)')

    ax = axes[2]
    vals = [conditions[c]['delta_diff'].mean() for c in cond_list]
    ax.bar(x, vals, color=colors_list, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(cond_list, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Δ Diffusion (pert - ctrl)')
    ax.set_title('(C) Diffusion Change\n(>0 = more uncertain)')

    plt.suptitle('Perturbation Effect on Drift & Diffusion at t=0', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figA_drift_diffusion_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig B: 2D scatter — cos_sim vs Δ diffusion ───────────────
    # (사분면: 운명전환/속도증가/느린전환/미미한변화)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for cond, color in zip(cond_list, colors_list):
        m = conditions[cond]
        ax.scatter(m['cos_sim'], m['delta_diff'],
                   c=color, s=8, alpha=0.4, rasterized=True)
        # 평균점
        ax.scatter(m['cos_sim'].mean(), m['delta_diff'].mean(),
                   c=color, s=120, marker='*', zorder=5,
                   edgecolors='black', linewidths=0.5, label=cond)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axvline(0.95, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Drift Cosine Similarity (방향 유지도)')
    ax.set_ylabel('Δ Diffusion (불확실성 변화)')
    ax.set_title('(D) 운명 변화 분류 2D\n★=mean, ·=each cell')
    # 사분면 레이블
    ax.text(0.88, ax.get_ylim()[1]*0.9, '운명전환\n+불확실성↑', fontsize=7, ha='center', color='gray')
    ax.text(0.97, ax.get_ylim()[1]*0.9, '방향유지\n+불확실성↑', fontsize=7, ha='center', color='gray')
    ax.legend(fontsize=7, ncol=2, loc='lower left', framealpha=0.8)

    # ── Fig B2: Per cell-type cos_sim (E 전략) ───────────────────
    ax = axes[1]
    unique_ct = np.unique(cell_types)
    # IL17RD만 하이라이트
    for ci, cond in enumerate(['IL17RD_KO', 'IL17RD_OE']):
        if cond not in conditions: continue
        m = conditions[cond]
        ct_means = []
        ct_labels = []
        for ct in unique_ct:
            mask = cell_types == ct
            if mask.sum() < 5: continue
            ct_means.append(m['cos_sim'][mask].mean())
            ct_labels.append(ct)
        y = np.arange(len(ct_labels))
        ax.barh(y + ci*0.35, ct_means, 0.33,
                color=COLORS.get(cond,'gray'), alpha=0.85, label=cond)
    ax.set_yticks(np.arange(len(ct_labels)) + 0.175)
    ax.set_yticklabels(ct_labels, fontsize=8)
    ax.axvline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Mean Cosine Similarity')
    ax.set_title('(E) IL17RD: Cell Type별 Drift 변화\n(1.0 = 변화없음)')
    ax.legend(fontsize=8)
    ax.set_xlim(0.9, 1.01)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "figB_2D_and_celltype.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig C: Quiver — ctrl vs IL17RD KO/OE (PCA velocity field) ─
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pca = PCA(n_components=2)
    pca.fit(z_ctrl)

    def draw_quiver(ax, z, mu, title, color='steelblue', n_grid=20):
        pos_2d   = pca.transform(z)          # 세포 위치 [N, 2]
        drift_2d = mu @ pca.components_.T    # drift 투영 [N, 2]

        # 그리드 평균
        x_range = np.linspace(pos_2d[:,0].min(), pos_2d[:,0].max(), n_grid)
        y_range = np.linspace(pos_2d[:,1].min(), pos_2d[:,1].max(), n_grid)

        gx, gy, gu, gv, gc = [], [], [], [], []
        for xi in x_range:
            for yi in y_range:
                mask = ((np.abs(pos_2d[:,0]-xi) < (x_range[1]-x_range[0])) &
                        (np.abs(pos_2d[:,1]-yi) < (y_range[1]-y_range[0])))
                if mask.sum() < 3: continue
                gx.append(xi); gy.append(yi)
                gu.append(drift_2d[mask, 0].mean())
                gv.append(drift_2d[mask, 1].mean())
                gc.append(mask.sum())

        ax.scatter(pos_2d[:,0], pos_2d[:,1], c='#e8e8e8', s=2, alpha=0.3, rasterized=True)
        ax.quiver(gx, gy, gu, gv, color=color, alpha=0.8, scale=None,
                  width=0.003, headwidth=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

    draw_quiver(axes[0], z_ctrl,   mu_ctrl,                   'Control', color='#555555')
    if 'IL17RD_KO' in conditions:
        draw_quiver(axes[1], conditions['IL17RD_KO']['z'],
                    conditions['IL17RD_KO']['mu'], 'IL17RD KO', color=COLORS['IL17RD_KO'])
    if 'IL17RD_OE' in conditions:
        draw_quiver(axes[2], conditions['IL17RD_OE']['z'],
                    conditions['IL17RD_OE']['mu'], 'IL17RD OE 3x', color=COLORS['IL17RD_OE'])

    plt.suptitle('(F) Velocity Field (Quiver) in Latent PCA Space\nArrows = drift direction & magnitude', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figC_quiver_IL17RD.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig D: 전체 조건 cos_sim 분포 violin ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    data_cos  = [conditions[c]['cos_sim']   for c in cond_list]
    parts = ax.violinplot(data_cos, positions=np.arange(len(cond_list)),
                          showmedians=True, showextrema=False)
    for pc, color in zip(parts['bodies'], colors_list):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    ax.set_xticks(range(len(cond_list)))
    ax.set_xticklabels(cond_list, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Drift Cosine Similarity Distribution\n(per cell)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8)

    ax = axes[1]
    data_diff = [conditions[c]['delta_diff'] for c in cond_list]
    parts2 = ax.violinplot(data_diff, positions=np.arange(len(cond_list)),
                           showmedians=True, showextrema=False)
    for pc, color in zip(parts2['bodies'], colors_list):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    ax.set_xticks(range(len(cond_list)))
    ax.set_xticklabels(cond_list, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Δ Diffusion')
    ax.set_title('Δ Diffusion Distribution\n(per cell)')
    ax.axhline(0, color='k', ls='--', lw=0.8)

    plt.suptitle('Drift & Diffusion Change: All Perturbation Conditions', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figD_violin.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── 최종 요약 ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  DRIFT / DIFFUSION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Condition':<18} {'cos_sim':>9} {'angle(°)':>9} {'speed_ratio':>12} {'Δ diffusion':>12}")
    print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*12}  {'-'*12}")
    for _, row in df_summary.iterrows():
        print(f"  {row['condition']:<18}  {row['mean_cos_sim']:>9.4f}  "
              f"{row['mean_angle_deg']:>9.2f}  {row['mean_speed_ratio']:>12.4f}  "
              f"{row['mean_delta_diff']:>12.4f}")
    print(f"\n  Outputs: {OUT_DIR}")
    print("    figA_drift_diffusion_summary.png")
    print("    figB_2D_and_celltype.png")
    print("    figC_quiver_IL17RD.png")
    print("    figD_violin.png")
    print("    drift_diffusion_summary.csv")


if __name__ == "__main__":
    main()
