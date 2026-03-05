#!/usr/bin/env python
"""
step3_latent_gene_program.py  (scArches_env)

IL17RD KO perturbation이 scVI latent space에서 만드는 방향(direction)을 찾고,
그 방향에 atlas 세포들을 projection → 유전자 발현 상관관계 분석.

접근 방식:
  1. IL17RD KO direction at START: delta_z = z_ko - z_ctrl  (per-cell, 30-dim)
  2. Atlas t70d/t115d RG 세포들을 이 direction에 project → "perturbation score"
  3. 각 유전자 발현과 perturbation score 간 Pearson/Spearman 상관
  4. 상위 상관 유전자 = IL17RD-associated gene program
  5. GSEA (hallmark/GO gene sets) for ranked gene list

Supplementary:
  - IL17RD KO endpoint effect on ENDPOINT latent distribution (UMAP)
  - Comparison: ctrl vs KO endpoint distributions

Output: results/trial6/gene_expression_recon/gene_program/
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"
OUT    = TRIAL6 / "gene_expression_recon" / "gene_program"
OUT.mkdir(parents=True, exist_ok=True)

PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]
TIME_KEY      = "age_time_norm"
USE_KEY       = "X_scVI"

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}


def get_ko_direction(t0_tag):
    """IL17RD KO의 per-cell latent shift → dominant direction (PC1)"""
    LAT_DIR = TRIAL6 / f"perturb_latents_{t0_tag}"
    z_ctrl = np.load(LAT_DIR / "z_ctrl.npy")
    z_ko   = np.load(LAT_DIR / "z_IL17RD_KO.npy")
    delta  = z_ko - z_ctrl      # [N, 30]
    pca = PCA(n_components=1)
    pca.fit(delta)
    direction = pca.components_[0]   # 30-dim unit vector (PC1 of delta)
    sign = np.sign(delta.mean(0) @ direction)  # KO 방향이 양수
    return direction * sign, delta, z_ctrl


def project_score(z, direction):
    """각 세포를 KO direction에 projection"""
    return z @ direction   # [N]


def correlation_per_gene(expr_mat, scores, method='pearson', batch_size=500):
    """
    expr_mat: [N, G] (dense or use .toarray())
    scores:   [N]
    Returns:  corr [G], pval [G]
    """
    G = expr_mat.shape[1]
    corr = np.zeros(G)
    pval = np.ones(G)

    for start in range(0, G, batch_size):
        end = min(start + batch_size, G)
        if sp.issparse(expr_mat):
            block = expr_mat[:, start:end].toarray()
        else:
            block = expr_mat[:, start:end]

        for j in range(end - start):
            g_expr = block[:, j]
            if g_expr.std() < 1e-8:
                continue
            if method == 'pearson':
                r, p = pearsonr(scores, g_expr)
            else:
                r, p = spearmanr(scores, g_expr)
            corr[start + j] = r
            pval[start + j] = p

    return corr, pval


def plot_top_genes_bar(df_top, cond, out_path, n=20):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#d73027' if r > 0 else '#4575b4' for r in df_top['pearson_r'].values[:n]]
    ax.barh(df_top['gene'].values[:n][::-1],
            df_top['pearson_r'].values[:n][::-1],
            color=colors[::-1], alpha=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Pearson r (gene expr vs KO-direction score)')
    ax.set_title(f'Gene Program: {cond}  (Top {n})', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_endpoint_umap(z_ctrl_ep, z_ko_ep, z_ref, umap_ref, out_path, label):
    """endpoint latents를 reference UMAP에 projection해서 시각화"""
    try:
        import umap
    except ImportError:
        print("  [SKIP] umap-learn not installed")
        return

    # reference UMAP을 이미 atlas에서 계산된 X_umap 사용
    # endpoint latents를 reference와 함께 plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Endpoint Latent Distribution: {label}', fontweight='bold')

    for ax, (z, name, color) in zip(axes,
            [(z_ctrl_ep, 'ctrl', '#1a73e8'),
             (z_ko_ep, 'IL17RD_KO', '#d93025')]):
        # Simple 2D PCA for visualization
        pca2 = PCA(n_components=2)
        pca2.fit(z_ref)
        z2d = pca2.transform(np.vstack([z_ctrl_ep, z_ko_ep]))
        z_c = z2d[:len(z_ctrl_ep)]
        z_k = z2d[len(z_ctrl_ep):]

        axes[0].scatter(z_c[:, 0], z_c[:, 1], s=1, alpha=0.3, c='#1a73e8', label='ctrl')
        axes[0].scatter(z_k[:, 0], z_k[:, 1], s=1, alpha=0.3, c='#d93025', label='IL17RD_KO')
        axes[0].set_title('PCA of endpoint latents')
        axes[0].legend(markerscale=5, fontsize=8)
        break

    # Density comparison in PC1
    pca1 = PCA(n_components=2)
    pca1.fit(np.vstack([z_ctrl_ep, z_ko_ep]))
    c_pc = pca1.transform(z_ctrl_ep)[:, 0]
    k_pc = pca1.transform(z_ko_ep)[:, 0]
    axes[1].hist(c_pc, bins=50, alpha=0.5, color='#1a73e8', density=True, label='ctrl')
    axes[1].hist(k_pc, bins=50, alpha=0.5, color='#d93025', density=True, label='IL17RD_KO')
    axes[1].set_xlabel('PC1 of endpoint latents')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution shift in PC1')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 65)
    print("Step 3: Latent Gene Program Analysis")
    print("=" * 65)

    # ── 1. Atlas adata 로드 (logcounts 사용) ─────────────────────
    print("\n[1] Loading atlas adata...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    gene_names = np.array(adata.var_names)
    print(f"  shape: {adata.shape}")

    # logcounts 사용 (scVI 학습에 사용된 counts layer)
    if 'logcounts' in adata.layers:
        expr = adata.layers['logcounts']
    elif 'counts' in adata.layers:
        expr = adata.layers['counts'].copy()
        # log1p normalization
        if sp.issparse(expr):
            expr = expr.astype(np.float32)
        else:
            expr = np.log1p(expr.astype(np.float32))
    else:
        expr = adata.X

    print(f"  expr type: {type(expr)}, dtype: {expr.dtype if hasattr(expr,'dtype') else 'unknown'}")

    all_results = {}

    for t0_tag, t0 in T0_OPTIONS.items():
        print(f"\n{'='*50}")
        print(f"[{t0_tag}]  t={t0}")

        # ── 2. IL17RD KO direction ────────────────────────────────
        print("\n  [2] Computing IL17RD KO direction...")
        direction, delta_z, z_start_ctrl = get_ko_direction(t0_tag)
        print(f"  direction L2: {np.linalg.norm(direction):.4f}")
        print(f"  top dims (|loading|): {np.argsort(np.abs(direction))[::-1][:5].tolist()}")

        # ── 3. RG 세포 선택 ──────────────────────────────────────
        mask_tp = (adata.obs[TIME_KEY] - t0).abs() < 1e-4
        mask_rg = adata.obs.get('CellType_refine', adata.obs.get('Cell Type', pd.Series(['Unknown']*adata.n_obs, index=adata.obs_names)))
        if isinstance(mask_rg, pd.Series):
            mask_rg = mask_rg.str.contains('RG|Radial', case=False, na=False)
        else:
            mask_rg = pd.Series(True, index=adata.obs_names)

        mask = mask_tp & mask_rg
        n_rg = mask.sum()
        print(f"  RG cells at t={t0}: {n_rg}")

        if n_rg < 50:
            print(f"  [WARN] Too few RG cells, using all cells at this timepoint")
            mask = mask_tp

        adata_tp = adata[mask]
        z_tp     = adata_tp.obsm[USE_KEY]   # [N_tp, 30]
        n_cells  = z_tp.shape[0]
        print(f"  Cells for analysis: {n_cells}")

        # ── 4. Projection score ───────────────────────────────────
        scores = project_score(z_tp, direction)  # [N_tp]
        print(f"  Perturbation score: mean={scores.mean():.4f}  std={scores.std():.4f}")

        # ── 5. Gene correlation ───────────────────────────────────
        print(f"\n  [3] Computing gene correlations ({n_cells} cells × {len(gene_names)} genes)...")
        mask_arr = np.array(mask)
        if sp.issparse(expr):
            expr_tp = expr[mask_arr]
        else:
            expr_tp = expr[mask_arr]

        corr, pval = correlation_per_gene(expr_tp, scores, method='pearson')
        _, padj, _, _ = multipletests(pval, method='fdr_bh')

        df_corr = pd.DataFrame({
            'gene': gene_names,
            'pearson_r': corr,
            'pval': pval,
            'padj': padj,
        })
        df_corr['abs_r'] = df_corr['pearson_r'].abs()
        df_corr = df_corr.sort_values('abs_r', ascending=False).reset_index(drop=True)
        df_corr.to_csv(OUT / f"gene_program_{t0_tag}.csv", index=False)
        all_results[t0_tag] = df_corr

        # ── 6. Summary ───────────────────────────────────────────
        sig_pos = ((df_corr['padj'] < 0.05) & (df_corr['pearson_r'] > 0)).sum()
        sig_neg = ((df_corr['padj'] < 0.05) & (df_corr['pearson_r'] < 0)).sum()
        print(f"  Sig corr (padj<0.05): positive={sig_pos}  negative={sig_neg}")
        print(f"\n  Top 20 genes (by |Pearson r|):")
        print(f"  {'Gene':15s}  {'r':>8}  {'padj':>10}")
        for _, row in df_corr.head(20).iterrows():
            flag = "*" if row['padj'] < 0.05 else ""
            print(f"  {row['gene']:15s}  {row['pearson_r']:+8.4f}  {row['padj']:10.2e}  {flag}")

        # ── 7. Plot ──────────────────────────────────────────────
        plot_top_genes_bar(df_corr, f"IL17RD_KO_{t0_tag}",
                           OUT / f"fig_gene_program_{t0_tag}.png")

    # ── 8. 두 타임포인트 간 일치도 ─────────────────────────────
    print(f"\n{'='*50}")
    print("[4] Cross-timepoint concordance (t70d vs t115d)...")
    df_70  = all_results.get('t70d_RG')
    df_115 = all_results.get('t115d_RG')
    if df_70 is not None and df_115 is not None:
        merged = df_70[['gene','pearson_r']].merge(
            df_115[['gene','pearson_r']], on='gene', suffixes=('_70d', '_115d'))

        from scipy.stats import spearmanr as spr
        r, p = spr(merged['pearson_r_70d'], merged['pearson_r_115d'])
        print(f"  Gene program Spearman concordance: r={r:.4f}  p={p:.2e}")
        merged.to_csv(OUT / "gene_program_concordance.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(merged['pearson_r_70d'], merged['pearson_r_115d'],
                   s=2, alpha=0.3, c='#555555')
        # IL17RD 위치
        il17rd_row = merged[merged['gene'] == 'IL17RD']
        if len(il17rd_row) > 0:
            ax.scatter(il17rd_row['pearson_r_70d'], il17rd_row['pearson_r_115d'],
                       s=80, c='red', zorder=5, label='IL17RD')
            ax.legend(fontsize=9)
        ax.set_xlabel('Pearson r (t70d RG)')
        ax.set_ylabel('Pearson r (t115d RG)')
        ax.set_title(f'Gene Program Concordance: t70d vs t115d\n(Spearman r={r:.3f})',
                     fontweight='bold')
        plt.tight_layout()
        fig.savefig(OUT / "fig_gene_program_concordance.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ── 9. IL17RD 자체 상관 ──────────────────────────────────────
    print("\n[5] IL17RD self-correlation check:")
    for t0_tag, df_corr in all_results.items():
        row = df_corr[df_corr['gene'] == 'IL17RD']
        if len(row) > 0:
            print(f"  {t0_tag}: IL17RD rank={row.index[0]+1}  "
                  f"r={row['pearson_r'].values[0]:+.4f}  "
                  f"padj={row['padj'].values[0]:.2e}")

    # ── 10. Endpoint latent 시각화 ────────────────────────────────
    print("\n[6] Endpoint distribution visualization...")
    for t0_tag in T0_OPTIONS.keys():
        EP_DIR = TRIAL6 / "gene_expression_recon" / t0_tag / "endpoints"
        if not EP_DIR.exists():
            continue
        z_ctrl_ep = np.load(EP_DIR / "z_ctrl_endpoint.npy")
        z_ko_ep   = EP_DIR / "z_IL17RD_KO_endpoint.npy"
        if not z_ko_ep.exists():
            continue
        z_ko_ep = np.load(z_ko_ep)

        # PCA 2D
        z_ref = adata.obsm[USE_KEY]
        plot_endpoint_umap(z_ctrl_ep, z_ko_ep, z_ref, None,
                           OUT / f"fig_endpoint_pca_{t0_tag}.png",
                           t0_tag)
        print(f"  {t0_tag} endpoint PCA saved.")

    print(f"\n✓ All results saved to: {OUT}")


if __name__ == "__main__":
    main()
