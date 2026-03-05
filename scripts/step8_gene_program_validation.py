#!/usr/bin/env python
"""
step8_gene_program_validation.py  (scArches_env)

Perturbation effect 큰 유전자들 (PAX6, HES1, ASCL1 등)에 대해서도
step3와 동일하게 Gene Program Analysis 수행 → 기존 연구와 비교 검증.

방법:
  1. 각 유전자 KO direction (PC1 of Δz = z_KO - z_ctrl)
  2. Atlas RG 세포들에 projection → Pearson r(score, gene expression)
  3. 결과를 문헌 known targets와 비교 (e.g. PAX6→EOMES, HES1→NEUROG2)

Known biology (validation 기준):
  PAX6 KO  → EOMES↓, TBR1↓, excitatory neuron↓ → 잘못된 cell fate
  HES1 KO  → NEUROG2↑, ASCL1↑, premature neurogenesis
  ASCL1 KO → Inhibitory neuron↓, DLX2↓
  NEUROG2 KO → upper-layer excitatory↓, SATB2↓

Output: results/trial6/gene_expression_recon/gene_program_validation/
    gene_program_{gene}_{t0_tag}.csv
    fig_gene_program_comparison_{t0_tag}.png
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
OUT    = TRIAL6 / "gene_expression_recon" / "gene_program_validation"
OUT.mkdir(parents=True, exist_ok=True)

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}
TIME_KEY = "age_time_norm"
CT_KEY   = "CellType_refine"
USE_KEY  = "X_scVI"

# 분석할 유전자 (IL17RD 포함)
TARGET_GENES = ["IL17RD", "PAX6", "HES1", "ASCL1", "NEUROG2", "DLX2"]

# Known biology: 각 유전자 KO 시 예상되는 상위 상관 유전자들 (문헌 기반)
KNOWN_TARGETS = {
    "PAX6":    {"up": ["EOMES", "TBR2", "SOX2", "HOPX"],        "down": ["TBR1", "SATB2", "BCL11B"]},
    "HES1":    {"up": ["NEUROG2", "ASCL1", "EOMES"],             "down": ["VIM", "GFAP", "HES5"]},
    "ASCL1":   {"up": ["DLX1", "DLX2", "GAD1", "GAD2"],         "down": ["CALB2", "SST", "VIP"]},
    "NEUROG2": {"up": ["TBR1", "SATB2", "EMX1"],                 "down": ["ASCL1", "DLX2", "GAD1"]},
    "DLX2":    {"up": ["GAD1", "GAD2", "SST", "PV"],             "down": ["NEUROG2", "TBR1"]},
    "IL17RD":  {"up": ["FGFR3", "NR2F1"],                        "down": ["AURKA", "CDC20"]},
}


def get_ko_direction(t0_tag, gene):
    """gene KO의 per-cell latent shift → dominant direction (PC1)"""
    LAT_DIR = TRIAL6 / f"perturb_latents_{t0_tag}"
    z_ctrl_path = LAT_DIR / "z_ctrl.npy"
    z_ko_path   = LAT_DIR / f"z_{gene}_KO.npy"

    if not z_ctrl_path.exists() or not z_ko_path.exists():
        return None, None, None

    z_ctrl = np.load(z_ctrl_path)
    z_ko   = np.load(z_ko_path)
    delta  = z_ko - z_ctrl    # [N, 30]

    pca = PCA(n_components=1)
    pca.fit(delta)
    direction = pca.components_[0]
    sign = np.sign(delta.mean(0) @ direction)
    return direction * sign, delta, z_ctrl


def project_score(z, direction):
    return z @ direction


def correlation_per_gene(expr_mat, scores, batch_size=500):
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
            r, p = pearsonr(scores, g_expr)
            corr[start + j] = r
            pval[start + j] = p
    return corr, pval


def check_known_targets(df_corr, gene):
    """Known targets가 상위에 있는지 확인"""
    known = KNOWN_TARGETS.get(gene, {})
    results = {}
    for direction, genes in known.items():
        for kg in genes:
            row = df_corr[df_corr['gene'] == kg]
            if len(row):
                r = row.iloc[0]
                expected_sign = '+' if direction == 'up' else '-'
                actual_sign   = '+' if r['pearson_r'] > 0 else '-'
                match = expected_sign == actual_sign
                results[kg] = {
                    'rank': r.name + 1,
                    'r': r['pearson_r'],
                    'padj': r['padj'],
                    'expected': direction,
                    'match': match,
                }
    return results


def plot_comparison(all_results, t0_tag, out_path):
    """모든 유전자의 top gene program 비교 heatmap"""
    n_genes = len(all_results)
    n_show  = 15

    fig, axes = plt.subplots(1, n_genes, figsize=(4 * n_genes, 6), sharey=False)
    if n_genes == 1:
        axes = [axes]
    fig.suptitle(f'Gene Program Analysis: KO direction correlation  [{t0_tag}]',
                 fontweight='bold')

    for ax, (gene, df) in zip(axes, all_results.items()):
        if df is None:
            ax.set_title(f'{gene} (N/A)')
            continue

        df_top = df.nlargest(n_show // 2, 'pearson_r')
        df_bot = df.nsmallest(n_show // 2, 'pearson_r')
        df_plot = pd.concat([df_top, df_bot]).sort_values('pearson_r', ascending=False)

        colors = ['#d73027' if r > 0 else '#4575b4' for r in df_plot['pearson_r']]
        ax.barh(df_plot['gene'][::-1], df_plot['pearson_r'][::-1],
                color=colors[::-1], alpha=0.85)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(f'{gene} KO', fontweight='bold')
        ax.set_xlabel('Pearson r')
        ax.tick_params(axis='y', labelsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 65)
    print("Step 8: Gene Program Validation  (all target genes)")
    print("=" * 65)

    # ── 1. Atlas 로드 ─────────────────────────────────────────────
    print("\n[1] Loading atlas adata...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    gene_names = np.array(adata.var_names)
    print(f"  Shape: {adata.shape}")

    if 'logcounts' in adata.layers:
        expr = adata.layers['logcounts']
    elif 'counts' in adata.layers:
        import scipy.sparse as sp2
        expr = adata.layers['counts'].copy()
        if sp2.issparse(expr):
            expr = expr.astype(np.float32)
    else:
        expr = adata.X

    summary_rows = []

    for t0_tag, t0 in T0_OPTIONS.items():
        print(f"\n{'='*55}")
        print(f"[{t0_tag}]  t={t0}")

        # RG cells at this timepoint
        mask_t  = (adata.obs[TIME_KEY] - t0).abs() < 1e-4
        mask_rg = adata.obs[CT_KEY].str.contains('RG|Radial', case=False, na=False)
        mask = mask_t & mask_rg
        if mask.sum() < 50:
            mask = mask_t

        adata_tp = adata[mask]
        z_tp     = adata_tp.obsm[USE_KEY]
        n_cells  = z_tp.shape[0]
        print(f"  RG cells: {n_cells}")

        mask_arr = np.array(mask)
        if sp.issparse(expr):
            expr_tp = expr[mask_arr]
        else:
            expr_tp = expr[mask_arr]

        all_results = {}
        validation_summary = {}

        for gene in TARGET_GENES:
            print(f"\n  [{gene}]")

            direction, delta_z, _ = get_ko_direction(t0_tag, gene)
            if direction is None:
                print(f"    [SKIP] latent not found")
                all_results[gene] = None
                continue

            scores = project_score(z_tp, direction)
            print(f"  score: mean={scores.mean():.4f}  std={scores.std():.4f}")

            # Gene correlation
            corr, pval = correlation_per_gene(expr_tp, scores)
            _, padj, _, _ = multipletests(pval, method='fdr_bh')

            df_corr = pd.DataFrame({
                'gene': gene_names,
                'pearson_r': corr,
                'pval': pval,
                'padj': padj,
            }).sort_values('pearson_r', ascending=False).reset_index(drop=True)
            df_corr['abs_r'] = df_corr['pearson_r'].abs()
            df_corr['rank_by_abs'] = df_corr['abs_r'].rank(ascending=False).astype(int)

            df_corr.to_csv(OUT / f"gene_program_{gene}_{t0_tag}.csv", index=False)
            all_results[gene] = df_corr

            # Top 10 출력
            sig_pos = ((df_corr['padj'] < 0.05) & (df_corr['pearson_r'] > 0)).sum()
            sig_neg = ((df_corr['padj'] < 0.05) & (df_corr['pearson_r'] < 0)).sum()
            print(f"  Sig genes (padj<0.05): +{sig_pos} / -{sig_neg}")
            print(f"  Top 10 by |r|:")
            for _, row in df_corr.nlargest(10, 'abs_r').iterrows():
                flag = '*' if row['padj'] < 0.05 else ''
                print(f"    {row['gene']:15s}  r={row['pearson_r']:+.4f}  padj={row['padj']:.2e} {flag}")

            # Known target validation
            val = check_known_targets(df_corr, gene)
            validation_summary[gene] = val
            if val:
                n_match = sum(v['match'] for v in val.values())
                n_total = len(val)
                print(f"\n  Known target validation ({n_match}/{n_total} match):")
                for kg, info in val.items():
                    m = '✅' if info['match'] else '❌'
                    print(f"    {m} {kg:12s}  rank={info['rank']:5d}  "
                          f"r={info['r']:+.4f}  expected={info['expected']}")

            # Summary row
            for _, row in df_corr.head(5).iterrows():
                summary_rows.append({
                    't0_tag': t0_tag, 'perturb_gene': gene,
                    'top_gene': row['gene'], 'pearson_r': row['pearson_r'],
                    'padj': row['padj'],
                })

        # ── 비교 figure ──────────────────────────────────────────
        plot_comparison(all_results, t0_tag,
                        OUT / f"fig_gene_program_comparison_{t0_tag}.png")

        # ── Cross-gene correlation (do all genes agree on direction?) ──
        print(f"\n  Cross-gene program Spearman correlation matrix [{t0_tag}]:")
        valid_genes = [g for g in TARGET_GENES if all_results.get(g) is not None]
        n_vg = len(valid_genes)
        corr_matrix = np.eye(n_vg)
        gene_r_vecs = {}
        for g in valid_genes:
            gene_r_vecs[g] = all_results[g].set_index('gene')['pearson_r']

        print(f"  {'':12s}", end='')
        for g in valid_genes:
            print(f"  {g[:8]:>8}", end='')
        print()
        for i, g1 in enumerate(valid_genes):
            print(f"  {g1:12s}", end='')
            for j, g2 in enumerate(valid_genes):
                if i == j:
                    r = 1.0
                else:
                    common = gene_r_vecs[g1].index.intersection(gene_r_vecs[g2].index)
                    r, _ = spearmanr(gene_r_vecs[g1][common], gene_r_vecs[g2][common])
                    corr_matrix[i, j] = r
                print(f"  {r:8.3f}", end='')
            print()

        # Heatmap
        fig_hm, ax_hm = plt.subplots(figsize=(7, 6))
        im = ax_hm.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax_hm, label='Spearman r')
        ax_hm.set_xticks(range(n_vg))
        ax_hm.set_yticks(range(n_vg))
        ax_hm.set_xticklabels(valid_genes, rotation=45, ha='right')
        ax_hm.set_yticklabels(valid_genes)
        for i in range(n_vg):
            for j in range(n_vg):
                ax_hm.text(j, i, f'{corr_matrix[i,j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if abs(corr_matrix[i,j]) > 0.6 else 'black')
        ax_hm.set_title(f'Gene Program Similarity (KO direction concordance)\n[{t0_tag}]',
                        fontweight='bold')
        plt.tight_layout()
        fig_hm.savefig(OUT / f"fig_gene_program_heatmap_{t0_tag}.png",
                       dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Heatmap saved: fig_gene_program_heatmap_{t0_tag}.png")

    # ── Summary CSV ─────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / "gene_program_validation_summary.csv", index=False)
    print(f"\n✓ All results saved to: {OUT}")


if __name__ == "__main__":
    main()
