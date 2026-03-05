#!/usr/bin/env python
"""
step2_decode_deg.py  (scArches_env)

Step 1에서 저장한 endpoint latents를 scVI decoder로 gene expression 복원 후
ctrl vs each condition DEG 분석.

Usage:
    python scripts/step2_decode_deg.py --t0_tag t70d_RG
    python scripts/step2_decode_deg.py --t0_tag t115d_RG

Output:
    results/trial6/gene_expression_recon/{t0_tag}/
        deg_{cond}.csv          (all genes, ranked by abs logFC)
        top_deg_summary.csv     (top20 per condition, combined)
        fig_volcano_{cond}.png
        fig_heatmap_top20.png   (IL17RD_KO + IL17RD_OE3x top genes)
"""

import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import scvi
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
SCVI_MODEL_DIR = str(TRIAL5 / "scvi_dim30")

PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

# scVI decode 시 사용할 배치: 70d 세포들의 최빈 배치 사용
# (step1에서 복사한 start_cells_obs.csv로부터 추론)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t0_tag", type=str, required=True,
                   choices=["t70d_RG", "t115d_RG"])
    p.add_argument("--n_top_deg", type=int, default=50,
                   help="Top N DEGs to report per condition")
    p.add_argument("--lib_size", type=float, default=2000.0,
                   help="Fixed library size for decoding (counts)")
    return p.parse_args()


def decode_latents(module, z_np, batch_idx, lib_size, batch_size=512):
    """
    z_np:      [N, 30] numpy array
    batch_idx: int  (scVI batch index)
    lib_size:  float (total counts for library normalization)
    Returns:   [N, n_genes] numpy array (NB mean expression)
    """
    device = next(module.parameters()).device
    module.eval()

    N = len(z_np)
    results = []
    lib_val = np.log(lib_size)

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        z_b   = torch.tensor(z_np[start:end].astype(np.float32), device=device)
        lib_b = torch.full((end - start, 1), lib_val, device=device)
        bat_b = torch.full((end - start, 1), batch_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            out = module.generative(z=z_b, library=lib_b, batch_index=bat_b)
        # NB mean = px.mean  (already scaled by library in DecoderSCVI)
        mu = out['px'].mean.cpu().numpy()   # [B, n_genes]
        results.append(mu)

    return np.concatenate(results, axis=0)


def run_deg(expr_ctrl, expr_pert, gene_names, eps=1e-6):
    """
    Wilcoxon rank-sum test + log2FC.
    Returns DataFrame sorted by abs(log2FC), with BH-corrected pvals.
    """
    n_genes = expr_ctrl.shape[1]
    log2fc  = np.log2((expr_pert.mean(0) + eps) / (expr_ctrl.mean(0) + eps))

    pvals = np.zeros(n_genes)
    for i in range(n_genes):
        try:
            _, p = stats.mannwhitneyu(expr_ctrl[:, i], expr_pert[:, i],
                                       alternative='two-sided')
        except Exception:
            p = 1.0
        pvals[i] = p

    _, padj, _, _ = multipletests(pvals, method='fdr_bh')

    df = pd.DataFrame({
        'gene':     gene_names,
        'log2FC':   log2fc,
        'mean_ctrl': expr_ctrl.mean(0),
        'mean_pert': expr_pert.mean(0),
        'pval':     pvals,
        'padj':     padj,
    })
    df['abs_log2FC'] = df['log2FC'].abs()
    df = df.sort_values('abs_log2FC', ascending=False).reset_index(drop=True)
    return df


def plot_volcano(df, cond, out_path, top_n=20):
    fig, ax = plt.subplots(figsize=(8, 6))

    sig = (df['padj'] < 0.05) & (df['abs_log2FC'] > 0.5)
    up  = sig & (df['log2FC'] > 0)
    dn  = sig & (df['log2FC'] < 0)

    ax.scatter(df.loc[~sig, 'log2FC'], -np.log10(df.loc[~sig, 'pval'] + 1e-300),
               s=4, c='#cccccc', alpha=0.5, label='ns')
    ax.scatter(df.loc[dn, 'log2FC'],  -np.log10(df.loc[dn, 'pval'] + 1e-300),
               s=8, c='#4575b4', alpha=0.8, label=f'Down ({dn.sum()})')
    ax.scatter(df.loc[up, 'log2FC'],  -np.log10(df.loc[up, 'pval'] + 1e-300),
               s=8, c='#d73027', alpha=0.8, label=f'Up ({up.sum()})')

    # 상위 라벨
    top_df = df.head(top_n)
    for _, row in top_df.iterrows():
        ax.annotate(row['gene'],
                    xy=(row['log2FC'], -np.log10(row['pval'] + 1e-300)),
                    fontsize=6, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='gray'),
                    xytext=(row['log2FC'], -np.log10(row['pval'] + 1e-300) + 0.5))

    ax.axvline(0.5, color='gray', lw=0.8, ls='--')
    ax.axvline(-0.5, color='gray', lw=0.8, ls='--')
    ax.axhline(-np.log10(0.05), color='gray', lw=0.8, ls='--')
    ax.set_xlabel('log2 Fold Change (perturb / ctrl)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f'DEG: {cond}', fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap(expr_ctrl, expr_pert_dict, gene_names, top_genes, cond_labels, out_path):
    """top_genes 기준으로 조건 간 평균 expression heatmap"""
    gene_idx = [gene_names.tolist().index(g) for g in top_genes if g in gene_names]
    genes    = [gene_names[i] for i in gene_idx]

    rows  = [('ctrl', expr_ctrl.mean(0)[gene_idx])]
    for cond, expr_pert in expr_pert_dict.items():
        rows.append((cond, expr_pert.mean(0)[gene_idx]))

    mat  = np.array([r[1] for r in rows])
    lbls = [r[0] for r in rows]

    # log1p + z-score per gene
    mat_log = np.log1p(mat)
    mat_z   = (mat_log - mat_log.mean(0, keepdims=True)) / (mat_log.std(0, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(8, len(genes) * 0.5), len(rows) * 0.6 + 1))
    im = ax.imshow(mat_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(lbls)));  ax.set_yticklabels(lbls, fontsize=8)
    plt.colorbar(im, ax=ax, label='z-score (log1p)')
    ax.set_title('Top DEG Expression Heatmap (ctrl vs perturbations)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    EP_DIR = BASE / "results" / "trial6" / "gene_expression_recon" / args.t0_tag / "endpoints"
    OUT    = BASE / "results" / "trial6" / "gene_expression_recon" / args.t0_tag
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Step 2: Decode → DEG  [{args.t0_tag}]")
    print(f"  lib_size = {args.lib_size}")
    print("=" * 60)

    # ── 1. 배치 인덱스 결정 (최빈값) ─────────────────────────────
    obs_csv = EP_DIR / "start_cells_obs.csv"
    if obs_csv.exists():
        obs = pd.read_csv(obs_csv, index_col=0)
        if '_scvi_batch' in obs.columns:
            batch_idx = int(obs['_scvi_batch'].mode()[0])
        else:
            batch_idx = 0
        print(f"\n  batch_idx (mode): {batch_idx}")
    else:
        batch_idx = 0
        print(f"\n  start_cells_obs.csv not found, batch_idx = 0")

    # ── 2. scVI 모델 로드 ────────────────────────────────────────
    print("\n[1] Loading scVI model...")
    adata_ref = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    scvi_model = scvi.model.SCVI.load(SCVI_MODEL_DIR, adata=adata_ref)
    module = scvi_model.module.cuda()
    gene_names = np.array(adata_ref.var_names)
    print(f"  n_genes = {len(gene_names)}")

    # ── 3. ctrl endpoint 로드 & decode ───────────────────────────
    print("\n[2] Decoding ctrl endpoint...")
    z_ctrl = np.load(EP_DIR / "z_ctrl_endpoint.npy")
    print(f"  z_ctrl endpoint shape: {z_ctrl.shape}")
    expr_ctrl = decode_latents(module, z_ctrl, batch_idx, args.lib_size)
    print(f"  expr_ctrl shape: {expr_ctrl.shape}  "
          f"mean={expr_ctrl.mean():.4f}")
    np.save(OUT / "expr_ctrl_endpoint.npy", expr_ctrl)

    # ── 4. 각 condition decode & DEG ─────────────────────────────
    print("\n[3] Decoding perturbation conditions & DEG analysis...")
    conditions = {}
    for gene in PERTURB_GENES:
        for ctype in ['KO', 'OE3x']:
            cond = f"{gene}_{ctype}"
            ep_path = EP_DIR / f"z_{cond}_endpoint.npy"
            if not ep_path.exists():
                print(f"  [SKIP] {cond}: endpoint not found")
                continue

            print(f"\n  [{cond}]")
            z_pert   = np.load(ep_path)
            expr_pert = decode_latents(module, z_pert, batch_idx, args.lib_size)
            np.save(OUT / f"expr_{cond}_endpoint.npy", expr_pert)
            conditions[cond] = expr_pert

            # DEG
            df_deg = run_deg(expr_ctrl, expr_pert, gene_names)
            df_deg.to_csv(OUT / f"deg_{cond}.csv", index=False)

            sig_up = ((df_deg['padj'] < 0.05) & (df_deg['log2FC'] > 0.5)).sum()
            sig_dn = ((df_deg['padj'] < 0.05) & (df_deg['log2FC'] < -0.5)).sum()
            print(f"  Sig up={sig_up}  down={sig_dn}")
            print(f"  Top5:")
            for _, r in df_deg.head(5).iterrows():
                print(f"    {r['gene']:12s}  log2FC={r['log2FC']:+.3f}  padj={r['padj']:.2e}")

            # Volcano
            plot_volcano(df_deg, cond,
                         OUT / f"fig_volcano_{cond}.png",
                         top_n=args.n_top_deg)

    # ── 5. Top DEG summary ───────────────────────────────────────
    print("\n[4] Building top DEG summary...")
    rows = []
    for cond in conditions.keys():
        df = pd.read_csv(OUT / f"deg_{cond}.csv")
        top = df.head(args.n_top_deg).copy()
        top.insert(0, 'condition', cond)
        rows.append(top)
    df_summary = pd.concat(rows, ignore_index=True)
    df_summary.to_csv(OUT / "top_deg_summary.csv", index=False)
    print(f"  Saved top_deg_summary.csv ({len(df_summary)} rows)")

    # ── 6. IL17RD heatmap ────────────────────────────────────────
    print("\n[5] Drawing heatmap...")
    # IL17RD_KO + IL17RD_OE3x top genes 합집합
    top_genes = []
    for cond in ['IL17RD_KO', 'IL17RD_OE3x']:
        if cond in conditions:
            df = pd.read_csv(OUT / f"deg_{cond}.csv")
            top_genes.extend(df.head(20)['gene'].tolist())
    top_genes = list(dict.fromkeys(top_genes))  # 순서 유지 dedup

    focus_conds = {c: v for c, v in conditions.items()
                   if 'IL17RD' in c or c in ['PAX6_KO', 'NEUROG2_KO']}

    if top_genes and focus_conds:
        plot_heatmap(expr_ctrl, focus_conds, gene_names, top_genes,
                     list(focus_conds.keys()),
                     OUT / "fig_heatmap_IL17RD_top_deg.png")
        print(f"  Heatmap saved ({len(top_genes)} genes)")

    # ── 7. IL17RD_KO summary 출력 ────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — IL17RD_KO DEG")
    for cond in ['IL17RD_KO', 'IL17RD_OE3x']:
        if cond not in conditions:
            continue
        df = pd.read_csv(OUT / f"deg_{cond}.csv")
        print(f"\n  [{cond}]")
        print(f"  {'Gene':15s}  {'log2FC':>8}  {'padj':>10}  {'mean_ctrl':>10}  {'mean_pert':>10}")
        for _, r in df.head(20).iterrows():
            flag = "*" if r['padj'] < 0.05 else ""
            print(f"  {r['gene']:15s}  {r['log2FC']:+8.3f}  {r['padj']:10.2e}  "
                  f"{r['mean_ctrl']:10.4f}  {r['mean_pert']:10.4f}  {flag}")

    print(f"\n✓ All results saved to: {OUT}")


if __name__ == "__main__":
    main()
