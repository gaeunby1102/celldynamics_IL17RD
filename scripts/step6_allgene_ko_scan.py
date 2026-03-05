#!/usr/bin/env python
"""
step6_allgene_ko_scan.py  (scArches_env)

모든 HVG 유전자 KO → scVI 재인코딩 → t=0 latent shift L2 계산.
시뮬레이션 없이 latent shift landscape만 측정 (Option B).

방법:
  1. t70d_RG / t115d_RG start cells의 raw counts 로드
  2. 각 HVG 유전자마다 count → 0 (KO)
  3. scVI 재인코딩 → z_KO
  4. paired L2 = mean_i ||z_KO_i - z_ctrl_i||  저장

Output: results/trial6/allgene_scan/
    allgene_ko_scan_{t0_tag}.csv
    fig_allgene_ko_landscape_{t0_tag}.png
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"
OUT    = TRIAL6 / "allgene_scan"
OUT.mkdir(parents=True, exist_ok=True)

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}
TIME_KEY = "age_time_norm"
CT_KEY   = "CellType_refine"
LOG_INTERVAL = 200

# 기존 6개 유전자 (비교 기준)
REF_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]


def load_scvi_model(adata):
    import scvi
    model = scvi.model.SCVI.load(str(TRIAL5 / "scvi_dim30"), adata=adata)
    model.to_device("cuda:0")
    model.module.eval()
    return model


def get_batch_index(adata_start, model):
    """obs['batch'] → scVI batch index tensor"""
    cats = model.adata.obs['batch'].cat.categories
    batch_map = {b: i for i, b in enumerate(cats)}
    idx = [batch_map[b] for b in adata_start.obs['batch']]
    return torch.tensor(idx, dtype=torch.long).unsqueeze(1)


def encode_batch(module, x_tensor, batch_idx, device):
    """scVI module.inference 직접 호출 → z (mean)"""
    x = x_tensor.to(device)
    bi = batch_idx.to(device)
    with torch.no_grad():
        out = module.inference(x, bi, n_samples=1)
    return out['qz'].loc.cpu().numpy()   # posterior mean (deterministic)


def allgene_ko_scan(model, adata_start, z_ctrl, t0_tag):
    """모든 유전자 KO → latent shift L2"""
    device = next(model.module.parameters()).device
    gene_names = np.array(adata_start.var_names)
    G = len(gene_names)

    # Raw counts → dense float32 tensor (한 번만)
    counts = adata_start.layers['counts']
    if sp.issparse(counts):
        x_np = counts.toarray().astype(np.float32)
    else:
        x_np = counts.astype(np.float32)

    x_base = torch.tensor(x_np, dtype=torch.float32)
    batch_idx = get_batch_index(adata_start, model)

    # Gene metadata
    mean_expr   = x_np.mean(0)          # [G]
    frac_expr   = (x_np > 0).mean(0)    # [G]

    results = []
    print(f"\n  Scanning {G} genes for {t0_tag}...")

    for gi, gene in enumerate(gene_names):
        # KO: 해당 gene column을  0으로
        x_ko = x_base.clone()
        x_ko[:, gi] = 0.0

        z_ko = encode_batch(model.module, x_ko, batch_idx, device)
        diff = z_ko - z_ctrl
        l2   = np.linalg.norm(diff, axis=1).mean()
        mean_shift = np.linalg.norm(diff.mean(0))   # centroid shift

        results.append({
            'gene': gene,
            'ko_l2_mean': l2,
            'ko_centroid_shift': mean_shift,
            'mean_expr': mean_expr[gi],
            'frac_expr': frac_expr[gi],
        })

        if (gi + 1) % LOG_INTERVAL == 0:
            print(f"  [{gi+1:5d}/{G}] {gene:15s}  L2={l2:.5f}", flush=True)

    df = pd.DataFrame(results).sort_values('ko_l2_mean', ascending=False)
    df['rank'] = np.arange(1, len(df) + 1)

    # 기존 6개 유전자 순위 추가
    ref_info = df[df['gene'].isin(REF_GENES)][['gene','rank','ko_l2_mean']].copy()
    print(f"\n  Reference gene ranks in all-gene scan:")
    for _, r in ref_info.sort_values('rank').iterrows():
        print(f"    {r['gene']:12s}  rank={int(r['rank']):4d} / {G}  L2={r['ko_l2_mean']:.5f}")

    return df


def plot_landscape(df, t0_tag, out_path):
    """KO latent shift landscape scatter"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'All-gene KO Landscape  [{t0_tag}]  (N={len(df)} HVGs)',
                 fontweight='bold')

    # ── Panel 1: rank vs L2 (volcano-style) ─────────────────────
    ax = axes[0]
    ax.scatter(df['rank'], df['ko_l2_mean'],
               s=3, alpha=0.3, color='#555555', label='all genes')

    # 기존 6개 유전자 강조
    colors_ref = {
        'IL17RD': '#d73027', 'PAX6': '#fc8d59', 'NEUROG2': '#fee090',
        'ASCL1': '#91bfdb', 'DLX2': '#4575b4', 'HES1': '#313695',
    }
    for gene, col in colors_ref.items():
        row = df[df['gene'] == gene]
        if len(row):
            ax.scatter(row['rank'], row['ko_l2_mean'],
                       s=60, color=col, zorder=5, label=gene)
            ax.annotate(gene, (row['rank'].values[0], row['ko_l2_mean'].values[0]),
                        xytext=(5, 2), textcoords='offset points', fontsize=7)

    ax.set_xlabel('Rank (by KO latent L2)')
    ax.set_ylabel('KO paired L2 (t=0)')
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title('KO effect landscape')

    # ── Panel 2: mean_expr vs L2 scatter ────────────────────────
    ax2 = axes[1]
    sc = ax2.scatter(df['mean_expr'], df['ko_l2_mean'],
                     s=3, alpha=0.3, color='#555555')
    for gene, col in colors_ref.items():
        row = df[df['gene'] == gene]
        if len(row):
            ax2.scatter(row['mean_expr'], row['ko_l2_mean'],
                        s=60, color=col, zorder=5, label=gene)
            ax2.annotate(gene, (row['mean_expr'].values[0], row['ko_l2_mean'].values[0]),
                         xytext=(3, 2), textcoords='offset points', fontsize=7)

    ax2.set_xlabel('Mean expression (start cells)')
    ax2.set_ylabel('KO paired L2 (t=0)')
    ax2.legend(fontsize=7, markerscale=2)
    ax2.set_title('Expression level vs KO effect')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig saved: {out_path.name}")


def main():
    print("=" * 65)
    print("Step 6: All-gene KO Scan  (scVI latent shift, no simulation)")
    print("=" * 65)

    # ── 1. 데이터 및 모델 로드 ───────────────────────────────────
    print("\n[1] Loading atlas adata & scVI model...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.n_obs)]

    model = load_scvi_model(adata)
    print(f"  Model loaded — n_latent={model.module.n_latent}  GPU={next(model.module.parameters()).device}")
    print(f"  Atlas shape: {adata.shape}")

    all_dfs = {}

    for t0_tag, t0 in T0_OPTIONS.items():
        print(f"\n{'='*55}")
        print(f"[{t0_tag}]  t={t0}")

        # ── 2. Start cells 선택 ──────────────────────────────────
        LAT_DIR = TRIAL6 / f"perturb_latents_{t0_tag}"
        obs_csv = LAT_DIR / "start_cells_obs.csv"
        z_ctrl  = np.load(LAT_DIR / "z_ctrl.npy")

        if obs_csv.exists():
            obs_df = pd.read_csv(obs_csv, index_col=0)
            start_barcodes = obs_df.index.tolist()
            # adata의 original_barcode로 매칭
            mask = adata.obs['original_barcode'].isin(start_barcodes)
        else:
            # fallback: timepoint + RG 필터
            mask_t = (adata.obs[TIME_KEY] - t0).abs() < 1e-4
            mask_rg = adata.obs[CT_KEY].str.contains('RG|Radial', case=False, na=False)
            mask = mask_t & mask_rg

        adata_start = adata[mask].copy()
        print(f"  Start cells: {adata_start.n_obs}  (z_ctrl shape: {z_ctrl.shape})")

        assert adata_start.n_obs == len(z_ctrl), \
            f"Cell count mismatch: {adata_start.n_obs} vs {len(z_ctrl)}"

        # ── 3. 전체 유전자 KO 스캔 ──────────────────────────────
        df = allgene_ko_scan(model, adata_start, z_ctrl, t0_tag)
        df['t0_tag'] = t0_tag
        all_dfs[t0_tag] = df

        # ── 4. 저장 ─────────────────────────────────────────────
        csv_path = OUT / f"allgene_ko_scan_{t0_tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  ✓ Saved: {csv_path.name}")

        # ── 5. Top 20 출력 ──────────────────────────────────────
        print(f"\n  Top 20 KO genes by L2 [{t0_tag}]:")
        print(f"  {'Gene':15s}  {'L2':>8}  {'Centroid':>10}  {'MeanExpr':>10}  {'FracExpr':>9}")
        for _, row in df.head(20).iterrows():
            flag = " ←" if row['gene'] in REF_GENES else ""
            print(f"  {row['gene']:15s}  {row['ko_l2_mean']:8.5f}  "
                  f"{row['ko_centroid_shift']:10.5f}  "
                  f"{row['mean_expr']:10.4f}  "
                  f"{row['frac_expr']:9.3f}{flag}")

        # ── 6. Figure ───────────────────────────────────────────
        plot_landscape(df, t0_tag, OUT / f"fig_allgene_ko_landscape_{t0_tag}.png")

    # ── 7. 두 타임포인트 비교 ────────────────────────────────────
    df70  = all_dfs.get('t70d_RG')
    df115 = all_dfs.get('t115d_RG')
    if df70 is not None and df115 is not None:
        from scipy.stats import spearmanr
        merged = df70[['gene','ko_l2_mean']].merge(
            df115[['gene','ko_l2_mean']], on='gene', suffixes=('_70d','_115d'))
        r, p = spearmanr(merged['ko_l2_mean_70d'], merged['ko_l2_mean_115d'])
        print(f"\n  Cross-timepoint Spearman r = {r:.4f}  p = {p:.2e}")
        merged.to_csv(OUT / "allgene_ko_scan_merged.csv", index=False)

        # Scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(merged['ko_l2_mean_70d'], merged['ko_l2_mean_115d'],
                   s=3, alpha=0.3, color='#555555')
        colors_ref = {
            'IL17RD': '#d73027', 'PAX6': '#fc8d59', 'NEUROG2': '#fee090',
            'ASCL1': '#91bfdb', 'DLX2': '#4575b4', 'HES1': '#313695',
        }
        for gene, col in colors_ref.items():
            row = merged[merged['gene'] == gene]
            if len(row):
                ax.scatter(row['ko_l2_mean_70d'], row['ko_l2_mean_115d'],
                           s=60, color=col, zorder=5, label=gene)
                ax.annotate(gene, (row['ko_l2_mean_70d'].values[0],
                                   row['ko_l2_mean_115d'].values[0]),
                            xytext=(3, 2), textcoords='offset points', fontsize=8)
        ax.set_xlabel('KO L2 (t70d_RG)')
        ax.set_ylabel('KO L2 (t115d_RG)')
        ax.set_title(f'All-gene KO scan: t70d vs t115d\n(Spearman r={r:.3f})',
                     fontweight='bold')
        ax.legend(fontsize=8, markerscale=2)
        plt.tight_layout()
        fig.savefig(OUT / "fig_allgene_ko_cross_timepoint.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Fig saved: fig_allgene_ko_cross_timepoint.png")

    print(f"\n✓ All results saved to: {OUT}")
    print("\n⟶ Next: run step7_topgene_simulation.py (scdiffeq_env) for trajectory simulation")


if __name__ == "__main__":
    main()
