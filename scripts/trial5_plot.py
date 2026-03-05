#!/usr/bin/env python
"""
trial5_plot.py — 통합 비교 figure + UMAP 6개 생성
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path('/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5')
DIMS = [10, 30, 100]

# ── 1. 통합 비교 figure ───────────────────────────────────────
print("[1] Drawing comparison figure...")

df_scvi = pd.read_csv(BASE / 'trial5_scvi_results.csv')
df_mrvi = pd.read_csv(BASE / 'trial5_mrvi_results.csv')
df_all  = pd.concat([df_scvi, df_mrvi], ignore_index=True)
df_all.to_csv(BASE / 'trial5_all_results.csv', index=False)

metrics = [
    ('sil_bio',          'Silhouette (bio)\n↑ 높을수록 세포유형 잘 분리'),
    ('ilisi',            'iLISI (batch mixing)\n↑ 높을수록 배치 잘 섞임'),
    ('sil_batch',        'Silhouette batch\n↓ 낮을수록 배치 제거 잘 됨'),
    ('IL17RD_KO_mean',   'IL17RD KO latent shift\n↑ 높을수록 민감'),
    ('active_dims',      'Active dims\n(posterior collapse 확인)'),
]

model_colors = {'scVI': '#2196F3', 'MrVI': '#E91E63'}
x = np.arange(len(DIMS))
w = 0.35

fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.suptitle('Trial 5 — scVI vs MrVI  |  HVG: scanpy_batch  |  dim: 10 / 30 / 100',
             fontsize=13, fontweight='bold')

for ci, (col, title) in enumerate(metrics):
    ax = axes[ci]
    for mi, (mname, color) in enumerate(model_colors.items()):
        sub = df_all[df_all['model'] == mname].sort_values('dim')
        vals = sub[col].values.astype(float)
        bars = ax.bar(x + (mi - 0.5) * w, vals, w,
                      label=mname, color=color, alpha=0.85, edgecolor='white')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + max(abs(vals)) * 0.02,
                    f'{v:.3f}' if col != 'active_dims' else str(int(v)),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlabel('Latent dim', fontsize=9)
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, max(df_all[col].astype(float)) * 1.25)
    if ci == 0:
        ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(BASE / 'trial5_all_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: trial5_all_results.png")

# ── 2. UMAP 6개 ───────────────────────────────────────────────
configs = [
    ('scVI',  'X_scVI',    'scvi'),
    ('MrVI',  'X_MrVI_z',  'mrvi'),
]

for model_name, obsm_key, prefix in configs:
    for dim in DIMS:
        tag  = f'{prefix}_dim{dim}'
        path = BASE / f'trial5_latent_{tag}.h5ad'
        if not path.exists():
            print(f"  SKIP (not found): {path.name}")
            continue

        print(f"\n[2] UMAP: {model_name} dim={dim}...")
        adata = sc.read_h5ad(path)

        # 항상 해당 latent key로 UMAP 새로 계산 (기존 X_umap 무시)
        print(f"  Computing neighbors + UMAP from {obsm_key}...")
        sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=15)
        sc.tl.umap(adata, min_dist=0.3)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{model_name}  dim={dim}  UMAP',
                     fontsize=13, fontweight='bold')

        sc.pl.umap(adata, color='cluster_annotated',
                   ax=axes[0], show=False, title='Cell type',
                   legend_loc='right margin', legend_fontsize=6, size=3)

        sc.pl.umap(adata, color='age_time_norm',
                   ax=axes[1], show=False, title='age_time_norm',
                   cmap='viridis', size=3)

        sc.pl.umap(adata, color='batch',
                   ax=axes[2], show=False, title='Batch',
                   legend_loc='right margin', legend_fontsize=5, size=3)

        plt.tight_layout()
        out_png = BASE / f'trial5_umap_{tag}.png'
        fig.savefig(out_png, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_png.name}")

        # 새로 계산한 UMAP을 h5ad에 덮어쓰기
        adata.write_h5ad(path)

print("\nAll plots done!")
print(f"Results: {BASE}")
