#!/usr/bin/env python
"""
trial5_MrVI.py  (scArches_env)  —  GPU 0

HVG: scanpy_batch 고정 (hvg_A.json, scVI 스크립트가 먼저 생성)
Dim: 10 / 30 / 100 비교

평가:
  - active dims
  - scib: sil_bio, iLISI, sil_batch
  - IL17RD KO / OE latent shift  (cell-level z, give_z=True)
"""

import os, warnings, json, time
import numpy as np
import pandas as pd
import scanpy as sc
from scvi.external import MRVI
import scib
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE       = Path('/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5')
DIMS       = [10, 30, 100]
BATCH      = 'batch'
SAMPLE_KEY = 'donorID'
MAX_EPOCHS = 400

print("=" * 60)
print("Trial 5 — MrVI  |  GPU 0  |  dim: 10 / 30 / 100")
print(f"  sample_key='{SAMPLE_KEY}'  batch_key='{BATCH}'")
print("=" * 60)

# ── hvg_A.json 대기 (scVI 스크립트가 생성할 때까지) ─────────────
hvg_path = BASE / 'hvg_A.json'
print("\n[1] Waiting for hvg_A.json from scVI script...")
while not hvg_path.exists():
    time.sleep(30)
    print("  ... waiting")

with open(hvg_path) as f:
    hvg_genes = json.load(f)
print(f"  Loaded {len(hvg_genes):,} HVGs")

# ── 데이터 로드 ────────────────────────────────────────────────
print("\n[2] Loading data...")
adata_full = sc.read_h5ad(BASE / 'trial5_train_cap10k.h5ad')
adata_full.X = adata_full.layers['counts'].copy()

adata_hvg = adata_full[:, hvg_genes].copy()
adata_hvg.var['highly_variable'] = True
il17rd_idx = list(adata_hvg.var_names).index('IL17RD')
print(f"  {adata_hvg.n_obs:,} x {adata_hvg.n_vars:,}")
print(f"  Donors: {adata_hvg.obs[SAMPLE_KEY].nunique()}")

# ── 평가 함수 ─────────────────────────────────────────────────
def eval_model(model, model_dir, adata_dim, Z, dim):
    active = int((Z.std(axis=0) > 0.1).sum())

    adata_eval = adata_dim.copy()
    adata_eval.obsm['X_emb'] = Z
    sc.pp.neighbors(adata_eval, use_rep='X_emb', n_neighbors=15)

    try:    sil_bio   = scib.me.silhouette(adata_eval, label_key='cluster_annotated', embed='X_emb')
    except: sil_bio   = float('nan')
    try:    ilisi     = scib.me.ilisi_graph(adata_eval, batch_key=BATCH, type_='embed', use_rep='X_emb')
    except: ilisi     = float('nan')
    try:    sil_batch = scib.me.silhouette_batch(adata_eval, batch_key=BATCH, label_key='cluster_annotated', embed='X_emb')
    except: sil_batch = float('nan')

    print(f"  active={active}/{dim}  sil_bio={sil_bio:.4f}  ilisi={ilisi:.4f}  sil_batch={sil_batch:.4f}")

    shifts = {}
    for label, factor in [('KO', 0.0), ('OE', 3.0)]:
        ap = adata_dim.copy()
        X  = ap.X.toarray() if sp.issparse(ap.X) else ap.X.copy()
        X[:, il17rd_idx] = X[:, il17rd_idx] * factor
        ap.X = X.astype(np.float32)
        z_p  = MRVI.load(str(model_dir), adata=ap).get_latent_representation(ap, give_z=True)
        s    = np.linalg.norm(z_p - Z, axis=1)
        print(f"  IL17RD {label}: mean={s.mean():.6f}  max={s.max():.4f}")
        shifts[label] = (float(s.mean()), float(s.max()))

    return {'active_dims': active,
            'sil_bio': round(sil_bio,4), 'ilisi': round(ilisi,4),
            'sil_batch': round(sil_batch,4),
            'IL17RD_KO_mean': round(shifts['KO'][0],6),
            'IL17RD_KO_max':  round(shifts['KO'][1],4),
            'IL17RD_OE_mean': round(shifts['OE'][0],6)}

# ── dim별 학습 + 평가 ─────────────────────────────────────────
rows = []
for dim in DIMS:
    print(f"\n{'─'*60}")
    print(f"  MrVI  dim={dim}  (max_epochs={MAX_EPOCHS})")

    adata_dim = adata_hvg.copy()
    MRVI.setup_anndata(adata_dim, sample_key=SAMPLE_KEY, batch_key=BATCH)
    model = MRVI(adata_dim, n_latent=dim)
    model.train(max_epochs=MAX_EPOCHS, early_stopping=True,
                early_stopping_patience=20)

    model_dir = BASE / f'mrvi_dim{dim}'
    model.save(str(model_dir), overwrite=True)

    Z = model.get_latent_representation(adata_dim, give_z=True)
    out = adata_hvg.copy(); out.obsm['X_MrVI_z'] = Z

    # UMAP 계산 + 저장
    print("  Computing UMAP...")
    sc.pp.neighbors(out, use_rep='X_MrVI_z', n_neighbors=15)
    sc.tl.umap(out, min_dist=0.3)
    out.write_h5ad(BASE / f'trial5_latent_mrvi_dim{dim}.h5ad')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'MrVI dim={dim} UMAP  (cell-level z)', fontsize=12, fontweight='bold')
    sc.pl.umap(out, color='cluster_annotated', ax=axes[0], show=False,
               title='Cell type', legend_loc='right margin', legend_fontsize=6)
    sc.pl.umap(out, color='age_time_norm', ax=axes[1], show=False,
               title='age_time_norm', cmap='viridis')
    sc.pl.umap(out, color=BATCH, ax=axes[2], show=False,
               title='batch', legend_loc='right margin', legend_fontsize=6)
    plt.tight_layout()
    fig.savefig(BASE / f'trial5_umap_mrvi_dim{dim}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  UMAP saved: trial5_umap_mrvi_dim{dim}.png")

    metrics = eval_model(model, model_dir, adata_dim, Z, dim)
    rows.append({'model':'MrVI', 'dim':dim, **metrics})

# ── Summary + 통합 비교 ───────────────────────────────────────
df_mrvi = pd.DataFrame(rows)
print("\n\n" + "="*60 + "\nMrVI SUMMARY\n" + "="*60)
print(df_mrvi.to_string(index=False))
df_mrvi.to_csv(BASE / 'trial5_mrvi_results.csv', index=False)

scvi_csv = BASE / 'trial5_scvi_results.csv'
if scvi_csv.exists():
    df_all = pd.concat([pd.read_csv(scvi_csv), df_mrvi], ignore_index=True)
    df_all.to_csv(BASE / 'trial5_all_results.csv', index=False)
    print("\n[FULL COMPARISON]")
    print(df_all.to_string(index=False))

    # 통합 figure
    metrics_plot = [('sil_bio','Silhouette bio\n↑'), ('ilisi','iLISI\n↑'),
                    ('sil_batch','Sil batch\n↓'), ('IL17RD_KO_mean','IL17RD KO shift\n↑')]
    dims_str = [str(d) for d in DIMS]
    x = np.arange(len(DIMS))
    w = 0.35
    model_colors = {'scVI':'#2196F3', 'MrVI':'#E91E63'}

    fig, axes = plt.subplots(1, 4, figsize=(20,5))
    fig.suptitle('Trial 5 — scVI vs MrVI  (HVG: scanpy_batch, dim: 10/30/100)',
                 fontsize=12, fontweight='bold')
    for ci, (col, title) in enumerate(metrics_plot):
        ax = axes[ci]
        for mi, (mname, color) in enumerate(model_colors.items()):
            sub = df_all[df_all['model']==mname].sort_values('dim')
            if sub.empty: continue
            vals = sub[col].values
            bars = ax.bar(x + (mi-0.5)*w, vals, w,
                          label=mname, color=color, alpha=0.85, edgecolor='white')
            for b,v in zip(bars,vals):
                if not np.isnan(v):
                    ax.text(b.get_x()+b.get_width()/2,
                            b.get_height()+max(abs(vals))*0.02,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(dims_str)
        ax.set_xlabel('Latent dim'); ax.set_title(title, fontsize=9)
        if ci==0: ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(BASE / 'trial5_all_results.png', dpi=150, bbox_inches='tight')
    print("Saved: trial5_all_results.png")

print("\nDone! → trial5_all_results.csv / .png")
