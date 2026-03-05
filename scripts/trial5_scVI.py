#!/usr/bin/env python
"""
trial5_scVI.py  (scArches_env)  —  GPU 1

HVG: scanpy_batch 고정 (seurat_v3, batch_key='batch')
Dim: 10 / 30 / 100 비교

평가:
  - active dims (posterior collapse 체크)
  - scib: sil_bio, iLISI, sil_batch
  - IL17RD KO / OE latent shift
"""

import os, warnings, json
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import scib
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BASE       = Path('/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5')
DIMS       = [10, 30, 100]
N_HVG      = 5010
BATCH      = 'batch'
MAX_EPOCHS = 400

print("=" * 60)
print("Trial 5 — scVI  |  GPU 1  |  dim: 10 / 30 / 100")
print("=" * 60)

# ── 데이터 로드 ────────────────────────────────────────────────
print("\n[1] Loading data...")
adata_full = sc.read_h5ad(BASE / 'trial5_train_cap10k.h5ad')
adata_full.X = adata_full.layers['counts'].copy()
print(f"  {adata_full.n_obs:,} x {adata_full.n_vars:,}")

# ── HVG 선택 (scanpy_batch) ────────────────────────────────────
print(f"\n[2] HVG selection (seurat_v3, batch_key='{BATCH}')...")
adata_pp = adata_full.copy()
sc.pp.normalize_total(adata_pp, target_sum=1e4)
sc.pp.log1p(adata_pp)
sc.pp.highly_variable_genes(adata_pp, n_top_genes=N_HVG,
                             batch_key=BATCH, flavor='seurat_v3')
hvg_genes = adata_pp.var_names[adata_pp.var['highly_variable']].tolist()
if 'IL17RD' not in hvg_genes:
    hvg_genes.append('IL17RD')
print(f"  HVG: {len(hvg_genes):,}  |  IL17RD included: {'IL17RD' in hvg_genes}")

# HVG 목록 저장 (MrVI 스크립트에서 읽어 씀)
with open(BASE / 'hvg_A.json', 'w') as f:
    json.dump(hvg_genes, f)
print("  Saved: hvg_A.json")

adata_hvg = adata_full[:, hvg_genes].copy()
adata_hvg.var['highly_variable'] = True
il17rd_idx = list(adata_hvg.var_names).index('IL17RD')

# ── dim별 학습 + 평가 ─────────────────────────────────────────
def eval_model(vae, model_dir, adata_dim, Z, dim):
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
        z_p  = scvi.model.SCVI.load(str(model_dir), adata=ap).get_latent_representation()
        s    = np.linalg.norm(z_p - Z, axis=1)
        print(f"  IL17RD {label}: mean={s.mean():.6f}  max={s.max():.4f}")
        shifts[label] = (float(s.mean()), float(s.max()))

    return {'active_dims': active,
            'sil_bio': round(sil_bio,4), 'ilisi': round(ilisi,4),
            'sil_batch': round(sil_batch,4),
            'IL17RD_KO_mean': round(shifts['KO'][0],6),
            'IL17RD_KO_max':  round(shifts['KO'][1],4),
            'IL17RD_OE_mean': round(shifts['OE'][0],6)}

rows = []
for dim in DIMS:
    print(f"\n{'─'*60}")
    print(f"  scVI  dim={dim}  (max_epochs={MAX_EPOCHS})")

    adata_dim = adata_hvg.copy()
    scvi.model.SCVI.setup_anndata(adata_dim, batch_key=BATCH)
    vae = scvi.model.SCVI(adata_dim, n_latent=dim, n_layers=2,
                          n_hidden=128, gene_likelihood='nb')
    vae.train(max_epochs=MAX_EPOCHS, early_stopping=True,
              early_stopping_patience=20, plan_kwargs={'lr':1e-3})

    model_dir = BASE / f'scvi_dim{dim}'
    vae.save(str(model_dir), overwrite=True)

    Z = vae.get_latent_representation()
    out = adata_hvg.copy(); out.obsm['X_scVI'] = Z

    # UMAP 계산 + 저장
    print("  Computing UMAP...")
    sc.pp.neighbors(out, use_rep='X_scVI', n_neighbors=15)
    sc.tl.umap(out, min_dist=0.3)
    out.write_h5ad(BASE / f'trial5_latent_scvi_dim{dim}.h5ad')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'scVI dim={dim} UMAP', fontsize=12, fontweight='bold')
    sc.pl.umap(out, color='cluster_annotated', ax=axes[0], show=False,
               title='Cell type', legend_loc='right margin', legend_fontsize=6)
    sc.pl.umap(out, color='age_time_norm', ax=axes[1], show=False,
               title='age_time_norm', cmap='viridis')
    sc.pl.umap(out, color=BATCH, ax=axes[2], show=False,
               title='batch', legend_loc='right margin', legend_fontsize=6)
    plt.tight_layout()
    fig.savefig(BASE / f'trial5_umap_scvi_dim{dim}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  UMAP saved: trial5_umap_scvi_dim{dim}.png")

    metrics = eval_model(vae, model_dir, adata_dim, Z, dim)
    rows.append({'model':'scVI', 'dim':dim, **metrics})

# ── Summary ───────────────────────────────────────────────────
df = pd.DataFrame(rows)
print("\n\n" + "="*60 + "\nscVI SUMMARY\n" + "="*60)
print(df.to_string(index=False))
df.to_csv(BASE / 'trial5_scvi_results.csv', index=False)

# figure
fig, axes = plt.subplots(1, 4, figsize=(18,4))
fig.suptitle('Trial 5 — scVI dim comparison', fontweight='bold')
metrics_plot = [('sil_bio','Silhouette bio\n↑'), ('ilisi','iLISI\n↑'),
                ('sil_batch','Sil batch\n↓'), ('IL17RD_KO_mean','IL17RD KO shift\n↑')]
colors = ['#2196F3','#4CAF50','#FF5722']
for ax, (col, title) in zip(axes, metrics_plot):
    vals = df[col].values
    bars = ax.bar([str(d) for d in DIMS], vals, color=colors)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(abs(vals))*0.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(title, fontsize=9); ax.set_xlabel('dim')
plt.tight_layout()
fig.savefig(BASE / 'trial5_scvi_results.png', dpi=150, bbox_inches='tight')

print("\nDone! → trial5_scvi_results.csv / .png")
