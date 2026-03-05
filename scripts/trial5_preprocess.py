#!/usr/bin/env python
"""
trial5_preprocess.py  (scArches_env)

Trial 5 전처리:
  1. Fetal 시점만 필터 (post-natal 제외)
  2. age_time_norm 재정규화 (49일 → 168일 기준 [0, 1])
  3. Holdout: 98일 (new t ≈ 0.412)
  4. 학습 데이터: 시점당 최대 10,000 세포 cap
  5. 분포 보고 + 시각화
"""

import os, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

SRC  = Path('/data2/Atlas_Normal/IL17RD_scdiffeq/results/Preprocess_fetal_neuron_trainset_before_scVI.h5ad')
OUT  = Path('/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5')
OUT.mkdir(parents=True, exist_ok=True)

TMP  = Path('/tmp/train_before_scVI_fixed.h5ad')

# ── 1. uns/log1p null 패치 후 로드 ─────────────────────────────────────────
print("=" * 60)
print("[1] Loading raw counts data...")
print("=" * 60)

import h5py
shutil.copy2(SRC, TMP)
with h5py.File(TMP, 'a') as h:
    if 'uns/log1p' in h:
        del h['uns/log1p']
        print("  Patched: removed uns/log1p (null base issue)")

import scanpy as sc
adata = sc.read_h5ad(TMP)
print(f"  Full data: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# ── 2. Fetal 필터 ──────────────────────────────────────────────────────────
print("\n[2] Filtering to Fetal stages only...")
fetal_mask = adata.obs['Stage'].str.contains('Fetal', na=False)
adata = adata[fetal_mask].copy()
print(f"  After filter: {adata.n_obs:,} cells")

# ── 3. Age time 재정규화 ────────────────────────────────────────────────────
print("\n[3] Re-normalizing age_time_norm (49d → 168d = [0, 1])...")

AGE_MIN = 49.0   # 1_Embryonic
AGE_MAX = 168.0  # 7_Late_fetal

age_repr = adata.obs['age_days_repr'].astype(float)
adata.obs['age_time_norm'] = (age_repr - AGE_MIN) / (AGE_MAX - AGE_MIN)
adata.obs['age_time_norm'] = adata.obs['age_time_norm'].clip(0.0, 1.0)

# timepoint별 새 norm 값 매핑 출력
tp_map = (adata.obs[['age_days_repr','age_time_norm']]
          .drop_duplicates()
          .sort_values('age_days_repr'))
print("\n  age_days_repr → new age_time_norm:")
for _, row in tp_map.iterrows():
    print(f"    {row['age_days_repr']:6.1f}d → {row['age_time_norm']:.4f}")

# ── 4. Holdout 분리 (98일) ──────────────────────────────────────────────────
print("\n[4] Splitting holdout (98d, t≈0.412)...")

HOLDOUT_AGE = 98.0
holdout_mask = (adata.obs['age_days_repr'].astype(float).round(1) == HOLDOUT_AGE)
adata_holdout = adata[holdout_mask].copy()
adata_train   = adata[~holdout_mask].copy()

print(f"  Holdout: {adata_holdout.n_obs:,} cells  (t={adata_holdout.obs['age_time_norm'].iloc[0]:.4f})")
print(f"  Train  : {adata_train.n_obs:,} cells")

# ── 5. 시점당 10k cap ──────────────────────────────────────────────────────
print("\n[5] Applying 10,000 cells/timepoint cap on training data...")

CAP = 10_000
rng = np.random.default_rng(42)

keep_idx = []
for tp, grp in adata_train.obs.groupby('age_days_repr', observed=True):
    if len(grp) > CAP:
        sampled = rng.choice(grp.index, size=CAP, replace=False)
        keep_idx.extend(sampled.tolist())
        print(f"  t={tp:.1f}d: {len(grp):,} → {CAP:,} (capped)")
    else:
        keep_idx.extend(grp.index.tolist())
        print(f"  t={tp:.1f}d: {len(grp):,} (kept all)")

adata_train_cap = adata_train[keep_idx].copy()
adata_train_cap.obs_names_make_unique()
print(f"\n  Train (capped): {adata_train_cap.n_obs:,} cells")

# ── 6. 분포 보고 ────────────────────────────────────────────────────────────
print("\n[6] Distribution report:")
print("\n  Training set (cap 10k):")
train_dist = (adata_train_cap.obs
              .groupby('age_days_repr', observed=True)
              .agg(n_cells=('age_time_norm','count'),
                   t_norm=('age_time_norm','first'))
              .sort_values('age_days_repr')
              .reset_index())
print(train_dist.to_string(index=False))

print("\n  Holdout set:")
print(f"    age_days_repr=98.0d, t_norm={adata_holdout.obs['age_time_norm'].iloc[0]:.4f}, "
      f"n_cells={adata_holdout.n_obs:,}")

print("\n  Cell type distribution (train):")
ct_dist = adata_train_cap.obs['cluster_annotated'].value_counts()
for ct, n in ct_dist.items():
    print(f"    {ct:<40s} {n:>6,}")

# ── 7. 시각화 ──────────────────────────────────────────────────────────────
print("\n[7] Saving distribution plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Trial 5 — Fetal-only dataset (49~168d)', fontsize=13, fontweight='bold')

# (a) 시점별 세포 수
t_norms = train_dist['t_norm'].values
n_cells = train_dist['n_cells'].values
ages    = train_dist['age_days_repr'].values

ax = axes[0]
bars = ax.bar([f"{a:.0f}d\n(t={t:.3f})" for a, t in zip(ages, t_norms)],
              n_cells, color='steelblue', edgecolor='white')
ax.axhline(CAP, color='red', linestyle='--', linewidth=1, label='10k cap')
for bar, n in zip(bars, n_cells):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{n:,}', ha='center', va='bottom', fontsize=8)
ax.set_xlabel('Timepoint'); ax.set_ylabel('# cells')
ax.set_title('Cells per timepoint (train)')
ax.legend(); ax.set_ylim(0, CAP * 1.25)

# (b) 시점 간격 비교 (old vs new)
ax = axes[1]
old_t = [0.000, 0.033, 0.073, 0.122, 0.296, 0.871, 1.000]
new_t = sorted(adata_train_cap.obs['age_time_norm'].unique().tolist())
for i, (ts, label, color) in enumerate([(old_t, 'Old (with post-natal)', 'salmon'),
                                          (new_t, 'New Trial 5 (fetal only)', 'steelblue')]):
    ax.scatter(ts, [i]*len(ts), s=120, color=color, zorder=5, label=label)
    ax.plot(ts, [i]*len(ts), color=color, alpha=0.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(['Old', 'New Trial 5'])
ax.set_xlabel('age_time_norm'); ax.set_title('Timepoint spacing comparison')
ax.legend(loc='upper left', fontsize=8); ax.grid(axis='x', alpha=0.3)

# (c) 세포 유형 분포
ax = axes[2]
top10 = ct_dist.head(10)
colors = plt.cm.tab10(np.linspace(0, 1, len(top10)))
ax.barh(range(len(top10)), top10.values, color=colors)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels([ct.replace(' ','\n') for ct in top10.index], fontsize=7)
ax.set_xlabel('# cells'); ax.set_title('Top 10 cell types (train)')
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(OUT / 'trial5_distribution.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT}/trial5_distribution.png")

# ── 8. 저장 ────────────────────────────────────────────────────────────────
print("\n[8] Saving h5ad files...")

adata_train_cap.write_h5ad(OUT / 'trial5_train_cap10k.h5ad')
print(f"  train : {OUT}/trial5_train_cap10k.h5ad  ({adata_train_cap.n_obs:,} cells)")

adata_holdout.write_h5ad(OUT / 'trial5_holdout_98d.h5ad')
print(f"  holdout: {OUT}/trial5_holdout_98d.h5ad  ({adata_holdout.n_obs:,} cells)")

print("\n" + "=" * 60)
print("Done! Next step: HVG selection + scVI encoding (trial5_scVI.py)")
print("=" * 60)
