#!/usr/bin/env python
"""
enforce=0 / 0.5 / 1.0 drift 결과 비교 요약 생성.
각 drift_diffusion_results/{tag}/drift_diffusion_summary.csv 읽어서 비교.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

TRIAL5  = Path("/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5")
TAGS    = ["enforce0", "enforce05", "enforce1"]
LABELS  = {"enforce0": "enforce=0.0", "enforce05": "enforce=0.5", "enforce1": "enforce=1.0"}
COLORS  = {"enforce0": "#95a5a6",     "enforce05": "#3498db",      "enforce1": "#e74c3c"}

rows = []
for tag in TAGS:
    csv = TRIAL5 / "drift_diffusion_results" / tag / "drift_diffusion_summary.csv"
    if not csv.exists():
        print(f"  [{tag}] NOT READY: {csv}")
        continue
    df = pd.read_csv(csv)
    df['tag'] = tag
    df['label'] = LABELS[tag]
    rows.append(df)
    print(f"  [{tag}] loaded: {len(df)} conditions")

if len(rows) < 2:
    print("Not enough results yet.")
    import sys; sys.exit(0)

all_df = pd.concat(rows, ignore_index=True)

# ── 비교 figure ──────────────────────────────────────────────────
metrics = [
    ('mean_cos_sim',     'Mean Cosine Similarity\n(drift 방향 변화, 낮을수록 큰 변화)'),
    ('mean_angle_deg',   'Mean Angle (°)\n(높을수록 drift 방향 더 많이 바뀜)'),
    ('mean_speed_ratio', 'Mean Speed Ratio\n(1.0=변화없음)'),
    ('mean_delta_diff',  'Mean Δ Diffusion\n(>0=불확실성 증가)'),
]

conditions = all_df['condition'].unique()
n_cond = len(conditions)
x = np.arange(n_cond)
w = 0.25

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Perturbation Effect: enforce=0 vs 0.5 vs 1.0', fontsize=14, fontweight='bold')

for ax, (col, title) in zip(axes.flatten(), metrics):
    for i, tag in enumerate([t for t in TAGS if t in all_df['tag'].values]):
        sub = all_df[all_df['tag'] == tag].set_index('condition')
        vals = [sub.loc[c, col] if c in sub.index else np.nan for c in conditions]
        ax.bar(x + (i-1)*w, vals, w, label=LABELS[tag], color=COLORS[tag], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    if col == 'mean_cos_sim':
        ax.set_ylim(0.99, 1.001)
    if col == 'mean_angle_deg':
        ax.axhline(0, color='k', lw=0.5, ls='--')

plt.tight_layout()
out = TRIAL5 / "drift_diffusion_results" / "enforce_comparison.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

# ── IL17RD 특별 비교 ─────────────────────────────────────────────
print("\n=== IL17RD Drift Comparison ===")
print(f"{'Tag':<14} {'KO cos_sim':>12} {'KO angle°':>10} {'OE cos_sim':>12} {'OE angle°':>10}")
print("-" * 62)
for tag in TAGS:
    sub = all_df[all_df['tag'] == tag].set_index('condition')
    for ctype in ['KO', 'OE']:
        key = f"IL17RD_{ctype}"
        if key in sub.index:
            row = sub.loc[key]
            if ctype == 'KO':
                print(f"{LABELS[tag]:<14}  {row['mean_cos_sim']:>12.6f}  {row['mean_angle_deg']:>10.4f}", end="")
            else:
                print(f"  {row['mean_cos_sim']:>12.6f}  {row['mean_angle_deg']:>10.4f}")

print("\n=== Full Summary ===")
pivot = all_df.pivot_table(index='condition', columns='tag', values='mean_angle_deg')
print(pivot.round(4).to_string())

# DONE 마커 파일 생성
done_tags = all_df['tag'].unique().tolist()
(TRIAL5 / "drift_diffusion_results" / f"DONE_{'_'.join(done_tags)}.txt").write_text(
    f"Comparison ready: {done_tags}\n" + all_df.to_string()
)
print(f"\nDONE marker written.")
