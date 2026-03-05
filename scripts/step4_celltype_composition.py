#!/usr/bin/env python
"""
step4_celltype_composition.py  (scdiffeq_env or base)

Endpoint latents (ctrl vs each perturbation)를 atlas KNN으로 매핑하여
cell type composition 변화를 분석.

방법:
  1. Atlas latent + cell type labels 로드
  2. Endpoint latents → KNN (k=15) → 가장 가까운 atlas 세포들의 cell type 분포
  3. ctrl vs KO cell type 비율 비교
  4. Bootstrap CI 계산

Output: results/trial6/gene_expression_recon/celltype_comp/
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"
OUT    = TRIAL6 / "gene_expression_recon" / "celltype_comp"
OUT.mkdir(parents=True, exist_ok=True)

USE_KEY    = "X_scVI"
CT_KEY     = "CellType_refine"
K_NEIGHBORS = 15
N_BOOTSTRAP = 200
PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}

COLORS = {
    'RG': '#2196F3',
    'Neuroblast': '#FF9800',
    'Ext': '#4CAF50',
    'Fetal_Ext': '#8BC34A',
    'Fetal_Inh': '#F44336',
    'Inh': '#9C27B0',
}


def knn_celltype(z_query, z_ref, labels_ref, k=15):
    """z_query [N, D] → per-cell cell type (majority vote among k neighbors)"""
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(z_ref, labels_ref)
    return knn.predict(z_query)


def knn_composition(z_query, z_ref, labels_ref, k=15):
    """z_query → proportion of each cell type (soft, using k neighbors)"""
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(z_ref, labels_ref)
    proba = knn.predict_proba(z_query)   # [N, n_classes]
    classes = knn.classes_
    # Mean composition over all query cells
    return dict(zip(classes, proba.mean(0)))


def bootstrap_composition(z_query, z_ref, labels_ref, k=15, n_boot=200, seed=42):
    """Bootstrap CI on cell type composition"""
    rng = np.random.default_rng(seed)
    N = len(z_query)
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(z_ref, labels_ref)
    classes = knn.classes_

    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        proba = knn.predict_proba(z_query[idx])
        boot_means.append(proba.mean(0))
    boot_means = np.array(boot_means)  # [n_boot, n_classes]
    mean_comp = boot_means.mean(0)
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)
    return {c: (mean_comp[i], ci_lo[i], ci_hi[i]) for i, c in enumerate(classes)}


def plot_composition_bar(comp_ctrl, comp_pert_dict, t0_tag, out_path):
    """ctrl vs perturbations cell type composition bar chart"""
    all_types = sorted(set(comp_ctrl.keys()))
    n_conds = 1 + len(comp_pert_dict)
    cond_names = ['ctrl'] + list(comp_pert_dict.keys())

    comps = []
    comps.append([comp_ctrl[ct][0] for ct in all_types])
    for pert_comp in comp_pert_dict.values():
        comps.append([pert_comp[ct][0] if ct in pert_comp else 0 for ct in all_types])
    comps = np.array(comps)   # [n_conds, n_types]

    fig, ax = plt.subplots(figsize=(max(10, n_conds * 0.8), 6))
    x = np.arange(n_conds)
    bottom = np.zeros(n_conds)
    for i, ct in enumerate(all_types):
        color = COLORS.get(ct, '#aaaaaa')
        ax.bar(x, comps[:, i], bottom=bottom, label=ct, color=color, alpha=0.85)
        bottom += comps[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cell type proportion (KNN k=15)')
    ax.set_ylim(0, 1)
    ax.set_title(f'Cell Type Composition at Endpoint [{t0_tag}]', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_delta_bar(comp_ctrl, comp_pert, cond, t0_tag, out_path):
    """ctrl vs single perturbation delta bar with CI"""
    types = sorted(comp_ctrl.keys())
    delta_mean = []
    ci_lo_list = []
    ci_hi_list = []

    for ct in types:
        c_mean, c_lo, c_hi = comp_ctrl[ct]
        p_mean, p_lo, p_hi = comp_pert.get(ct, (0, 0, 0))
        delta_mean.append(p_mean - c_mean)
        # Error propagation (approximate)
        ci_lo_list.append((p_mean - p_hi) - (c_mean - c_lo))  # lower delta CI
        ci_hi_list.append((p_hi - p_mean) + (c_mean - c_lo))  # upper delta CI

    delta = np.array(delta_mean)
    err_lo = np.abs(ci_lo_list)
    err_hi = np.abs(ci_hi_list)
    colors = ['#d73027' if d > 0 else '#4575b4' for d in delta]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(types, delta, xerr=[err_lo, err_hi],
            color=colors, alpha=0.8, capsize=4)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Δ proportion (KO - ctrl)')
    ax.set_title(f'{cond} vs ctrl  [{t0_tag}]', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("Step 4: Cell Type Composition Analysis")
    print(f"  KNN k={K_NEIGHBORS}  Bootstrap n={N_BOOTSTRAP}")
    print("=" * 60)

    # ── 1. Atlas 로드 ─────────────────────────────────────────────
    print("\n[1] Loading atlas...")
    adata = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    z_ref    = adata.obsm[USE_KEY]
    ct_ref   = adata.obs[CT_KEY].values
    print(f"  Atlas: {adata.n_obs} cells  {len(set(ct_ref))} cell types")
    print(f"  Cell types: {sorted(set(ct_ref))}")

    all_summary = []

    for t0_tag, t0 in T0_OPTIONS.items():
        EP_DIR = TRIAL6 / "gene_expression_recon" / t0_tag / "endpoints"
        if not EP_DIR.exists():
            print(f"\n  [SKIP] {t0_tag}: endpoints not found")
            continue

        print(f"\n{'='*50}")
        print(f"[{t0_tag}]  t={t0}")

        # ── 2. ctrl endpoint ──────────────────────────────────────
        z_ctrl = np.load(EP_DIR / "z_ctrl_endpoint.npy")
        print(f"  z_ctrl endpoint: {z_ctrl.shape}")

        print("  Computing ctrl composition (bootstrap)...")
        comp_ctrl = bootstrap_composition(z_ctrl, z_ref, ct_ref,
                                          k=K_NEIGHBORS, n_boot=N_BOOTSTRAP)
        print(f"  ctrl:")
        for ct, (m, lo, hi) in sorted(comp_ctrl.items()):
            print(f"    {ct:15s}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")

        # ── 3. 각 perturbation ────────────────────────────────────
        comp_pert_dict = {}
        for gene in PERTURB_GENES:
            for ctype in ['KO', 'OE3x']:
                cond = f"{gene}_{ctype}"
                ep_path = EP_DIR / f"z_{cond}_endpoint.npy"
                if not ep_path.exists():
                    continue

                z_pert = np.load(ep_path)
                print(f"\n  [{cond}] composition (bootstrap)...")
                comp_pert = bootstrap_composition(z_pert, z_ref, ct_ref,
                                                   k=K_NEIGHBORS, n_boot=N_BOOTSTRAP)
                comp_pert_dict[cond] = comp_pert

                print(f"  {cond}:")
                for ct in sorted(comp_ctrl.keys()):
                    m_c, lo_c, hi_c = comp_ctrl[ct]
                    m_p, lo_p, hi_p = comp_pert.get(ct, (0, 0, 0))
                    delta = m_p - m_c
                    sig = "**" if abs(delta) > max(hi_c - m_c, m_c - lo_c) * 2 else ""
                    print(f"    {ct:15s}: ctrl={m_c:.3f}  pert={m_p:.3f}  Δ={delta:+.3f}  {sig}")

                # Delta bar plot
                plot_delta_bar(comp_ctrl, comp_pert, cond, t0_tag,
                               OUT / f"fig_delta_{cond}_{t0_tag}.png")

                # Summary row
                for ct in sorted(comp_ctrl.keys()):
                    m_c, lo_c, hi_c = comp_ctrl[ct]
                    m_p, lo_p, hi_p = comp_pert.get(ct, (0, 0, 0))
                    all_summary.append({
                        't0_tag': t0_tag, 'condition': cond, 'cell_type': ct,
                        'ctrl_mean': m_c, 'ctrl_lo': lo_c, 'ctrl_hi': hi_c,
                        'pert_mean': m_p, 'pert_lo': lo_p, 'pert_hi': hi_p,
                        'delta': m_p - m_c,
                    })

        # ── 4. 전체 composition bar ────────────────────────────────
        # IL17RD만 포함
        focus = {k: v for k, v in comp_pert_dict.items() if 'IL17RD' in k}
        if focus:
            plot_composition_bar(comp_ctrl, focus, t0_tag,
                                 OUT / f"fig_composition_IL17RD_{t0_tag}.png")

        # 전체
        plot_composition_bar(comp_ctrl, comp_pert_dict, t0_tag,
                             OUT / f"fig_composition_all_{t0_tag}.png")

    # ── 5. Summary CSV & print ────────────────────────────────────
    df_summary = pd.DataFrame(all_summary)
    df_summary.to_csv(OUT / "celltype_composition_summary.csv", index=False)
    print(f"\n✓ Summary saved: {OUT}/celltype_composition_summary.csv")

    # IL17RD_KO 요약 출력
    print("\n" + "=" * 60)
    print("SUMMARY — IL17RD_KO Cell Type Composition")
    for t0_tag in T0_OPTIONS.keys():
        df_t = df_summary[(df_summary['t0_tag'] == t0_tag) &
                          (df_summary['condition'] == 'IL17RD_KO')]
        if len(df_t) == 0:
            continue
        print(f"\n  [{t0_tag}]")
        print(f"  {'Cell Type':15s}  {'ctrl':>7}  {'KO':>7}  {'Δ':>7}")
        for _, r in df_t.sort_values('delta').iterrows():
            print(f"  {r['cell_type']:15s}  {r['ctrl_mean']:7.3f}  "
                  f"{r['pert_mean']:7.3f}  {r['delta']:+7.3f}")

    print(f"\n✓ All figures saved to: {OUT}")


if __name__ == "__main__":
    main()
