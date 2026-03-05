#!/usr/bin/env python
"""
step5_uncertainty_quantification.py  (scdiffeq_env)

SDE 반복 시뮬레이션 → perturbation L2/SWD의 95% CI 계산.

핵심 질문: IL17RD_KO의 trajectoy divergence가
SDE stochasticity (ctrl vs ctrl 노이즈) 대비 유의하게 큰가?

방법:
  1. ctrl 세포를 N_RUNS번 서로 다른 seed로 시뮬레이션 → ctrl-vs-ctrl L2 분포 (노이즈 기준선)
  2. ctrl vs each KO/OE를 N_RUNS번 시뮬레이션 → perturbation L2 분포
  3. KO L2 > ctrl noise 95th percentile → 유의하게 큰 perturbation

Output: results/trial6/gene_expression_recon/uncertainty/
"""

import torch
_orig = torch.load
def _p(*a, **k): k['weights_only'] = False; return _orig(*a, **k)
torch.load = _p
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"
OUT    = TRIAL6 / "gene_expression_recon" / "uncertainty"
OUT.mkdir(parents=True, exist_ok=True)

CKPT5 = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}
PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]
N_RUNS      = 10    # 반복 횟수 (SDE stochastic 노이즈 평가)
N_SIM_STEPS = 100   # 시뮬레이션 step 수


def load_model(ckpt_path, adata_ref):
    import scdiffeq as sdq
    from scdiffeq.io import load_diffeq
    diffeq  = load_diffeq(ckpt_path=str(ckpt_path))
    hparams = dict(diffeq.hparams)
    hparams['time_key'] = 'age_time_norm'
    hparams['use_key']  = 'X_scVI'
    model = sdq.scDiffEq(adata=adata_ref, **hparams)
    model.configure_data(adata=adata_ref)
    model.configure_model(diffeq, loading_existing=True)
    model.DiffEq.eval()
    return model


def simulate_endpoint(model, z_init, n_steps, t_start, seed=None):
    """z_init → endpoint latent (t=1.0)"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    device = next(model.DiffEq.parameters()).device
    X0     = torch.tensor(z_init.astype(np.float32), device=device, requires_grad=True)
    t_grid = torch.linspace(t_start, 1.0, n_steps + 1).to(device)
    traj   = model.DiffEq.forward(X0, t_grid)
    return traj[-1].detach().cpu().numpy()


def paired_l2(z1, z2):
    return np.linalg.norm(z1 - z2, axis=1).mean()


def sliced_wasserstein(z1, z2, n_proj=50, seed=0):
    rng  = np.random.default_rng(seed)
    D    = z1.shape[1]
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    swds = []
    for d in dirs:
        swds.append(wasserstein_distance(z1 @ d, z2 @ d))
    return np.mean(swds)


def main():
    print("=" * 65)
    print(f"Step 5: Uncertainty Quantification  (N_RUNS={N_RUNS})")
    print("=" * 65)

    # ── 1. 모델 로드 ─────────────────────────────────────────────
    print("\n[1] Loading model...")
    adata_ref = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata_ref.obs['original_barcode'] = adata_ref.obs_names.copy()
    adata_ref.obs_names = [str(i) for i in range(adata_ref.n_obs)]
    model = load_model(CKPT5, adata_ref)
    print("  Model loaded.")

    all_rows = []

    for t0_tag, t0 in T0_OPTIONS.items():
        LAT_DIR = TRIAL6 / f"perturb_latents_{t0_tag}"
        print(f"\n{'='*55}")
        print(f"[{t0_tag}]  t={t0}")

        z_ctrl = np.load(LAT_DIR / "z_ctrl.npy")
        print(f"  z_ctrl: {z_ctrl.shape}")

        # ── 2. Ctrl-vs-Ctrl 노이즈 기준선 ─────────────────────
        print(f"\n  [2] Ctrl noise baseline ({N_RUNS} runs)...")
        ctrl_endpoints = []
        for run in range(N_RUNS):
            seed = 1000 + run
            z_ep = simulate_endpoint(model, z_ctrl, N_SIM_STEPS, t0, seed=seed)
            ctrl_endpoints.append(z_ep)
            print(f"    run {run}  endpoint L2 from run0: "
                  f"{paired_l2(ctrl_endpoints[0], z_ep):.4f}", flush=True)

        # ctrl-vs-ctrl paired L2 분포 (run0 vs run1,2,...N-1)
        ctrl_noise_l2 = [paired_l2(ctrl_endpoints[0], ctrl_endpoints[r])
                         for r in range(1, N_RUNS)]
        ctrl_noise_swd = [sliced_wasserstein(ctrl_endpoints[0], ctrl_endpoints[r])
                          for r in range(1, N_RUNS)]

        print(f"\n  Ctrl noise L2:  mean={np.mean(ctrl_noise_l2):.4f}  "
              f"95th={np.percentile(ctrl_noise_l2, 95):.4f}  "
              f"max={np.max(ctrl_noise_l2):.4f}")
        print(f"  Ctrl noise SWD: mean={np.mean(ctrl_noise_swd):.4f}  "
              f"95th={np.percentile(ctrl_noise_swd, 95):.4f}")

        # ── 3. 각 perturbation CI ─────────────────────────────
        print(f"\n  [3] Perturbation runs...")
        for gene in PERTURB_GENES:
            for ctype in ['KO', 'OE3x']:
                cond    = f"{gene}_{ctype}"
                z_pert_path = LAT_DIR / f"z_{gene}_{ctype}.npy"
                if not z_pert_path.exists():
                    continue

                z_pert = np.load(z_pert_path)
                pert_l2_runs  = []
                pert_swd_runs = []

                for run in range(N_RUNS):
                    seed = 1000 + run
                    z_ctrl_ep = ctrl_endpoints[run]
                    z_pert_ep = simulate_endpoint(model, z_pert, N_SIM_STEPS, t0, seed=seed)
                    pert_l2_runs.append(paired_l2(z_ctrl_ep, z_pert_ep))
                    pert_swd_runs.append(sliced_wasserstein(z_ctrl_ep, z_pert_ep))

                l2_mean  = np.mean(pert_l2_runs)
                l2_lo    = np.percentile(pert_l2_runs, 2.5)
                l2_hi    = np.percentile(pert_l2_runs, 97.5)
                swd_mean = np.mean(pert_swd_runs)
                swd_lo   = np.percentile(pert_swd_runs, 2.5)
                swd_hi   = np.percentile(pert_swd_runs, 97.5)

                noise_95 = np.percentile(ctrl_noise_l2, 95)
                sig = l2_lo > noise_95   # 95% CI 하한이 noise 95th 초과

                print(f"    {cond:18s}  L2={l2_mean:.4f} [{l2_lo:.4f},{l2_hi:.4f}]  "
                      f"SWD={swd_mean:.4f}  {'*SIG*' if sig else ''}")

                all_rows.append({
                    't0_tag': t0_tag, 'condition': cond, 'gene': gene, 'perturb_type': ctype,
                    'l2_mean': l2_mean, 'l2_lo': l2_lo, 'l2_hi': l2_hi,
                    'swd_mean': swd_mean, 'swd_lo': swd_lo, 'swd_hi': swd_hi,
                    'ctrl_noise_l2_mean': np.mean(ctrl_noise_l2),
                    'ctrl_noise_l2_95th': noise_95,
                    'significant_l2': sig,
                    'n_runs': N_RUNS,
                })

    # ── 4. CSV 저장 ──────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "perturbation_uncertainty.csv", index=False)
    print(f"\n✓ Saved: {OUT}/perturbation_uncertainty.csv")

    # ── 5. Plot ──────────────────────────────────────────────────
    for t0_tag in T0_OPTIONS.keys():
        df_t = df[df['t0_tag'] == t0_tag].copy()
        if len(df_t) == 0:
            continue

        df_t = df_t.sort_values('l2_mean', ascending=False).reset_index(drop=True)
        ctrl_noise_val = df_t['ctrl_noise_l2_mean'].iloc[0]
        ctrl_95th      = df_t['ctrl_noise_l2_95th'].iloc[0]

        fig, ax = plt.subplots(figsize=(9, len(df_t) * 0.5 + 2))
        colors = ['#d73027' if 'IL17RD' in c else
                  '#f46d43' if 'PAX6' in c else
                  '#74add1' for c in df_t['condition']]
        y = np.arange(len(df_t))
        ax.barh(y, df_t['l2_mean'], xerr=[df_t['l2_mean'] - df_t['l2_lo'],
                                           df_t['l2_hi'] - df_t['l2_mean']],
                color=colors, alpha=0.8, capsize=3, height=0.6)
        ax.axvline(ctrl_noise_val, color='gray',  lw=1.5, ls='--',
                   label=f'ctrl noise mean ({ctrl_noise_val:.4f})')
        ax.axvline(ctrl_95th,      color='black', lw=1.5, ls=':',
                   label=f'ctrl noise 95th ({ctrl_95th:.4f})')
        ax.set_yticks(y)
        ax.set_yticklabels(df_t['condition'], fontsize=8)
        ax.set_xlabel('Paired L2  (ctrl vs perturbation endpoint)')
        ax.set_title(f'Perturbation Uncertainty [{t0_tag}]  (N={N_RUNS} runs)',
                     fontweight='bold')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(OUT / f"fig_uncertainty_{t0_tag}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Fig saved: fig_uncertainty_{t0_tag}.png")

    # ── 6. IL17RD 요약 ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY — IL17RD Perturbation Uncertainty")
    il17rd_df = df[df['gene'] == 'IL17RD']
    print(f"\n  {'Tag':12s}  {'Cond':18s}  {'L2 mean':>8}  {'95% CI':>18}  {'Noise 95th':>11}  {'Sig?':>5}")
    for _, r in il17rd_df.iterrows():
        print(f"  {r['t0_tag']:12s}  {r['condition']:18s}  "
              f"{r['l2_mean']:8.4f}  [{r['l2_lo']:.4f}, {r['l2_hi']:.4f}]  "
              f"{r['ctrl_noise_l2_95th']:11.4f}  {'YES' if r['significant_l2'] else 'no':>5}")


if __name__ == "__main__":
    main()
