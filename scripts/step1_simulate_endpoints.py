#!/usr/bin/env python
"""
step1_simulate_endpoints.py  (scdiffeq_env)

Trial5 enforce1 모델로 perturbation 궤적 시뮬레이션 → endpoint latents 저장.
Step 2 (scArches_env, scVI decode) 에서 불러씀.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/step1_simulate_endpoints.py \
        --t0_tag t70d_RG [t115d_RG]

Output:
    results/trial6/gene_expression_recon/{t0_tag}/endpoints/
        z_ctrl_endpoint.npy
        z_{gene}_{cond}_endpoint.npy
        ...
        start_cells_obs.csv   (배치 정보용)
"""

import torch
_orig = torch.load
def _p(*a, **k): k['weights_only'] = False; return _orig(*a, **k)
torch.load = _p
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE   = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
TRIAL5 = BASE / "results" / "trial5"
TRIAL6 = BASE / "results" / "trial6"

CKPT5_ENFORCE1 = (TRIAL5 / "train" /
    "trial5_SDE_enforce1_20260225_192525" /
    "LightningSDE-FixedPotential-RegularizedVelocityRatio" /
    "version_0" / "checkpoints" / "last.ckpt")

T0_OPTIONS = {
    "t70d_RG":  0.2471,
    "t115d_RG": 0.5588,
}
PERTURB_GENES = ["IL17RD", "PAX6", "NEUROG2", "ASCL1", "DLX2", "HES1"]
N_SIM_STEPS   = 200   # 더 촘촘하게 (endpoint 품질 ↑)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t0_tag", type=str, required=True,
                   choices=list(T0_OPTIONS.keys()))
    return p.parse_args()


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


def simulate_endpoint(model, z_init, n_steps, t_start):
    """z_init → endpoint latent (t=1.0 위치)
    Note: potential SDE drift = -grad(psi) → autograd.grad 사용 → no_grad() 불가
    """
    device = next(model.DiffEq.parameters()).device
    # requires_grad=True 필수 (potential SDE drift 계산에 필요)
    X0     = torch.tensor(z_init.astype(np.float32), device=device, requires_grad=True)
    t_grid = torch.linspace(t_start, 1.0, n_steps + 1).to(device)
    traj = model.DiffEq.forward(X0, t_grid)  # [T+1, N, D]
    return traj[-1].detach().cpu().numpy()    # endpoint [N, D]


def main():
    args = parse_args()
    t0   = T0_OPTIONS[args.t0_tag]

    OUT_EP = BASE / "results" / "trial6" / "gene_expression_recon" / args.t0_tag / "endpoints"
    OUT_EP.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Step 1: Simulate endpoints — {args.t0_tag}  t0={t0}")
    print(f"  n_sim_steps = {N_SIM_STEPS}")
    print("=" * 60)

    # ── 1. 모델 로드 ──────────────────────────────────────────────
    print("\n[1] Loading model...")
    adata_ref = ad.read_h5ad(TRIAL5 / "trial5_latent_scvi_dim30.h5ad")
    adata_ref.obs['original_barcode'] = adata_ref.obs_names.copy()
    adata_ref.obs_names = [str(i) for i in range(adata_ref.n_obs)]

    model = load_model(CKPT5_ENFORCE1, adata_ref)
    print(f"  Checkpoint: {CKPT5_ENFORCE1.parent.parent.parent.parent.name}")
    print("  Model loaded.")

    # ── 2. Latents 로드 ───────────────────────────────────────────
    LAT_DIR = TRIAL6 / f"perturb_latents_{args.t0_tag}"
    print(f"\n[2] Loading perturbation latents from {LAT_DIR.name}...")

    z_ctrl = np.load(LAT_DIR / "z_ctrl.npy")
    print(f"  z_ctrl: {z_ctrl.shape}")

    conditions = {"ctrl": z_ctrl}
    for gene in PERTURB_GENES:
        for ctype, fname in [("KO", f"z_{gene}_KO.npy"),
                             ("OE3x", f"z_{gene}_OE3x.npy")]:
            path = LAT_DIR / fname
            if path.exists():
                conditions[f"{gene}_{ctype}"] = np.load(path)
    print(f"  Conditions: {list(conditions.keys())}")

    # start_cells_obs.csv (배치 정보 보존)
    obs_src = LAT_DIR / "start_cells_obs.csv"
    if obs_src.exists():
        import shutil
        shutil.copy(obs_src, OUT_EP / "start_cells_obs.csv")
        print(f"  Copied start_cells_obs.csv")

    # ── 3. 시뮬레이션 ─────────────────────────────────────────────
    print(f"\n[3] Simulating {N_SIM_STEPS} steps  t={t0:.4f} → 1.0 ...")
    for cond, z_start in conditions.items():
        print(f"  {cond}...", end="", flush=True)
        z_end = simulate_endpoint(model, z_start, N_SIM_STEPS, t0)
        np.save(OUT_EP / f"z_{cond}_endpoint.npy", z_end)
        print(f"  → {z_end.shape}  range=[{z_end.min():.3f}, {z_end.max():.3f}]")

    print(f"\n✓ Saved {len(conditions)} endpoint arrays to:")
    print(f"  {OUT_EP}")
    print("\nStep 1 complete. Run Step 2 with scArches_env.")


if __name__ == "__main__":
    main()
