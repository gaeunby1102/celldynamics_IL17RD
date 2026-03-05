#!/usr/bin/env python
"""
train_new_holdout.py  (scdiffeq_env)

새 holdout split (t=0.165 제외, cap 10k/tp) 데이터로 scDiffeq 학습.
GPU 1번 사용 (CUDA_VISIBLE_DEVICES=1).

Usage:
    python train_new_holdout.py
    python train_new_holdout.py --train_epochs 3000
"""

# ── PyTorch 2.6+ checkpoint 호환 패치 (scdiffeq 로드 전에 먼저) ────────────
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ── Safe globals for optimizer state ──────────────────────────────────────────
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from lightning.pytorch.callbacks import EarlyStopping, Callback
from scdiffeq.core.callbacks import BasicProgressBar

# =============================================================================
# Custom Logging Callback (동일)
# =============================================================================

class EpochMetricsLogger(Callback):
    def __init__(self, log_file: str, log_every_n_epochs: int = 10):
        super().__init__()
        self.log_file = log_file
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss = metrics.get('epoch_train_loss', None)

        epoch_data = {
            'epoch': epoch,
            'train_loss': float(train_loss) if train_loss is not None else None,
        }

        sinkhorn_sum = velo_sum = sinkhorn_n = velo_n = 0
        for key, value in metrics.items():
            if 'sinkhorn' in key and 'training' in key:
                try: sinkhorn_sum += float(value); sinkhorn_n += 1
                except: pass
            if 'velo_ratio' in key and 'training' in key:
                try: velo_sum += float(value); velo_n += 1
                except: pass

        epoch_data['mean_sinkhorn']   = sinkhorn_sum / sinkhorn_n if sinkhorn_n > 0 else None
        epoch_data['mean_velo_ratio'] = velo_sum / velo_n         if velo_n > 0     else None
        self.epoch_metrics.append(epoch_data)

        if epoch % self.log_every_n_epochs == 0 or epoch == 0:
            msg = f"[Epoch {epoch:4d}]"
            if epoch_data['train_loss'] is not None:
                msg += f" loss={epoch_data['train_loss']:.4f}"
            if epoch_data['mean_sinkhorn'] is not None:
                msg += f" sinkhorn={epoch_data['mean_sinkhorn']:.2f}"
            if epoch_data['mean_velo_ratio'] is not None:
                msg += f" velo_ratio={epoch_data['mean_velo_ratio']:.2f}"
            print(msg, flush=True)
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

    def on_train_end(self, trainer, pl_module):
        import csv
        csv_file = self.log_file.replace('.log', '_metrics.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','mean_sinkhorn','mean_velo_ratio'])
            writer.writeheader()
            writer.writerows(self.epoch_metrics)
        print(f"Metrics CSV saved: {csv_file}")

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR      = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
DATA_PATH     = BASE_DIR / "results" / "new_run" / "train_cap10k_holdout0165.h5ad"
EXPERIMENT_DIR= BASE_DIR / "results" / "new_run" / "train"

LATENT_DIM    = 30
TIME_KEY      = "age_time_norm"
USE_KEY       = "X_scVI"
MU_HIDDEN     = [512, 512]
SIGMA_HIDDEN  = [32, 32]
BATCH_SIZE    = 1024
LR            = 1e-4
DT            = 0.01
TRAIN_EPOCHS  = 2000
CHECKPOINT_EVERY = 25
LOG_EVERY     = 10
SEED          = 42

def parse_args():
    p = argparse.ArgumentParser(description="Train scDiffeq (holdout run)")
    p.add_argument("--train_epochs",   type=int, default=TRAIN_EPOCHS)
    p.add_argument("--batch_size",     type=int, default=BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--dt",             type=float, default=DT)
    p.add_argument("--checkpoint_every",type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--seed",           type=int, default=SEED)
    p.add_argument("--experiment_name",type=str, default=None)
    p.add_argument("--resume_from",    type=str, default=None)
    p.add_argument("--debug",          action="store_true")
    return p.parse_args()

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    if args.debug:
        args.train_epochs = 20

    set_seed(args.seed)

    print("=" * 70)
    print("scDiffeq Training  (holdout t=0.165, cap 10k/tp, GPU 1)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Import scDiffeq
    # ------------------------------------------------------------------
    print("\nImporting scdiffeq ...")
    import scdiffeq as sdq
    print(f"  version: {getattr(sdq, '__version__', 'unknown')}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading data: {DATA_PATH}")
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run preprocess_new_holdout.py first.")
        sys.exit(1)

    t0 = time.time()
    adata = ad.read_h5ad(DATA_PATH)
    print(f"    Loaded in {time.time()-t0:.1f}s  |  {adata.shape[0]:,} cells")

    # age_time_norm 이미 있음 — train.py와 동일하게 'time' 컬럼 추가
    if 'time' not in adata.obs.columns:
        # age_time_norm을 그대로 사용 (이미 [0,1] 범위)
        adata.obs['time'] = adata.obs[TIME_KEY].values
    print(f"    Time range: {adata.obs[TIME_KEY].min():.4f} – {adata.obs[TIME_KEY].max():.4f}")
    print(f"    Timepoints: {sorted(adata.obs[TIME_KEY].unique())}")
    print(f"    X_scVI    : {adata.obsm['X_scVI'].shape}")

    # obs_names가 이미 리셋돼 있지만 혹시 모르니 체크
    if not adata.obs_names[0].isdigit():
        print("    Resetting obs_names ...")
        adata.obs['original_barcode'] = adata.obs_names.copy()
        adata.obs_names = [str(i) for i in range(adata.shape[0])]

    # ------------------------------------------------------------------
    # Setup experiment dir
    # ------------------------------------------------------------------
    exp_name = args.experiment_name or f"holdout0165_SDE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir  = EXPERIMENT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] Experiment dir: {exp_dir}")

    # Log file
    log_file = str(exp_dir / "training.log")
    with open(log_file, 'w') as f:
        f.write(f"scDiffeq training log\n")
        f.write(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data : {DATA_PATH}\n")
        f.write(f"Cells: {adata.shape[0]:,}\n")
        f.write(f"Epochs: {args.train_epochs}\n\n")

    # Save config
    with open(exp_dir / "config.txt", 'w') as f:
        f.write(f"data_path      : {DATA_PATH}\n")
        f.write(f"holdout_tp     : 0.165423\n")
        f.write(f"cap_per_tp     : 10000\n")
        f.write(f"use_key        : {USE_KEY}\n")
        f.write(f"latent_dim     : {LATENT_DIM}\n")
        f.write(f"diffeq_type    : SDE\n")
        f.write(f"train_epochs   : {args.train_epochs}\n")
        f.write(f"batch_size     : {args.batch_size}\n")
        f.write(f"learning_rate  : {args.lr}\n")
        f.write(f"dt             : {args.dt}\n")
        f.write(f"mu_hidden      : {MU_HIDDEN}\n")
        f.write(f"sigma_hidden   : {SIGMA_HIDDEN}\n")
        f.write(f"GPU            : CUDA_VISIBLE_DEVICES=1\n")
        f.write(f"seed           : {args.seed}\n")

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    print(f"\n[3] Initializing scDiffeq model")
    print(f"    latent_dim={LATENT_DIM}, use_key={USE_KEY}, dt={args.dt}")
    print(f"    mu_hidden={MU_HIDDEN}, sigma_hidden={SIGMA_HIDDEN}")

    model = sdq.scDiffEq(
        adata=adata,
        latent_dim=LATENT_DIM,
        use_key=USE_KEY,
        time_key=TIME_KEY,
        DiffEq_type="SDE",
        mu_hidden=MU_HIDDEN,
        sigma_hidden=SIGMA_HIDDEN,
        batch_size=args.batch_size,
        dt=args.dt,
        working_dir=str(exp_dir),
        velocity_ratio_params={"target": 2.5, "enforce": 0.0, "method": "square"},
    )
    print("    Model initialized.")

    # ------------------------------------------------------------------
    # Configure data + model
    # ------------------------------------------------------------------
    print(f"\n[4] Configuring data and model")
    model.configure_data(adata=adata)
    model.configure_model()
    print("    Done.")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\n[5] Training  ({args.train_epochs} epochs)")
    print(f"    lr={args.lr}, batch_size={args.batch_size}, checkpoint_every={args.checkpoint_every}")

    callbacks = []
    callbacks.append(BasicProgressBar(
        total_epochs=args.train_epochs,
        print_every=LOG_EVERY
    ))
    callbacks.append(EpochMetricsLogger(log_file=log_file, log_every_n_epochs=LOG_EVERY))

    resume_ckpt = args.resume_from
    if resume_ckpt and not os.path.exists(resume_ckpt):
        print(f"WARNING: checkpoint not found ({resume_ckpt}), starting fresh.")
        resume_ckpt = None

    train_start = time.time()
    model.train(
        train_epochs=args.train_epochs,
        train_callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        lr=args.lr,
        checkpoint_every_n_epochs=args.checkpoint_every,
    )
    train_time = time.time() - train_start

    print(f"\nTraining done in {train_time:.1f}s ({train_time/60:.1f} min)")

    # ------------------------------------------------------------------
    # Print checkpoint info
    # ------------------------------------------------------------------
    ckpt_dir = exp_dir / "01_checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        print(f"\nCheckpoints ({len(ckpts)} files):")
        for c in ckpts[-5:]:
            print(f"  {c.name}")

    with open(log_file, 'a') as f:
        f.write(f"\nTraining complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {train_time:.1f}s\n")
        f.write(f"Experiment dir: {exp_dir}\n")

    print(f"\nExperiment dir: {exp_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
