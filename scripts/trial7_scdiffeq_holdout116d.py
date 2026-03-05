#!/usr/bin/env python
"""
trial7_scdiffeq_holdout116d.py  (scdiffeq_env)  —  GPU 0

Trial 7: enforce=1.0, 116d (t=0.5588) hold-out 후 재학습
  학습 타임포인트: 49d / 60d / 70d / 168d  (116d 제외)
  검증: hold-out 116d에서 interpolation 성능 측정

Output: results/trial7/train/
"""

import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])

import sys, time
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from lightning.pytorch.callbacks import Callback
from scdiffeq.core.callbacks import BasicProgressBar

# ── Logging callback (trial5와 동일) ─────────────────────────────────────────
class EpochMetricsLogger(Callback):
    def __init__(self, log_file, log_every_n_epochs=10):
        super().__init__()
        self.log_file = log_file
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch   = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss = metrics.get('epoch_train_loss', None)
        epoch_data = {'epoch': epoch,
                      'train_loss': float(train_loss) if train_loss is not None else None}
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
        print(f"Metrics CSV saved: {csv_file}", flush=True)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR      = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
DATA_PATH     = BASE_DIR / "results" / "trial5" / "trial5_latent_scvi_dim30.h5ad"
EXPERIMENT_DIR= BASE_DIR / "results" / "trial7" / "train"

HOLDOUT_T     = 0.5588   # 116d
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

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Trial 7 — scDiffeq  |  enforce=1.0  |  holdout=116d  |  GPU 0")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nImporting scdiffeq ...", flush=True)
    import scdiffeq as sdq

    # ── 데이터 로드 & 116d 제외 ──────────────────────────────────────────────
    print(f"\n[1] Loading data and removing holdout t={HOLDOUT_T} (116d)...", flush=True)
    t0 = time.time()
    adata_full = ad.read_h5ad(DATA_PATH)

    mask_holdout = (adata_full.obs[TIME_KEY] - HOLDOUT_T).abs() < 1e-4
    n_holdout    = mask_holdout.sum()
    adata        = adata_full[~mask_holdout].copy()

    print(f"    Full data  : {adata_full.n_obs:,} cells", flush=True)
    print(f"    Removed    : {n_holdout:,} cells at t={HOLDOUT_T} (116d)", flush=True)
    print(f"    Train data : {adata.n_obs:,} cells  in {time.time()-t0:.1f}s", flush=True)
    print(f"    Timepoints : {sorted(adata.obs[TIME_KEY].unique())}", flush=True)

    if 'time' not in adata.obs.columns:
        adata.obs['time'] = adata.obs[TIME_KEY].values
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.n_obs)]

    # ── Experiment dir ───────────────────────────────────────────────────────
    exp_name = f"trial7_SDE_holdout116d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir  = EXPERIMENT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] Experiment dir: {exp_dir}", flush=True)

    log_file = str(exp_dir / "training.log")
    with open(exp_dir / "config.txt", 'w') as f:
        f.write(f"data_path      : {DATA_PATH}\n")
        f.write(f"holdout_tp     : 116d (t={HOLDOUT_T}, truly excluded)\n")
        f.write(f"train_tps      : 49d/60d/70d/168d\n")
        f.write(f"use_key        : {USE_KEY}\n")
        f.write(f"latent_dim     : {LATENT_DIM}\n")
        f.write(f"diffeq_type    : SDE\n")
        f.write(f"enforce        : 1.0  target=2.5\n")
        f.write(f"train_epochs   : {TRAIN_EPOCHS}\n")
        f.write(f"batch_size     : {BATCH_SIZE}\n")
        f.write(f"learning_rate  : {LR}\n")
        f.write(f"dt             : {DT}\n")
        f.write(f"mu_hidden      : {MU_HIDDEN}\n")
        f.write(f"sigma_hidden   : {SIGMA_HIDDEN}\n")
        f.write(f"GPU            : CUDA_VISIBLE_DEVICES=0\n")
        f.write(f"seed           : {SEED}\n")

    # ── 모델 초기화 ──────────────────────────────────────────────────────────
    print(f"\n[3] Initializing scDiffeq...", flush=True)
    model = sdq.scDiffEq(
        adata=adata,
        latent_dim=LATENT_DIM,
        use_key=USE_KEY,
        time_key=TIME_KEY,
        DiffEq_type="SDE",
        mu_hidden=MU_HIDDEN,
        sigma_hidden=SIGMA_HIDDEN,
        batch_size=BATCH_SIZE,
        dt=DT,
        working_dir=str(exp_dir),
        velocity_ratio_params={"target": 2.5, "enforce": 1.0, "method": "square"},
    )
    model.configure_data(adata=adata)
    model.configure_model()
    print("    Model initialized.", flush=True)

    # ── 학습 ─────────────────────────────────────────────────────────────────
    print(f"\n[4] Training ({TRAIN_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})...", flush=True)
    callbacks = [
        BasicProgressBar(total_epochs=TRAIN_EPOCHS, print_every=LOG_EVERY),
        EpochMetricsLogger(log_file=log_file, log_every_n_epochs=LOG_EVERY),
    ]

    train_start = time.time()
    model.train(
        train_epochs=TRAIN_EPOCHS,
        train_callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        lr=LR,
        checkpoint_every_n_epochs=CHECKPOINT_EVERY,
    )
    train_time = time.time() - train_start

    print(f"\nTotal time: {train_time:.1f}s ({train_time/3600:.1f}h)", flush=True)
    print(f"Experiment dir: {exp_dir}")
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
