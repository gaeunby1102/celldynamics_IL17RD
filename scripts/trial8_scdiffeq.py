#!/usr/bin/env python
"""
trial8_scdiffeq.py  (scdiffeq_env)  —  GPU 0

Trial 8: 모델 튜닝 — drift 안정성 & 168d 과적합 억제
  변경 사항 (vs trial5 enforce1):
    velocity_ratio: target=10.0, enforce=5.0  (drift 강하게 우세)
    sigma_hidden: [16, 16]  (diffusion 더 작게 → enforce 달성 쉬움)
    dt: 0.005  (SDE integration 2배 정밀)
    n_predict: 3000  (학습량 증가)
    168d cells: 5k cap  (terminal state 과적합 억제, 나머지 10k 유지)

Output: results/trial8/train/
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

import time
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from lightning.pytorch.callbacks import Callback
from scdiffeq.core.callbacks import BasicProgressBar

# ── Logging callback ──────────────────────────────────────────────────────────
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
            writer = csv.DictWriter(
                f, fieldnames=['epoch','train_loss','mean_sinkhorn','mean_velo_ratio'])
            writer.writeheader()
            writer.writerows(self.epoch_metrics)
        with open(self.log_file, 'a') as f:
            f.write(f"\nDone! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Metrics CSV: {csv_file}", flush=True)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR       = Path("/data2/Atlas_Normal/IL17RD_scdiffeq")
DATA_PATH      = BASE_DIR / "results" / "trial5" / "trial5_latent_scvi_dim30.h5ad"
EXPERIMENT_DIR = BASE_DIR / "results" / "trial8" / "train"

TIME_KEY  = "age_time_norm"
USE_KEY   = "X_scVI"
SEED      = 42

# ── 튜닝 하이퍼파라미터 ───────────────────────────────────────────────────────
LATENT_DIM    = 30
MU_HIDDEN     = [512, 512]
SIGMA_HIDDEN  = [16, 16]       # ↓ 32→16 (enforce 달성 쉽게)
DT            = 0.005          # ↓ 0.01→0.005 (integration 정밀도)
TRAIN_EPOCHS  = 3000           # ↑ 2000→3000
BATCH_SIZE    = 1024
LR            = 1e-4
CHECKPOINT_EVERY = 25
LOG_EVERY     = 10

VELOCITY_RATIO_PARAMS = {
    "target":  10.0,           # ↑ 2.5→10.0 (drift 강하게 우세)
    "enforce": 5.0,            # ↑ 1.0→5.0
    "method":  "square",
}

CAP_168D = 5000               # 168d 세포 수 제한 (10k→5k, 과적합 억제)
T_168D   = 1.0


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Trial 8 — scDiffeq  (Tuned: drift↑, dt↓, ep↑, 168d cap)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  velocity_ratio: target={VELOCITY_RATIO_PARAMS['target']}  "
          f"enforce={VELOCITY_RATIO_PARAMS['enforce']}")
    print(f"  sigma_hidden={SIGMA_HIDDEN}  dt={DT}  epochs={TRAIN_EPOCHS}")
    print(f"  168d cap: {CAP_168D} cells (others: full)")
    print("=" * 70)

    print("\nImporting scdiffeq ...", flush=True)
    import scdiffeq as sdq
    print(f"  version: {getattr(sdq, '__version__', 'unknown')}", flush=True)

    # ── 1. 데이터 로드 & 168d downsampling ───────────────────────────────────
    print(f"\n[1] Loading data ...", flush=True)
    t0 = time.time()
    adata_full = ad.read_h5ad(DATA_PATH)
    print(f"    Full: {adata_full.n_obs:,} cells  ({time.time()-t0:.1f}s)", flush=True)

    tps = sorted(adata_full.obs[TIME_KEY].unique())
    print(f"    Timepoints: {[round(t,4) for t in tps]}")
    for t in tps:
        n = (adata_full.obs[TIME_KEY] == t).sum()
        print(f"      t={t:.4f}: {n:,} cells")

    # 168d downsample
    rng = np.random.default_rng(SEED)
    mask_168d = (adata_full.obs[TIME_KEY] - T_168D).abs() < 1e-4
    idx_168d  = np.where(mask_168d)[0]
    idx_other = np.where(~mask_168d)[0]

    if len(idx_168d) > CAP_168D:
        idx_168d_keep = rng.choice(idx_168d, CAP_168D, replace=False)
        idx_keep = np.concatenate([idx_other, idx_168d_keep])
        adata = adata_full[np.sort(idx_keep)].copy()
        print(f"\n    168d downsampled: {len(idx_168d):,} → {CAP_168D:,}", flush=True)
    else:
        adata = adata_full.copy()
        print(f"\n    168d already ≤ {CAP_168D}, no downsampling", flush=True)

    print(f"    Final train: {adata.n_obs:,} cells", flush=True)

    if 'time' not in adata.obs.columns:
        adata.obs['time'] = adata.obs[TIME_KEY].values
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.n_obs)]

    # ── 2. Experiment dir ────────────────────────────────────────────────────
    exp_name = f"trial8_SDE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir  = EXPERIMENT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] Experiment dir: {exp_dir}", flush=True)

    log_file = str(exp_dir / "training.log")
    with open(log_file, 'w') as f:
        f.write(f"Trial 8 scDiffeq training log\n")
        f.write(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"[Changes vs trial5 enforce1]\n")
        f.write(f"  velocity_ratio: target={VELOCITY_RATIO_PARAMS['target']}  "
                f"enforce={VELOCITY_RATIO_PARAMS['enforce']}\n")
        f.write(f"  sigma_hidden: {SIGMA_HIDDEN}  (was [32,32])\n")
        f.write(f"  dt: {DT}  (was 0.01)\n")
        f.write(f"  epochs: {TRAIN_EPOCHS}  (was 2000)\n")
        f.write(f"  168d cap: {CAP_168D}  (was 10000)\n\n")

    with open(exp_dir / "config.txt", 'w') as f:
        f.write(f"data_path      : {DATA_PATH}\n")
        f.write(f"train_tps      : all (49d/60d/70d/116d/168d)\n")
        f.write(f"168d_cap       : {CAP_168D}\n")
        f.write(f"use_key        : {USE_KEY}\n")
        f.write(f"latent_dim     : {LATENT_DIM}\n")
        f.write(f"diffeq_type    : SDE\n")
        f.write(f"velo_target    : {VELOCITY_RATIO_PARAMS['target']}\n")
        f.write(f"velo_enforce   : {VELOCITY_RATIO_PARAMS['enforce']}\n")
        f.write(f"train_epochs   : {TRAIN_EPOCHS}\n")
        f.write(f"batch_size     : {BATCH_SIZE}\n")
        f.write(f"learning_rate  : {LR}\n")
        f.write(f"dt             : {DT}\n")
        f.write(f"mu_hidden      : {MU_HIDDEN}\n")
        f.write(f"sigma_hidden   : {SIGMA_HIDDEN}\n")
        f.write(f"GPU            : CUDA_VISIBLE_DEVICES=0\n")
        f.write(f"seed           : {SEED}\n")

    # ── 3. 모델 초기화 ───────────────────────────────────────────────────────
    print(f"\n[3] Initializing scDiffeq ...", flush=True)
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
        velocity_ratio_params=VELOCITY_RATIO_PARAMS,
    )
    model.configure_data(adata=adata)
    model.configure_model()
    print("    Model initialized.", flush=True)

    # ── 4. 학습 ──────────────────────────────────────────────────────────────
    print(f"\n[4] Training ({TRAIN_EPOCHS} epochs, dt={DT}, batch={BATCH_SIZE}) ...", flush=True)
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
