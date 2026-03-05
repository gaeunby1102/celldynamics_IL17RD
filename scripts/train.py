#!/usr/bin/env python
# For PyTorch 2.6+ checkpoint compatibility - patch torch.load
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

"""
01_train_scDiffEq.py

Train scDiffEq model 

This script trains a Stochastic Differential Equation (SDE) model to learn
the dynamics of neural differentiation from NPCs to neurons.


"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import anndata as ad
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6+ checkpoint loading with optimizer state
from torch.optim import RMSprop, Adam, SGD
import torch.serialization
torch.serialization.add_safe_globals([RMSprop, Adam, SGD])
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from lightning.pytorch.callbacks import EarlyStopping, Callback
from scdiffeq.core.callbacks import BasicProgressBar

# =============================================================================
# Custom Logging Callback
# =============================================================================

class EpochMetricsLogger(Callback):
    """Custom callback to log metrics at the end of each epoch."""

    def __init__(self, log_file: str, log_every_n_epochs: int = 10):
        super().__init__()
        self.log_file = log_file
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        epoch = trainer.current_epoch

        # Get logged metrics from the trainer
        metrics = trainer.callback_metrics

        # Extract key metrics
        train_loss = metrics.get('epoch_train_loss', None)

        # Store metrics
        epoch_data = {
            'epoch': epoch,
            'train_loss': float(train_loss) if train_loss is not None else None,
        }

        # Try to get sinkhorn and velocity ratio metrics (they are per-timepoint)
        sinkhorn_sum = 0
        velo_ratio_sum = 0
        sinkhorn_count = 0
        velo_count = 0

        for key, value in metrics.items():
            if 'sinkhorn' in key and 'training' in key:
                try:
                    sinkhorn_sum += float(value)
                    sinkhorn_count += 1
                except:
                    pass
            if 'velo_ratio' in key and 'training' in key:
                try:
                    velo_ratio_sum += float(value)
                    velo_count += 1
                except:
                    pass

        epoch_data['mean_sinkhorn'] = sinkhorn_sum / sinkhorn_count if sinkhorn_count > 0 else None
        epoch_data['mean_velo_ratio'] = velo_ratio_sum / velo_count if velo_count > 0 else None

        self.epoch_metrics.append(epoch_data)

        # Log to console and file at specified intervals
        if epoch % self.log_every_n_epochs == 0 or epoch == 0:
            msg = f"[Epoch {epoch:4d}]"
            if epoch_data['train_loss'] is not None:
                msg += f" loss={epoch_data['train_loss']:.4f}"
            if epoch_data['mean_sinkhorn'] is not None:
                msg += f" sinkhorn={epoch_data['mean_sinkhorn']:.2f}"
            if epoch_data['mean_velo_ratio'] is not None:
                msg += f" velo_ratio={epoch_data['mean_velo_ratio']:.2f}"

            print(msg, flush=True)

            # Also write to log file
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

    def on_train_end(self, trainer, pl_module):
        """Save all metrics to CSV at the end of training."""
        import csv
        csv_file = self.log_file.replace('.log', '_metrics.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'mean_sinkhorn', 'mean_velo_ratio'])
            writer.writeheader()
            writer.writerows(self.epoch_metrics)
        print(f"Metrics saved to: {csv_file}")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Training configuration."""

    # Data paths - use normalized data (time scaled to [0, 1])
    DATA_PATH = "/data2/Atlas_Normal/IL17RD_scdiffeq/results/Input_fetal_neuron_trainset_after_scVI_hvg_5010_latent_dim30.h5ad"
    EXPERIMENT_DIR = "/data2/Atlas_Normal/IL17RD_scdiffeq/results/train_scVI/dim_test"

    # Model parameters
    DIFFEQ_TYPE = "SDE"  # "SDE" or "ODE"
    LATENT_DIM = 30     # Latent space dimension (matches X_scVI)
    DT = 0.01           # Integration step size (small for normalized time [0,1])

    # Drift network (deterministic velocity)
    MU_HIDDEN = [512, 512]

    # Diffusion network (stochastic noise) - only for SDE
    SIGMA_HIDDEN = [32, 32]

    # Training parameters
    # NOTE: When latent_dim == data_dim (e.g., using X_pca), there's no VAE
    # In that case, pretrain is not needed - just call fit() directly
    PRETRAIN_EPOCHS = 0      # Skip pretraining when no VAE (latent_dim == data_dim)
    TRAIN_EPOCHS = 2000      # Main training epochs (following tutorial)
    BATCH_SIZE = 1024        # Batch size
    LEARNING_RATE = 1e-4     # Learning rate

    # Early stopping - disabled by default as high velocity ratio loss is expected initially
    # The model needs many epochs to learn the dynamics
    EARLY_STOP_PATIENCE_PRETRAIN = None   # Patience for pretraining (None = disabled)
    EARLY_STOP_PATIENCE_TRAIN = None      # Patience for main training (None = disabled)

    # Data parameters
    TIME_KEY = "age_time_norm"                    # Column name for time
    CELL_TYPE_KEY = "cluster_annotated"     # Column name for cell type
    USE_KEY = "X_scVI"                    # Feature key for model input (e.g., X_scVI=100D, X_scVI_sampleID=10D, X_pca=50D)

    # Hardware
    ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    DEVICES = 1

    # Reproducibility
    SEED = 42

    # Logging
    LOG_EVERY_N_EPOCHS = 10              # Log metrics every N epochs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train scDiffEq model")

    parser.add_argument("--data_path", type=str, default=Config.DATA_PATH,
                        help="Path to preprocessed h5ad file")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment (default: auto-generated)")
    parser.add_argument("--diffeq_type", type=str, default=Config.DIFFEQ_TYPE,
                        choices=["SDE", "ODE"], help="Type of differential equation")
    parser.add_argument("--latent_dim", type=int, default=Config.LATENT_DIM,
                        help="Latent space dimension")
    parser.add_argument("--pretrain_epochs", type=int, default=Config.PRETRAIN_EPOCHS,
                        help="Number of VAE pretraining epochs")
    parser.add_argument("--train_epochs", type=int, default=Config.TRAIN_EPOCHS,
                        help="Number of main training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Random seed")
    parser.add_argument("--velocity_ratio_enforce", type=float, default=0.0,
                        help="Velocity ratio enforcement strength (0=disabled, default for stability)")
    parser.add_argument("--use_key", type=str, default=Config.USE_KEY,
                        choices=["X_pca", "X_pca_harmony", "X_scpoli", "X_scVI", "X_scVI_sampleID"],
                        help="Embedding key to use")
    parser.add_argument("--dt", type=float, default=Config.DT,
                        help="Integration step size (default: 0.001 for normalized time)")
    parser.add_argument("--checkpoint_every", type=int, default=25,
                        help="Save checkpoint every N epochs (default: 25)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with fewer epochs")

    return parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================

def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_experiment_name(diffeq_type: str) -> str:
    """Generate experiment name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"IL17RD_{diffeq_type}_{timestamp}"


def print_data_summary(adata: ad.AnnData):
    """Print summary of loaded data."""
    print(f"\nData Summary:")
    print(f"  Cells: {adata.shape[0]:,}")
    print(f"  Genes: {adata.shape[1]:,}")
    # Time is normalized [0,1]; show both if original available
    #time_min, time_max = adata.obs['time'].min(), adata.obs['time'].max()
    #if 'time_original' in adata.obs.columns:
    #    orig_min, orig_max = adata.obs['Age'].min(), adata.obs['Age'].max()
    #    print(f"  Time range: {time_min:.3f} - {time_max:.3f} (normalized), {orig_min:.0f} - {orig_max:.0f} days")
    #else:
    #    print(f"  Time range: {time_min:.3f} - {time_max:.3f} (normalized)")
    #print(f"  Timepoints: {adata.obs['time'].nunique()}")

    print(f"\n  Cell type distribution:")
    for ct, count in adata.obs['cluster_annotated'].value_counts().items():
        pct = count / len(adata) * 100
        print(f"    {ct}: {count:,} ({pct:.1f}%)")

    print(f"\n  Available embeddings:")
    for key in adata.obsm.keys():
        print(f"    {key}: {adata.obsm[key].shape}")

    print(f"\n  Available layers:")
    for key in adata.layers.keys():
        print(f"    {key}")


# =============================================================================
# Main Training Function
# =============================================================================

def train_model(args):
    """Main training function."""

    print_section("scDiffEq Model Training")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-set latent_dim based on use_key if using default
    embedding_dims = {
        "X_pca": 50,
        "X_pca_harmony": 50,
        "X_scpoli": 10,
        "X_scVI": 30,
        "X_scVI_sampleID": 10,
    }
    expected_dim = embedding_dims.get(args.use_key, 30)
    if args.latent_dim != expected_dim:
        print(f"\nWARNING: use_key={args.use_key} has {expected_dim}D, but latent_dim={args.latent_dim}")
        print(f"Auto-adjusting latent_dim to {expected_dim}")
        args.latent_dim = expected_dim

    # Set seed
    set_seed(args.seed)
    print(f"\nRandom seed: {args.seed}")

    # Import scDiffEq
    print("\nImporting scDiffEq...")
    try:
        import scdiffeq as sdq
        print(f"  scDiffEq version: {sdq.__version__ if hasattr(sdq, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"ERROR: Failed to import scDiffEq: {e}")
        print("\nPlease install scDiffEq:")
        print("  cd /opt/dlami/nvme/model3/scDiffEq && pip install -e .")
        sys.exit(1)

    # ==========================================================================
    # Step 1: Load Data
    # ==========================================================================
    print_section("Step 1: Loading Data")

    print(f"Loading: {args.data_path}")
    start_time = time.time()
    adata = ad.read_h5ad(args.data_path)
    
    adata.obs['time'] = (adata.obs['Age'] - adata.obs['Age'].min()) / (adata.obs['Age'].max() - adata.obs['Age'].min())
    print(f"Loaded in {time.time() - start_time:.1f}s")
    
    adata.obs['original_barcode'] = adata.obs_names.copy()
    adata.obs_names = [str(i) for i in range(adata.shape[0])]
    
    print(f"  -> Index reset complete. Total cells: {adata.shape[0]}")
    print(f"  -> [Check] New Index (0~4):      {adata.obs_names[:5].tolist()}")
    print(f"  -> [Check] Original Barcodes:    {adata.obs['original_barcode'].iloc[:5].tolist()}")

    print_data_summary(adata)

    # ==========================================================================
    # Step 2: Setup Experiment Directory
    # ==========================================================================
    print_section("Step 2: Setting Up Experiment")

    experiment_name = args.experiment_name or get_experiment_name(args.diffeq_type)
    experiment_dir = Path(Config.EXPERIMENT_DIR) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {experiment_name}")
    print(f"Experiment directory: {experiment_dir}")

    # Save config
    config_path = experiment_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"# scDiffEq Training Configuration\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"use_key: {args.use_key}\n")
        f.write(f"diffeq_type: {args.diffeq_type}\n")
        f.write(f"latent_dim: {args.latent_dim}\n")
        f.write(f"pretrain_epochs: {args.pretrain_epochs}\n")
        f.write(f"train_epochs: {args.train_epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"learning_rate: {args.lr}\n")
        f.write(f"dt: {args.dt}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"mu_hidden: {Config.MU_HIDDEN}\n")
        f.write(f"sigma_hidden: {Config.SIGMA_HIDDEN}\n")
    print(f"Config saved to: {config_path}")

    # ==========================================================================
    # Step 3: Initialize Model
    # ==========================================================================
    print_section("Step 3: Initializing Model")

    print(f"DiffEq type: {args.diffeq_type}")
    print(f"Use key: {args.use_key}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Integration step (dt): {args.dt}")
    print(f"Drift network hidden layers: {Config.MU_HIDDEN}")
    if args.diffeq_type == "SDE":
        print(f"Diffusion network hidden layers: {Config.SIGMA_HIDDEN}")

    # Initialize scDiffEq model
    # Note: sigma_hidden is provided for both SDE and ODE (ignored for ODE internally)
    # Velocity ratio parameters
    # NOTE: enforce=0 disables velocity ratio regularization for stability
    # High velocity ratios (~600) with enforce=100 causes 37M loss -> gradient explosion -> NaN
    velocity_ratio_params = {
        "target": 2.5,
        "enforce": args.velocity_ratio_enforce,
        "method": "square",
    }
    print(f"Velocity ratio params: {velocity_ratio_params}")

    model = sdq.scDiffEq(
        adata=adata,
        latent_dim=args.latent_dim,
        use_key=args.use_key,
        time_key=Config.TIME_KEY,
        DiffEq_type=args.diffeq_type,
        mu_hidden=Config.MU_HIDDEN,
        sigma_hidden=Config.SIGMA_HIDDEN,
        batch_size=args.batch_size,
        dt=args.dt,
        working_dir=str(experiment_dir),
        velocity_ratio_params=velocity_ratio_params,
    )

    print("\nModel initialized successfully!")
    print(f"Model type: {type(model)}")

    # ==========================================================================
    # Step 4: Configure Training
    # ==========================================================================
    print_section("Step 4: Configuring Training")

    print(f"Accelerator: {Config.ACCELERATOR}")
    print(f"Devices: {Config.DEVICES}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Configure data and model
    model.configure_data(adata=adata)
    model.configure_model()

    print("Training configuration complete!")

    # ==========================================================================
    # Step 5: Pretrain VAE (if applicable)
    # ==========================================================================
    # NOTE: Pretraining is only needed when using VAE (data_dim > latent_dim)
    # When using X_pca directly (data_dim == latent_dim), skip pretraining
    if args.pretrain_epochs > 0:
        print_section("Step 5: VAE Pretraining")

        print(f"Pretraining epochs: {args.pretrain_epochs}")
        print(f"Early stopping patience: {Config.EARLY_STOP_PATIENCE_PRETRAIN}")
        print("This step trains the encoder/decoder...")

        # Early stopping for pretraining (if enabled)
        pretrain_callbacks = []
        if Config.EARLY_STOP_PATIENCE_PRETRAIN is not None:
            early_stop_pretrain = EarlyStopping(
                monitor="epoch_train_loss",
                patience=Config.EARLY_STOP_PATIENCE_PRETRAIN,
                mode="min",
                verbose=True,
            )
            pretrain_callbacks.append(early_stop_pretrain)

        pretrain_start = time.time()
        model.pretrain(
            pretrain_epochs=args.pretrain_epochs,
            pretrain_callbacks=pretrain_callbacks,
            accelerator=Config.ACCELERATOR,
            devices=Config.DEVICES,
        )
        pretrain_time = time.time() - pretrain_start
        print(f"Pretraining completed in {pretrain_time:.1f}s ({pretrain_time/60:.1f} min)")
    else:
        print_section("Step 5: Skipping Pretraining")
        print("Pretraining is disabled (pretrain_epochs=0)")
        print("This is expected when latent_dim == data_dim (no VAE)")

    # ==========================================================================
    # Step 6: Main Training
    # ==========================================================================
    print_section("Step 6: Main Training")

    print(f"Training epochs: {args.train_epochs}")
    print(f"Early stopping patience: {Config.EARLY_STOP_PATIENCE_TRAIN}")
    print(f"Training the full SDE model (Drift + Diffusion networks)...")
    print("\nNOTE: High initial velocity ratio loss (~620) is expected and will decrease over time.")
    print("This may take a while. Monitor progress in the logs.")

    # Early stopping for main training (if enabled)
    train_callbacks = []
    if Config.EARLY_STOP_PATIENCE_TRAIN is not None:
        early_stop_train = EarlyStopping(
            monitor="epoch_train_loss",
            patience=Config.EARLY_STOP_PATIENCE_TRAIN,
            mode="min",
            verbose=True,
        )
        train_callbacks.append(early_stop_train)

    # Add BasicProgressBar for nicer output with train/val loss
    progress_bar = BasicProgressBar(
        total_epochs=args.train_epochs,
        print_every=Config.LOG_EVERY_N_EPOCHS
    )
    train_callbacks.append(progress_bar)
    print(f"Progress will be logged every {Config.LOG_EVERY_N_EPOCHS} epochs")

    # Handle resume from checkpoint
    resume_ckpt = args.resume_from
    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            print(f"\nRESUMING from checkpoint: {resume_ckpt}")
        else:
            print(f"\nWARNING: Checkpoint not found: {resume_ckpt}")
            print("Starting training from scratch.")
            resume_ckpt = None

    train_start = time.time()
    model.train(
        train_epochs=args.train_epochs,
        train_callbacks=train_callbacks,
        accelerator=Config.ACCELERATOR,
        devices=Config.DEVICES,
        lr=args.lr,
        checkpoint_every_n_epochs=args.checkpoint_every,
    )
    train_time = time.time() - train_start
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    # ==========================================================================
    # Step 7: Save Model
    # ==========================================================================
    print_section("Step 7: Saving Model")

    # The model should be automatically saved by PyTorch Lightning
    # Let's also save a reference to the best checkpoint
    checkpoint_dir = experiment_dir / "01_checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            print(f"Checkpoints saved to: {checkpoint_dir}")
            for ckpt in checkpoints:
                print(f"  - {ckpt.name}")

    # Save the final model reference
    model_info_path = experiment_dir / "model_info.txt"
    with open(model_info_path, 'w') as f:
        f.write(f"# scDiffEq Model Info\n")
        f.write(f"# Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"experiment_dir: {experiment_dir}\n")
        f.write(f"total_training_time: {train_time:.1f}s\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"n_cells: {adata.shape[0]}\n")
        f.write(f"n_genes: {adata.shape[1]}\n")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_section("Training Complete!")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run 02_evaluate_model.py to evaluate the trained model")
    print(f"  2. Run 03_simulate_and_predict_trajectory.py to simulate cell trajectories")

    return model, experiment_dir


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    # Debug mode: reduce epochs
    if args.debug:
        print("\n*** DEBUG MODE: Reducing epochs for quick testing ***")
        args.pretrain_epochs = 10
        args.train_epochs = 20

    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        print("\nPlease run 00_preprocessing.py first to create the preprocessed data.")
        sys.exit(1)

    # Train model
    model, experiment_dir = train_model(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
