# celldynamics_IL17RD

**scDiffeq-based trajectory perturbation analysis of IL17RD in fetal brain radial glia**

## Overview

This repository contains analysis scripts and results for studying the role of **IL17RD** (IL-17 receptor D) in fetal brain radial glia (RG) development using stochastic differential equation (SDE)-based trajectory modeling.

We use [scDiffeq](https://github.com/scDiffeq/scDiffeq) to simulate cell fate trajectories under in silico gene knockouts, enabling genome-wide perturbation screening without experimental intervention.

## Pipeline

```
Raw scRNA-seq atlas (fetal brain, n~50k cells)
        ↓
  [scVI] Latent embedding (dim=30)
        ↓
  [scDiffeq] SDE trajectory model (t=0 → t=1.0)
        ↓
  Step 6: All-gene KO scan (5,011 HVGs)
          → KO at t0: set gene count → 0, re-encode via scVI → paired L2
        ↓
  Step 7: Top gene trajectory simulation
          → Top 150 bio genes → scDiffeq simulation → L2 at t=1.0
        ↓
  Step 8: Gene program validation
          → KO direction (Δz PC1) × atlas gene expression → Pearson r
        ↓
  Residual L2 analysis
          → Correct for expression confound → expression-independent ranking
```

## Key Results

### All-gene KO Landscape (Step 6)
- 5,011 HVG scan; cross-timepoint Spearman r = **0.877** (t70d vs t115d)
- Housekeeping gene filter removes MT-\*, RPL\*, RPS\*, MALAT1 etc.

### Top-gene Trajectory Simulation (Step 7)
- Scan L2 vs Simulation L2 Spearman r = **0.97** → encoding shift predicts trajectory divergence
- Top divergent genes: NFIA, NFIB, PTN, NRXN1, SOX11, AUTS2

### Expression-corrected Ranking (Residual L2)
- **88-91% of raw L2 variance** explained by expression level alone
- After correction: NFIA, NFIB, SOX11 remain top; PAX6 improves (raw 33 → corrected rank 12)
- IL17RD: low raw rank (966) but corrected rank (565) shows modest specific effect

### Gene Program Validation (Step 8)
- IL17RD KO direction matches 4/4 known targets (NR2F1, FGFR3 upregulated)
- Best known-target match rate among all 6 reference genes

## Reference genes
| Gene | Role | t70d bio rank | t70d corrected rank |
|------|------|:---:|:---:|
| IL17RD | Study target | 966 | 565 |
| PAX6 | RG identity TF | 33 | 12 |
| HES1 | Notch effector | 23 | 88 |
| ASCL1 | Neurogenic TF | 451 | 4441 |
| NEUROG2 | Neurogenic TF | 481 | 160 |
| DLX2 | GABAergic TF | 1337 | 4079 |

## Scripts

| Script | Environment | Description |
|--------|-------------|-------------|
| `scripts/step6_allgene_ko_scan.py` | scArches_env | Genome-wide KO encoding scan |
| `scripts/step7a_encode_topgenes.py` | scArches_env | Re-encode top genes |
| `scripts/step7b_simulate_topgenes.py` | scdiffeq_env | Trajectory simulation |
| `scripts/step8_gene_program_validation.py` | scArches_env | Gene program validation |
| `scripts/fig_IL17RD_perturbation.R` | BrainAtlas (R) | Main perturbation figures (A-D) |
| `scripts/figEF_allgene_landscape.R` | BrainAtlas (R) | All-gene landscape figures (E-G) |
| `scripts/residual_l2_ranking.R` | BrainAtlas (R) | Expression-corrected ranking (H) |
| `scripts/run_pipeline_step7.sh` | bash | Pipeline: step7a + 7b × 2 timepoints |

## Environments

- **scArches_env**: scVI, scArches (for encoding/KO perturbation)
- **scdiffeq_env**: scDiffeq (for trajectory simulation; no scvi import due to JAX conflict)
- **BrainAtlas**: R 4.3.3 with ggplot2, patchwork, ggrepel

## Data

> Large binary files (*.h5ad, *.npy, *.ckpt) are not tracked by git.

| File | Description |
|------|-------------|
| `results/trial6/allgene_scan/allgene_ko_scan_t70d_RG_bio.csv` | All-gene scan results (t70d) |
| `results/trial6/allgene_scan/allgene_ko_scan_t115d_RG_bio.csv` | All-gene scan results (t115d) |
| `results/trial6/allgene_scan/topgene_sim_t70d_RG.csv` | Simulation results (t70d) |
| `results/trial6/allgene_scan/topgene_sim_t115d_RG.csv` | Simulation results (t115d) |
| `results/trial6/allgene_scan/*_residual.csv` | Expression-corrected rankings |
| `results/trial6/gene_expression_recon/gene_program_validation/` | Gene program validation |

## Citation

- scDiffeq: [Kaminski et al.](https://github.com/scDiffeq/scDiffeq)
- scVI: [Lopez et al., Nature Methods 2018](https://doi.org/10.1038/s41592-018-0229-2)
- scArches: [Lotfollahi et al., Nature Biotechnology 2022](https://doi.org/10.1038/s41587-021-01001-7)
