#!/bin/bash
# run_pipeline_step7.sh
# step7a (encode, scArches_env) → step7b (simulate, scdiffeq_env) × 2 timepoints
set -e
cd /data2/Atlas_Normal/IL17RD_scdiffeq

SCARCHES="/home/t1/miniconda3/envs/scArches_env/bin/python"
SCDIFFEQ="/home/t1/miniconda3/envs/scdiffeq_env/bin/python"
LOG="results/trial6"

echo "============================================================"
echo " Step 7 full pipeline"
echo " $(date)"
echo "============================================================"

echo ""
echo "[7a] Encode — t70d_RG  $(date)"
CUDA_VISIBLE_DEVICES=1 $SCARCHES -u scripts/step7a_encode_topgenes.py \
    --t0_tag t70d_RG --top_n 150 2>&1 | tee $LOG/step7a_t70d.log

echo ""
echo "[7b] Simulate — t70d_RG  $(date)"
CUDA_VISIBLE_DEVICES=1 $SCDIFFEQ -u scripts/step7b_simulate_topgenes.py \
    --t0_tag t70d_RG 2>&1 | tee $LOG/step7b_t70d.log

echo ""
echo "[7a] Encode — t115d_RG  $(date)"
CUDA_VISIBLE_DEVICES=1 $SCARCHES -u scripts/step7a_encode_topgenes.py \
    --t0_tag t115d_RG --top_n 150 2>&1 | tee $LOG/step7a_t115d.log

echo ""
echo "[7b] Simulate — t115d_RG  $(date)"
CUDA_VISIBLE_DEVICES=1 $SCDIFFEQ -u scripts/step7b_simulate_topgenes.py \
    --t0_tag t115d_RG 2>&1 | tee $LOG/step7b_t115d.log

echo ""
echo "============================================================"
echo " All done.  $(date)"
echo "============================================================"
