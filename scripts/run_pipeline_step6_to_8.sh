#!/bin/bash
# run_pipeline_step6_to_8.sh
# step6 완료 후 step7(t70d, t115d) → 결과 요약까지 순차 실행
# Usage: bash scripts/run_pipeline_step6_to_8.sh

set -e
cd /data2/Atlas_Normal/IL17RD_scdiffeq

LOG_DIR="results/trial6"
SCDIFFEQ_PY="/home/t1/miniconda3/envs/scdiffeq_env/bin/python"
SCARCHES_PY="/home/t1/miniconda3/envs/scArches_env/bin/python"

echo "============================================================"
echo " Full pipeline: step6 → step7(t70d) → step7(t115d)"
echo " $(date)"
echo "============================================================"

# ── step7 t70d_RG ──────────────────────────────────────────────
echo ""
echo "[Step 7] t70d_RG  top_n=150  $(date)"
CUDA_VISIBLE_DEVICES=1 $SCDIFFEQ_PY -u scripts/step7_topgene_simulation.py \
    --t0_tag t70d_RG --top_n 150 \
    2>&1 | tee "$LOG_DIR/step7_t70d.log"

echo ""
echo "[Step 7] t115d_RG  top_n=150  $(date)"
CUDA_VISIBLE_DEVICES=1 $SCDIFFEQ_PY -u scripts/step7_topgene_simulation.py \
    --t0_tag t115d_RG --top_n 150 \
    2>&1 | tee "$LOG_DIR/step7_t115d.log"

echo ""
echo "============================================================"
echo " All steps done.  $(date)"
echo "============================================================"
