#!/usr/bin/env bash
# =============================================================================
# run_pipeline_background.sh
#
# Holdout pipeline: preprocess → train → evaluate
# GPU 1번 (CUDA_VISIBLE_DEVICES=1)
# ntfy 알림: ge_notification
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/results/new_run"
LOG_FILE="$LOG_DIR/pipeline.log"
NTFY_TOPIC="ge_notification"

mkdir -p "$LOG_DIR"

# ntfy 알림 함수
ntfy_send() {
    local title="$1"
    local msg="$2"
    local priority="${3:-default}"
    curl -s \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -d "$msg" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# =============================================================================
# Pipeline start
# =============================================================================

log "============================================================"
log "Pipeline START: holdout t=0.165, cap 10k/tp, GPU 1"
log "============================================================"
ntfy_send "🧬 Pipeline START" "전처리 시작합니다 (t=0.165 holdout, cap 10k)" "default"

# =============================================================================
# Step 1: Preprocessing (scArches_env)
# =============================================================================
log "--- Step 1: Preprocessing ---"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/conda")
SCARCHES_PYTHON="$CONDA_BASE/envs/scArches_env/bin/python"

if [ ! -f "$SCARCHES_PYTHON" ]; then
    # fallback: conda run
    PREPROCESS_CMD="conda run -n scArches_env python $SCRIPT_DIR/preprocess_new_holdout.py"
else
    PREPROCESS_CMD="$SCARCHES_PYTHON $SCRIPT_DIR/preprocess_new_holdout.py"
fi

log "Running: $PREPROCESS_CMD"
if $PREPROCESS_CMD >> "$LOG_FILE" 2>&1; then
    log "Step 1 DONE: Preprocessing complete"
    ntfy_send "✅ Step 1 Done" "전처리 완료! 학습 시작합니다 (GPU 1, 2000 epochs)" "default"
else
    log "Step 1 FAILED: Preprocessing error (see $LOG_FILE)"
    ntfy_send "❌ Step 1 FAILED" "전처리 실패. 로그 확인: $LOG_FILE" "high"
    exit 1
fi

# =============================================================================
# Step 2: Training (scdiffeq_env)
# =============================================================================
log "--- Step 2: Training ---"

SCDIFFEQ_PYTHON="$CONDA_BASE/envs/scdiffeq_env/bin/python"
if [ ! -f "$SCDIFFEQ_PYTHON" ]; then
    TRAIN_CMD="conda run -n scdiffeq_env python $SCRIPT_DIR/train_new_holdout.py"
else
    TRAIN_CMD="$SCDIFFEQ_PYTHON $SCRIPT_DIR/train_new_holdout.py"
fi

log "Running: $TRAIN_CMD"
TRAIN_START=$(date +%s)

if $TRAIN_CMD >> "$LOG_FILE" 2>&1; then
    TRAIN_END=$(date +%s)
    TRAIN_MIN=$(( (TRAIN_END - TRAIN_START) / 60 ))
    log "Step 2 DONE: Training complete (${TRAIN_MIN} min)"
    ntfy_send "✅ Step 2 Done" "학습 완료! (${TRAIN_MIN}분 소요) 평가 시작합니다" "default"
else
    log "Step 2 FAILED: Training error (see $LOG_FILE)"
    ntfy_send "❌ Step 2 FAILED" "학습 실패. 로그 확인: $LOG_FILE" "high"
    exit 1
fi

# =============================================================================
# Step 3: Evaluation (scdiffeq_env)
# =============================================================================
log "--- Step 3: Evaluation ---"

EVAL_CMD="$SCDIFFEQ_PYTHON $SCRIPT_DIR/evaluate_new_holdout.py"
if [ ! -f "$SCDIFFEQ_PYTHON" ]; then
    EVAL_CMD="conda run -n scdiffeq_env python $SCRIPT_DIR/evaluate_new_holdout.py"
fi

log "Running: $EVAL_CMD"
if $EVAL_CMD >> "$LOG_FILE" 2>&1; then
    log "Step 3 DONE: Evaluation complete"

    # summary.json에서 EMD 수치 추출
    SUMMARY_JSON="$BASE_DIR/results/new_run/evaluate/summary.json"
    if [ -f "$SUMMARY_JSON" ]; then
        EMD_PRED=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(f\"{d['mean_emd_predicted']:.4f}\")" 2>/dev/null || echo "?")
        EMD_BASE=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(f\"{d['mean_emd_baseline']:.4f}\")" 2>/dev/null || echo "?")
        IMPR=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(f\"{d['emd_improvement']:.4f}\")" 2>/dev/null || echo "?")
        ntfy_send "🎉 Pipeline DONE" "평가 완료!\n예측 EMD: ${EMD_PRED}\n기준 EMD: ${EMD_BASE}\n개선: ${IMPR}\n결과: results/new_run/evaluate/" "default"
    else
        ntfy_send "🎉 Pipeline DONE" "모든 단계 완료! results/new_run/evaluate/ 확인" "default"
    fi
else
    log "Step 3 FAILED: Evaluation error (see $LOG_FILE)"
    ntfy_send "❌ Step 3 FAILED" "평가 실패. 로그 확인: $LOG_FILE" "high"
    exit 1
fi

log "============================================================"
log "Pipeline COMPLETE. Results: $BASE_DIR/results/new_run/"
log "============================================================"
