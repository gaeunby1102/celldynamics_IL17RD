#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/t1/miniconda3")
SCARCHES="$CONDA_BASE/envs/scArches_env/bin/python"
SCDIFFEQ="$CONDA_BASE/envs/scdiffeq_env/bin/python"
SCRIPTS="/data2/Atlas_Normal/IL17RD_scdiffeq/scripts"
LOG="/data2/Atlas_Normal/IL17RD_scdiffeq/results/new_run/perturb_pipeline.log"
NTFY="ge_notification"

ntfy(){ curl -s -H "Title: $1" -H "Priority: default" -d "$2" "https://ntfy.sh/$NTFY" >/dev/null 2>&1 || true; }
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "===== Perturbation Pipeline START ====="
ntfy "🧬 Perturb START" "Step4 인코딩 시작 (IL17RD, PAX6, NEUROG2, ASCL1, DLX2, HES1)"

log "--- Step 4: Perturbation Encoding (scArches_env) ---"
if $SCARCHES "$SCRIPTS/step4_perturb_encode_scArches_env.py" >> "$LOG" 2>&1; then
    log "Step 4 DONE"
    ntfy "✅ Step 4 Done" "퍼터베이션 인코딩 완료. 시뮬레이션 시작합니다."
else
    log "Step 4 FAILED"
    ntfy "❌ Step 4 FAILED" "인코딩 실패. 로그: $LOG"
    exit 1
fi

log "--- Step 5: Perturbation Simulation (scdiffeq_env) ---"
if $SCDIFFEQ "$SCRIPTS/step5_perturb_simulate_new_model.py" >> "$LOG" 2>&1; then
    log "Step 5 DONE"
    ntfy "🎉 Perturb DONE" "퍼터베이션 시뮬레이션 완료! results/new_run/perturb_results/ 확인"
else
    log "Step 5 FAILED"
    ntfy "❌ Step 5 FAILED" "시뮬레이션 실패. 로그: $LOG"
    exit 1
fi

log "===== Pipeline COMPLETE ====="
