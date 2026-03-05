#!/bin/bash
# trial7_watch_and_run.sh
#
# trial7 학습 완료 감지 → 자동으로 실행:
#   1. trial7_interp_validate.py  (hold-out 116d 검증)
#   2. trial6_step4_encode.py --t0_tag t115d_RG  (115d RG perturbation latents)
#      → trial7용 perturb_latents 재생성 필요시 별도 실행
#   3. trial6_traj_divergence.py  (trial7 checkpoint 사용)
#
# 사용법:
#   nohup bash scripts/trial7_watch_and_run.sh > scripts/trial7_watcher.log 2>&1 &

BASE=/data2/Atlas_Normal/IL17RD_scdiffeq
TRIAL7_TRAIN=$BASE/results/trial7/train
LOG=$BASE/scripts/trial7_watcher.log

CONDA_ENV_SDE=scdiffeq_env
CONDA_ENV_ARCHES=scArches_env

echo "======================================================="
echo "Trial7 Watcher started: $(date)"
echo "Watching: $TRIAL7_TRAIN"
echo "======================================================="

# ── 1. trial7 last.ckpt 완료 대기 ─────────────────────────────────────
# 조건: results/trial7/train/*/LightningSDE-*/version_0/checkpoints/last.ckpt 존재
#       AND training.log 에 "Done!" 또는 "Total time:" 포함

POLL_SEC=120   # 2분마다 체크

while true; do
    # last.ckpt 탐색
    CKPT=$(find "$TRIAL7_TRAIN" -name "last.ckpt" 2>/dev/null | head -1)

    if [ -n "$CKPT" ]; then
        # training.log에서 완료 확인
        TRAIN_LOG=$(find "$TRIAL7_TRAIN" -name "training.log" 2>/dev/null | head -1)
        if [ -n "$TRAIN_LOG" ] && grep -q "Done!" "$TRAIN_LOG" 2>/dev/null; then
            echo ""
            echo "[$(date)] Training DONE detected!"
            echo "  Checkpoint: $CKPT"
            break
        fi
    fi

    echo "[$(date)] Waiting... (ckpt=$([ -n "$CKPT" ] && echo found || echo not_found))"
    sleep $POLL_SEC
done

echo ""
echo "======================================================="
echo "Trial7 training complete. Starting downstream pipeline."
echo "======================================================="

# ── 2. Interpolation validation ───────────────────────────────────────
echo ""
echo "[Step A] trial7_interp_validate.py  (GPU 0)"
conda run -n $CONDA_ENV_SDE python $BASE/scripts/trial7_interp_validate.py \
    --ckpt "$CKPT" \
    > $BASE/results/trial7/interp_validate.log 2>&1
echo "  Exit code: $?"
echo "  Log: $BASE/results/trial7/interp_validate.log"

# ── 3. Perturbation encoding (t115d_RG) with trial7 model ─────────────
# Note: 인코딩 자체는 scVI 기반 → trial6 perturb_latents_t115d_RG 재사용 가능
# (새 시작점 인코딩이 필요없으면 이 단계 skip)
echo ""
echo "[Step B] Checking if t115d_RG latents exist..."
LAT_115_RG=$BASE/results/trial6/perturb_latents_t115d_RG
if [ -f "$LAT_115_RG/z_ctrl.npy" ]; then
    echo "  t115d_RG latents found, reusing for traj_divergence."
else
    echo "  t115d_RG latents NOT found — running trial6_step4_encode.py (scArches_env)..."
    conda run -n $CONDA_ENV_ARCHES python $BASE/scripts/trial6_step4_encode.py \
        --t0_tag t115d_RG \
        > $BASE/results/trial6/step4_t115d_RG_redo.log 2>&1
    echo "  Encode exit code: $?"
fi

# t70d_RG 도 확인
LAT_70_RG=$BASE/results/trial6/perturb_latents_t70d_RG
if [ ! -f "$LAT_70_RG/z_ctrl.npy" ]; then
    echo "  t70d_RG latents NOT found — running..."
    conda run -n $CONDA_ENV_ARCHES python $BASE/scripts/trial6_step4_encode.py \
        --t0_tag t70d_RG \
        >> $BASE/results/trial6/step4_t70d_RG_redo.log 2>&1
fi

# ── 4. Trajectory divergence (trial7 checkpoint) ─────────────────────
echo ""
echo "[Step C] trial6_traj_divergence.py  with trial7 ckpt  (t70d_RG)"
TAG_70=trial7_holdout116d_t70d_RG
conda run -n $CONDA_ENV_SDE \
    CUDA_VISIBLE_DEVICES=0 \
    python $BASE/scripts/trial6_traj_divergence.py \
        --ckpt "$CKPT" \
        --t0_tag t70d_RG \
        --tag "$TAG_70" \
    > $BASE/results/trial6/traj_div_trial7_t70d_RG.log 2>&1
echo "  t70d_RG exit code: $?"

echo ""
echo "[Step D] trial6_traj_divergence.py  with trial7 ckpt  (t115d_RG)"
TAG_115=trial7_holdout116d_t115d_RG
conda run -n $CONDA_ENV_SDE \
    CUDA_VISIBLE_DEVICES=0 \
    python $BASE/scripts/trial6_traj_divergence.py \
        --ckpt "$CKPT" \
        --t0_tag t115d_RG \
        --tag "$TAG_115" \
    > $BASE/results/trial6/traj_div_trial7_t115d_RG.log 2>&1
echo "  t115d_RG exit code: $?"

# ── 5. 완료 알림 ──────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "[$(date)] All downstream steps completed!"
echo "  Checkpoint   : $CKPT"
echo "  Interp log   : $BASE/results/trial7/interp_validate.log"
echo "  Traj div log : $BASE/results/trial6/traj_div_trial7_*.log"
echo "======================================================="

# ntfy 알림 (선택)
if command -v curl &>/dev/null; then
    curl -s -d "Trial7 pipeline done! ckpt=$(basename $(dirname $(dirname $CKPT)))" \
        ntfy.sh/atlas_normal_run 2>/dev/null || true
fi
