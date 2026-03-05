#!/bin/bash
# enforce=1.0 학습 완료 대기 후 step5→6 자동 실행
TAG=$1          # e.g. "enforce1"
CKPT_GLOB=$2   # e.g. "*enforce1*/checkpoints/last.ckpt"
LOG=$3          # e.g. "results/trial5/train/scdiffeq_enforce1.log"

TRIAL5="/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial5"
OUT_LOG="$TRIAL5/train/perturb_${TAG}.log"

echo "[$(date)] Watching for training completion: $TAG"
echo "[$(date)] Polling: $LOG"

# 학습 완료 대기 ("Training complete" 문자열 감지)
while true; do
    if grep -q "Done!" "$LOG" 2>/dev/null; then
        echo "[$(date)] Training complete detected!"
        break
    fi
    sleep 120
done

# 체크포인트 경로 찾기
CKPT=$(find "$TRIAL5/train" -path "*${TAG}*/last.ckpt" 2>/dev/null | head -1)
echo "[$(date)] Checkpoint: $CKPT"
curl -s -d "🧠 Trial5 $TAG 학습 완료! perturbation 분석 시작..." ntfy.sh/9aeun

# step5
echo "[$(date)] Running step5 (tag=$TAG)..."
conda run -n scdiffeq_env python scripts/trial5_step5_simulate.py \
    --ckpt "$CKPT" --tag "$TAG" >> "$OUT_LOG" 2>&1
echo "[$(date)] step5 done."

# step6
echo "[$(date)] Running step6 (tag=$TAG)..."
conda run -n scdiffeq_env python scripts/trial5_step6_drift.py \
    --ckpt "$CKPT" --tag "$TAG" >> "$OUT_LOG" 2>&1
echo "[$(date)] step6 done. Output: $OUT_LOG"

# 비교 summary (enforce0 결과가 있으면 같이 비교)
echo "[$(date)] Running comparison summary..."
conda run -n scdiffeq_env python scripts/trial5_compare_enforce.py >> "$OUT_LOG" 2>&1
echo "[$(date)] ALL DONE for $TAG"

# ntfy 알림
curl -s -d "✅ Trial5 $TAG 완료! step5/6 + 비교 그래프 생성됨" ntfy.sh/9aeun
