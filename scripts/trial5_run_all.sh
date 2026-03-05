#!/bin/bash
set -e
BASE=/data2/Atlas_Normal/IL17RD_scdiffeq
cd $BASE

echo "[$(date)] Starting trial5_scVI.py..."
conda run -n scArches_env python -u scripts/trial5_scVI.py \
  >> results/trial5/scVI.log 2>&1

curl -s -d "trial5 scVI (dim 10/30/100 × HVG A/B) 완료" \
  ntfy.sh/ge_notification > /dev/null

echo "[$(date)] Starting trial5_MrVI.py..."
conda run -n scArches_env python -u scripts/trial5_MrVI.py \
  >> results/trial5/mrvi.log 2>&1

curl -s -d "trial5 MrVI (dim 10/30/100 × HVG A/B) 완료 — 전체 비교 완성!" \
  ntfy.sh/ge_notification > /dev/null

echo "[$(date)] All done."
