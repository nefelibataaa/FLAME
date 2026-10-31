#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export MODEL="${MODEL:-/apdcephfs_qy3_303725602/share_303725602/CodeLlama-7b-hf}"
export DATA="${DATA:-$HERE/data/smoke_code.jsonl}"
export PYTHON="${PYTHON:-python}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
export OUTPUT="${OUTPUT:-$HERE/outputs/train_$STAMP}"
LOG_FILE="${LOG_FILE:-$HERE/train_$STAMP.log}"
mkdir -p "$(dirname "$LOG_FILE")"
if [[ -e "$LOG_FILE" ]]; then
    echo "日志已存在，请指定新 LOG_FILE: $LOG_FILE" >&2
    exit 1
fi
command -v "$PYTHON" >/dev/null || { echo "找不到 Python: $PYTHON" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "模型目录不存在: $MODEL" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "数据文件不存在: $DATA" >&2; exit 1; }
nohup bash "$HERE/run_train.sh" --epochs 1 --max-length 128 --remove-layers 0 "$@" \
    > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$LOG_FILE.pid"
printf 'PID: %s\n日志: %s\n输出: %s\n' "$PID" "$LOG_FILE" "$OUTPUT"
