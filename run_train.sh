#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
: "${MODEL:?请设置 MODEL 为本地 CodeLlama 模型目录或 Hugging Face checkpoint ID}"
: "${DATA:?请设置 DATA 为带 code、label、language 字段的 JSONL 数据文件}"
PYTHON="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON")" || { echo "找不到 Python: $PYTHON" >&2; exit 1; }
OUTPUT="${OUTPUT:-$HERE/outputs/train_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE 必须为正整数" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
if [[ "$DEVICE" == cuda* ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    "$PYTHON_BIN" -c 'import sys,torch; n=int(sys.argv[1]); print(f"[预检] Python={sys.executable} torch={torch.__version__} CUDA={torch.version.cuda} 可见卡数={torch.cuda.device_count()}",flush=True); sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= n else "CUDA不可用或可见GPU数少于训练进程数")' "$NPROC_PER_NODE"
fi
ARGS=(run.py train --model "$MODEL" --data "$DATA" --device "$DEVICE"
    --output "$OUTPUT" --max-length 512 --batch-size 1 --rank 8 --lora-alpha 32
    --dropout 0.05 --lr 1e-4 --gradient-checkpointing "$@")
echo "[启动] 进程数=$NPROC_PER_NODE 可见GPU=${CUDA_VISIBLE_DEVICES:-未设置} 输出=$OUTPUT"
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    exec "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1 \
        --nproc_per_node "$NPROC_PER_NODE" "${ARGS[@]}"
fi
exec "$PYTHON_BIN" -u "${ARGS[@]}"
