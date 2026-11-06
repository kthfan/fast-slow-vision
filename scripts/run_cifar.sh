#!/usr/bin/env bash
# Train + evaluate a recurrent bViT on CIFAR using every visible GPU (Kaggle: 2x T4).
#
#   bash scripts/run_cifar.sh <run_name> [extra train.py args...]
#
# Env overrides: DATASET (cifar100), EPOCHS (200), MODEL (bvit_s), TMAX (8),
#                TIME_LIMIT (minutes, 0 = none), EVAL_ARGS (extra evaluate.py args).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${1:-c100_bvit_s}
shift || true
DATASET=${DATASET:-cifar100}
EPOCHS=${EPOCHS:-200}
MODEL=${MODEL:-bvit_s}
TMAX=${TMAX:-8}
TIME_LIMIT=${TIME_LIMIT:-0}
EVAL_ARGS=${EVAL_ARGS:-"--latency"}
OUT=runs/$NAME

NGPU=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
NCPU=$(nproc)
WORKERS=$(( NCPU / NGPU ))
[ "$WORKERS" -lt 1 ] && WORKERS=1
echo "[run_cifar] name=$NAME gpus=$NGPU cpus=$NCPU workers/gpu=$WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Download once before launching workers.
python -c "
import sys; sys.path.insert(0, '.')
from fsv.data import build_splits
build_splits('$DATASET', 'data', download=True)
"

torchrun --standalone --nproc_per_node="$NGPU" scripts/train.py \
  --dataset "$DATASET" --model "$MODEL" --t-max "$TMAX" --epochs "$EPOCHS" \
  --workers "$WORKERS" --time-limit "$TIME_LIMIT" --out "$OUT" --resume "$@"

python scripts/evaluate.py --ckpt "$OUT/model.pt" --workers "$NCPU" $EVAL_ARGS
