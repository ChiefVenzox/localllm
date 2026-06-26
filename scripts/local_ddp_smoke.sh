#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

PROCESSES="${PROCESSES:-2}"
PRESET="${PRESET:-nano-demo}"
DATA="${DATA:-data/bin}"
OUT="${OUT:-checkpoints/ddp_smoke}"
MAX_STEPS="${MAX_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEVICE="${DEVICE:-cpu}"
MASTER_PORT="${MASTER_PORT:-}"
export USE_LIBUV="${USE_LIBUV:-0}"

if [ -z "$MASTER_PORT" ]; then
  MASTER_PORT="$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node "$PROCESSES" \
  --master-addr 127.0.0.1 \
  --master-port "$MASTER_PORT" \
  train.py \
  --preset "$PRESET" \
  --data "$DATA" \
  --out "$OUT" \
  --device "$DEVICE" \
  --max-steps "$MAX_STEPS" \
  --lr-decay-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --dist-backend gloo \
  --dist-timeout-minutes 5 \
  "$@"
