#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?NODE_RANK gerekli, ornek: NODE_RANK=2}"
: "${MASTER_ADDR:?MASTER_ADDR gerekli, ornek: MASTER_ADDR=192.168.1.20}"

NNODES="${NNODES:-3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOCAL_RANK="${LOCAL_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
PRESET="${PRESET:-small-100m}"
DATA="${DATA:-data/bin}"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
export USE_LIBUV="${USE_LIBUV:-0}"
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"
export RANK="${RANK:-$((NODE_RANK * NPROC_PER_NODE + LOCAL_RANK))}"
export LOCAL_RANK

if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ "$NPROC_PER_NODE" -gt 1 ]; then
  echo "[lan_train] NPROC_PER_NODE > 1 icin her local rank'i ayri terminalde baslat: LOCAL_RANK=0, LOCAL_RANK=1, ..."
fi

"$PYTHON_BIN" train.py \
  --preset "$PRESET" \
  --data "$DATA" \
  --device "$DEVICE" \
  --dist-backend gloo \
  "$@"
