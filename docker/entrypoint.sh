#!/bin/sh
set -eu

role="${1:-api}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$role" in
  api)
    exec python -m uvicorn server.app:app \
      --host "${YERELLM_HOST:-0.0.0.0}" \
      --port "${YERELLM_PORT:-8000}" \
      "$@"
    ;;

  worker)
    set -- python -m worker.local_node \
      --server "${YERELLM_SERVER_URL:-http://api:8000}" \
      --name "${YERELLM_WORKER_NAME:-docker-worker}" \
      --repo /app \
      --node-id "${YERELLM_NODE_ID:-docker-worker}" \
      --node-role "${YERELLM_NODE_ROLE:-docker}" \
      --device "${YERELLM_DEVICE:-auto}" \
      --api-token "${YERELLM_API_TOKEN:-}" \
      "$@"
    if [ "${YERELLM_ALLOW_TRAINING_JOBS:-false}" = "true" ]; then
      set -- "$@" --allow-training-jobs
    fi
    if [ "${YERELLM_ALLOW_REMOTE_COMMANDS:-false}" = "true" ]; then
      set -- "$@" --allow-remote-commands
    fi
    exec "$@"
    ;;

  train)
    exec python train.py "$@"
    ;;

  generate)
    exec python generate.py "$@"
    ;;

  prepare-data)
    exec python -m data.prepare_data "$@"
    ;;

  make-chat-tokens)
    exec python -m data.make_chat_tokens "$@"
    ;;

  doctor|system-check)
    exec python scripts/system_check.py "$@"
    ;;

  bash|sh|python)
    exec "$role" "$@"
    ;;

  *)
    exec "$role" "$@"
    ;;
esac
