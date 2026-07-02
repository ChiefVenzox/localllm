# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG APP_UID=1000
ARG APP_GID=1000
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG INSTALL_TORCH=1
ARG INSTALL_BITSANDBYTES=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    YERELLM_HOST=0.0.0.0 \
    YERELLM_PORT=8000 \
    YERELLM_CKPT=/app/checkpoints/ckpt.pt \
    YERELLM_ADAPTER=auto \
    YERELLM_TOKENIZER=/app/tokenizer/tokenizer.json \
    YERELLM_REGISTRY_DB=/app/state/local_nodes.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ "${INSTALL_TORCH}" = "1" ]; then \
        python -m pip install torch --index-url "${TORCH_INDEX_URL}"; \
    fi \
    && awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*(torch|bitsandbytes)([<>=[:space:];#].*)?$/ {next} {print}' \
        /app/requirements.txt > /tmp/requirements-runtime.txt \
    && python -m pip install -r /tmp/requirements-runtime.txt \
    && if [ "${INSTALL_BITSANDBYTES}" = "1" ]; then \
        python -m pip install bitsandbytes; \
    fi

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/sh app

COPY --chown=app:app . /app

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/checkpoints /app/data /app/state \
    && chown -R app:app /app/checkpoints /app/data /app/state

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python /app/docker/healthcheck.py

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]
CMD ["api"]
