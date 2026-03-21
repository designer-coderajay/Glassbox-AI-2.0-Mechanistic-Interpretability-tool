# =============================================================================
# Glassbox AI — Production Dockerfile v3.4.0
# =============================================================================
# Supports two build targets:
#
#   api        — FastAPI compliance REST API  (default)
#   dashboard  — Gradio compliance dashboard
#
# Quick start
# -----------
#   docker build --target api -t glassbox-api:3.4.0 .
#   docker run -p 8000:8000 glassbox-api:3.4.0
#
# Production stack
#   docker compose up   (see docker-compose.yml)
#
# Air-gapped / VPC deployment
#   docker save glassbox-api:3.4.0 | gzip > glassbox-api-3.4.0.tar.gz
#   docker load < glassbox-api-3.4.0.tar.gz
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — base: slim Python 3.11 with system libraries
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Glassbox AI"
LABEL org.opencontainers.image.description="EU AI Act Annex IV compliance documentation toolkit"
LABEL org.opencontainers.image.version="3.4.0"
LABEL org.opencontainers.image.authors="Ajay Pravin Mahale <mahale.ajay01@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool"
LABEL org.opencontainers.image.licenses="BSL-1.1"

# System packages (BLAS for torch, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — mandatory in regulated financial / healthcare environments
RUN groupadd --gid 1001 glassbox \
    && useradd --uid 1001 --gid glassbox --shell /bin/bash --create-home glassbox

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2 — builder: install Python dependencies in layers
# ---------------------------------------------------------------------------
FROM base AS builder

RUN pip install --no-cache-dir --upgrade pip==24.3.1

# Layer 1: heavy dependency (torch CPU) — cached unless version changes
RUN pip install --no-cache-dir \
        torch==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Layer 2: project dependencies (cached unless pyproject.toml changes)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[hf,mlflow,notify]" --no-deps || \
    pip install --no-cache-dir transformer_lens einops fancy_einsum jaxtyping \
        huggingface_hub mlflow slack_sdk

# Layer 3: project source
COPY glassbox/ ./glassbox/
RUN pip install --no-cache-dir -e . --no-deps

# ---------------------------------------------------------------------------
# Stage 3 — api: FastAPI REST API (lean, no Gradio)
# ---------------------------------------------------------------------------
FROM builder AS api

COPY api/      ./api/
COPY scripts/  ./scripts/

# Health check — /health returns {"status": "ok", "version": "3.4.0"}
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER glassbox

EXPOSE 8000

# 2 workers by default; override with UVICORN_WORKERS env var for more CPUs
ENV UVICORN_WORKERS=2

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 \
     --workers ${UVICORN_WORKERS} --log-level info --access-log"]

# ---------------------------------------------------------------------------
# Stage 4 — dashboard: Gradio compliance dashboard
# ---------------------------------------------------------------------------
FROM builder AS dashboard

RUN pip install --no-cache-dir \
        gradio==4.44.0 \
        plotly==5.22.0

COPY dashboard/ ./dashboard/

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

USER glassbox

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
# Set to 1 to disable analytics pings in air-gapped environments
ENV GRADIO_ANALYTICS_ENABLED="False"

CMD ["python", "dashboard/app.py"]
