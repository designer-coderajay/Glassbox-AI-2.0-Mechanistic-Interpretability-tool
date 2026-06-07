# =============================================================================
# Glassbox AI — Standalone Compliance Scanner v4.3.0
# =============================================================================
#
# Single-command Annex IV report generation. No Python setup required.
#
# Usage:
#   docker build -f scan.Dockerfile -t glassbox-scan .
#
#   # Generate Annex IV PDF for any prompt — auto-selects corruption strategy
#   docker run --rm -v $(pwd)/output:/output \
#     glassbox-scan \
#     --model gpt2 \
#     --prompt "Loan application. Annual income: €42,000. Decision:" \
#     --correct " Approved" \
#     --incorrect " Denied" \
#     --purpose "Credit risk scoring for retail loans" \
#     --provider "Acme Bank NV" \
#     --output /output/annex_iv_report.pdf
#
#   # Large model with bfloat16 + checkpoint strategy
#   docker run --rm -v $(pwd)/output:/output \
#     -e HF_TOKEN=hf_... \
#     glassbox-scan \
#     --model meta-llama/Llama-3-8B \
#     --dtype bfloat16 \
#     --prompt "Patient presents with chest pain. Priority:" \
#     --correct " Urgent" --incorrect " Routine" \
#     --purpose "Medical triage prioritisation" \
#     --provider "Hospital NV" \
#     --output /output/annex_iv_medical.pdf
#
# Air-gapped deployment:
#   docker save glassbox-scan | gzip > glassbox-scan-4.3.0.tar.gz
#   # Transfer the .tar.gz to the air-gapped machine
#   docker load < glassbox-scan-4.3.0.tar.gz
# =============================================================================

FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Glassbox AI Compliance Scanner" \
      org.opencontainers.image.version="4.3.0" \
      org.opencontainers.image.description="Standalone EU AI Act Annex IV report generator" \
      org.opencontainers.image.licenses="MIT"

# System deps for PDF generation (WeasyPrint → pango, cairo)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b \
        libfontconfig1 libfreetype6 libjpeg62-turbo libpng16-16 \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (smaller image — no CUDA for compliance scanning)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install glassbox + compliance deps
RUN pip install --no-cache-dir \
    "glassbox-mech-interp[compliance]>=4.3.0" \
    transformer-lens \
    weasyprint \
    jinja2

# Copy scan entrypoint script
COPY scripts/docker_scan.py /app/docker_scan.py

# HuggingFace cache persists across runs when mounted
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache

# Default output directory
RUN mkdir -p /output

ENTRYPOINT ["python3", "/app/docker_scan.py"]
