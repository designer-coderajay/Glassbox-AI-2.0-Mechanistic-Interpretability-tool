FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — layered for cache efficiency
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source
COPY glassbox/ ./glassbox/
COPY api/       ./api/
COPY dashboard/ ./dashboard/
COPY pyproject.toml README.md ./

# Install package itself
RUN pip install --no-cache-dir -e .

# Non-root user
RUN adduser --disabled-password --gecos '' glassbox
USER glassbox

# Port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
