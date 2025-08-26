# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile for FastAPI + TensorFlow inference

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_WORKERS=1

# System deps (libgl for Pillow / image ops, curl for health debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement file separately for layer caching
COPY trained-model/requirements.txt /tmp/requirements.txt

# (Optional) Pin tensorflow to CPU build already pinned in requirements
# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code (only what is needed thanks to .dockerignore)
COPY trained-model/app ./app

# Expose port
EXPOSE 8000

# Create a non-root user
RUN useradd -u 1001 -m appuser && chown -R appuser /app
USER appuser

# Healthcheck (container-level)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
