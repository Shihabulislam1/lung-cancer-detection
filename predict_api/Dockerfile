# syntax=docker/dockerfile:1.6

# Lightweight Python base; TensorFlow has prebuilt wheels for this.
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code (including model + labels)
COPY app/ ./app/

EXPOSE 8000

# Default model path is set in code; can override with MODEL_FILE
ENV MODEL_FILE=lung_cancer_vit_model.keras

# Uvicorn server
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" ]
