# syntax=docker/dockerfile:1.6

# Hugging Face Docker Spaces requirement: listen on port 7860.
# Use slim base; TensorFlow has prebuilt wheels for Linux x86_64.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# System deps (curl used by health checks / debugging)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (recommended by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python deps first for better layer caching
COPY --chown=user requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user app/ ./app/

# Default model path is set in code; can override with MODEL_FILE.
# If the file is missing, the app can auto-download it via gdown.
ENV MODEL_FILE=lung_cancer_vit_model.keras

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
