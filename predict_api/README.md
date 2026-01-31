---
title: Cancer Predict API
emoji: ":microscope:"
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Lung Cancer Detection Backend (FastAPI + TensorFlow)

Production-ready containerized API serving a Vision Transformer Keras model for lung cancer image classification.

## Features

- FastAPI app with `/predict` endpoint (multipart file upload)
- Lightweight health & metadata endpoints: `/healthz`, `/`
- Batch & debugging endpoints under `/debug/*`
- Single global TensorFlow model loaded at startup
- Docker multi-stage build (slim Python base) + healthcheck
- docker-compose integration with frontend service

## API Summary

| Method | Path                          | Description                                             |
| ------ | ----------------------------- | ------------------------------------------------------- |
| GET    | `/`                           | API metadata                                            |
| GET    | `/healthz`                    | Liveness/health check                                   |
| POST   | `/predict`                    | Predict class for one image (UploadFile `file`)         |
| GET    | `/debug/sample-preds?limit=N` | Sample predictions from bundled training samples        |
| POST   | `/debug/logits`               | Raw logits + probabilities for an image                 |
| GET    | `/debug/analyze?limit=N`      | Aggregate per-class stats (needs training data present) |

### Example `curl`

```
curl -X POST http://localhost:7860/predict \
  -F "file=@sample.jpg"
```

Response:

```json
{
  "filename": "sample.jpg",
  "prediction": {
    "label": "Malignant cases",
    "confidence": 0.94,
    "probabilities": {
      "Bengin cases": 0.01,
      "Malignant cases": 0.94,
      "Normal cases": 0.05
    }
  }
}
```

## Environment Variables

| Name                   | Purpose                                    | Default                       |
| ---------------------- | ------------------------------------------ | ----------------------------- |
| `MODEL_FILE`           | Relative/absolute path to Keras model file | `lung_cancer_vit_model.keras` |
| `TZ`                   | Timezone inside container                  | `UTC`                         |
| `MODEL_AUTO_DOWNLOAD`  | Auto-download the model if missing (0/1)   | `1`                           |
| `MODEL_GDRIVE_FILE_ID` | Google Drive file id for the model         | (preconfigured)               |
| `MODEL_GDRIVE_URL`     | Google Drive URL for the model (optional)  | (derived from file id)        |

## Local (Non-Docker) Dev

```
python -m venv venv
source venv/Scripts/activate  # (Windows Git Bash) or venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs

## Docker Build & Run

```
# From this folder (predict_api)
docker build -t lung-cancer-api .

# Run
docker run --rm -p 7860:7860 --name lung-api lung-cancer-api
```

Health check:

```
curl http://localhost:7860/healthz
```

## docker-compose (Backend + Frontend)

```
docker compose up --build
```

Services:

- API: http://localhost:7860 (Hugging Face Docker Spaces requires port 7860)
- Frontend: http://localhost:3000 (calls API via internal DNS `http://api:8000`)

To rebuild after code changes:

```
docker compose build api && docker compose up -d api
```

## Deployment Guidance

## Deploy to Hugging Face Docker Space

This folder is already structured as a Docker Space (it contains `Dockerfile`, `requirements.txt`, and `app/`).

1) Create a Hugging Face Space with SDK = `Docker`.

2) Clone the Space repo and copy this folder's contents to the repo root (important: the Space expects `Dockerfile` at the repo root):

```
git clone https://huggingface.co/spaces/<your-username>/<your-space>
# Copy the contents of predict_api/ into the cloned repo root
```

3) Commit + push:

```
git add .
git commit -m "Deploy FastAPI predict API"
git push
```

4) Once built, the API will be available at your Space URL.

Notes:
- The container listens on port `7860` (required by Hugging Face).
- If `lung_cancer_vit_model.keras` is not present in the repo, the app will auto-download it at runtime (see `MODEL_AUTO_DOWNLOAD`, `MODEL_GDRIVE_*`).

### Image Tagging Strategy

Use semantic or git-based tags:

```
docker build -f trained-model/Dockerfile -t ghcr.io/your-org/lung-cancer-api:1.0.0 .
```

Push to registry (GHCR example):

```
echo $GH_PAT | docker login ghcr.io -u USERNAME --password-stdin
docker push ghcr.io/your-org/lung-cancer-api:1.0.0
```

### Production Runtime Suggestions

- Set `UVICORN_WORKERS` >1 only if memory permits (TensorFlow copies model per process)
- Allocate adequate memory (TensorFlow footprint) ~1-2GB baseline
- Use a reverse proxy (NGINX / API Gateway) for TLS & rate limiting
- Add request body size limits if exposing publicly
- Enable structured logging (can integrate `uvicorn --log-config` later)
- For GPU inference create a separate `Dockerfile.gpu` (nvidia/cuda base + `pip install tensorflow[and-cuda]`)

### Scaling

Because TensorFlow loads per process, prefer horizontal pod autoscaling (Kubernetes) with 1 worker per pod. Ensure readinessProbe hits `/healthz`.

Kubernetes snippet:

```yaml
livenessProbe:
  httpGet: { path: /healthz, port: 8000 }
  initialDelaySeconds: 30
  periodSeconds: 30
readinessProbe:
  httpGet: { path: /healthz, port: 8000 }
  initialDelaySeconds: 30
```

### Logging

Current configuration uses default Uvicorn logging. Consider adding a JSON logger and correlation IDs for production.

## Security Notes

- Remove unused debug endpoints before public exposure (or guard behind auth)
- Validate image MIME types & size (future enhancement)
- Keep dependencies updated; use Dependabot / Renovate

## Development Roadmap (Suggested)

- Add unit tests for preprocessing & prediction mapping
- Introduce Pydantic response models
- Add rate limiting / auth
- Provide GPU-enabled Docker variant
- CI pipeline (build + vulnerability scan + push)

## Troubleshooting

| Symptom                  | Possible Cause     | Fix                                              |
| ------------------------ | ------------------ | ------------------------------------------------ |
| 404 on `/predict`        | Wrong URL          | Use POST `/predict` with form-data file          |
| High memory              | Multiple workers   | Reduce `UVICORN_WORKERS` to 1                    |
| Slow first request       | Model cold load    | Warm-up by calling `/predict` at container start |
| 500 Model file not found | `MODEL_FILE` wrong | Ensure file exists in `/app/app/`                |

## Warm-Up Script (Optional)

Add to compose command if desired:

```
/bin/sh -c "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
  pid=$!; sleep 10; curl -s -o /dev/null -F file=@app/vit_lung_final.keras http://localhost:8000/healthz; wait $pid"
```

---

Maintained with best-practice lightweight containerization for ML inference.
