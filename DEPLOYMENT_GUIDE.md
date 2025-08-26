# Deployment & Operations Guide

This guide covers container build, local orchestration, production deployment patterns, and operational best practices for the Lung Cancer Detection platform (FastAPI backend + Next.js frontend).

## 1. Components

| Component   | Path                                     | Purpose                             |
| ----------- | ---------------------------------------- | ----------------------------------- |
| Backend API | `trained-model/app`                      | ML inference (TensorFlow + FastAPI) |
| Frontend    | `frontend/`                              | Web UI consuming API                |
| Model File  | `trained-model/app/vit_lung_final.keras` | Saved Vision Transformer            |

## 2. Quick Start (Local)

```
# Build & start (from repo root)
docker compose up --build

# Tear down
docker compose down
```

Services:

- API: http://localhost:8000 (Swagger UI at /docs)
- Frontend: http://localhost:3000

## 3. Directory Layout Highlights

- `trained-model/Dockerfile` – Production Python slim image; healthcheck
- `.dockerignore` – Excludes large/unneeded dev artifacts
- `docker-compose.yml` – Orchestrates api + frontend network

## 4. Configuration

| Variable   | Service  | Description                     | Default                 |
| ---------- | -------- | ------------------------------- | ----------------------- |
| MODEL_FILE | api      | Model filename in `app/`        | vit_lung_final.keras    |
| ML_API_URL | frontend | Endpoint to call for prediction | http://api:8000/predict |
| TZ         | api      | Container timezone              | UTC                     |

Add secrets (auth keys, Cloudinary, etc.) via `.env` (compose supports `env_file:`) – do NOT commit.

Example `.env`:

```
NEXTAUTH_SECRET=change-me
CLOUDINARY_API_SECRET=change-me
```

Then in compose:

```
    env_file:
      - ./frontend/.env
```

## 5. Image Build & Tagging

```
# Backend
docker build -f trained-model/Dockerfile -t ghcr.io/your-org/lung-cancer-api:$(git rev-parse --short HEAD) .

# Frontend
docker build -f frontend/Dockerfile -t ghcr.io/your-org/lung-cancer-frontend:$(git rev-parse --short HEAD) ./frontend
```

Push images after registry login.

## 6. Production Deployment Options

### Option A: Single VM / Docker Host

1. Install Docker + (optional) docker compose plugin.
2. Copy repo or deploy using remote `docker compose pull && docker compose up -d`.
3. Add reverse proxy (NGINX / Caddy) terminating TLS, proxying to:

```
/api -> http://localhost:8000
/    -> http://localhost:3000
```

### Option B: Kubernetes

Create separate Deployments:

- `api-deployment` (1 pod; scale horizontally if needed)
- `frontend-deployment`
  Use ClusterIP Services: `api` and `frontend`. Ingress routes `/` to frontend and `/predict` + other API paths to backend (or mount at `/api`). Set resource requests (e.g. api: 500m CPU, 1Gi memory).

Readiness / Liveness probes:

```
httpGet: { path: /healthz, port: 8000 }
```

### Option C: Serverless Containers (Cloud Run, ECS Fargate)

- Build & push image
- Deploy with 1 CPU / 2GB RAM (adjust after observing latency)
- Configure min instances to avoid cold starts if needed

## 7. Observability

| Aspect  | Recommendation                                              |
| ------- | ----------------------------------------------------------- |
| Logging | Use stdout; aggregate via Cloud logging / ELK / Loki        |
| Metrics | Add Prometheus endpoint (future) or integrate OpenTelemetry |
| Tracing | Optional: instrument FastAPI with OTEL SDK                  |
| Health  | `/healthz` endpoint already provided                        |

## 8. Security Hardening

- Remove or restrict `/debug/*` routes in production
- Enforce content size limit (e.g. reverse proxy `client_max_body_size 5M;`)
- Add MIME/type validation for uploads
- Run containers as non-root (already implemented)
- Scan images (Trivy / Grype) in CI

## 9. Performance Tuning

| Lever           | Effect                                              |
| --------------- | --------------------------------------------------- |
| IMAGE_SIZE      | Preprocessing cost; fixed to trained size           |
| UVICORN_WORKERS | Concurrency; each worker duplicates model RAM usage |
| Model format    | Consider TF SavedModel or TFLite for speed (future) |
| Warm-up         | Hit `/predict` once at startup to JIT kernels       |

Batching: Current design processes single images; add queued micro-batcher if throughput becomes an issue.

## 10. Backup & Recovery

- Model artifact is versioned in source; consider storing in object storage (S3/GCS) and pass URL or mount volume
- Keep infrastructure as code (Helm/compose) under version control

## 11. Updating the Model

1. Train & export new `vit_lung_final.keras` (or versioned name)
2. Update `MODEL_FILE` or replace file
3. Rebuild image & push
4. Deploy using rolling update (compose: `up -d --no-deps api` / K8s rolling deployment)
5. Monitor error rates & latency

## 12. Disaster Scenarios

| Scenario           | Mitigation                                                           |
| ------------------ | -------------------------------------------------------------------- |
| Corrupt model file | Versioned artifact store; fallback last good tag                     |
| Memory exhaustion  | Lower workers; add limits/requests; profile; possibly quantize model |
| Slow predictions   | Profile GPU vs CPU; optimize model; enable XLA (experimental)        |

## 13. Future Enhancements

- GPU-enabled Dockerfile variant
- Model registry (MLflow) integration
- Automated canary release (10% traffic) before full rollout
- Add authentication (API key / JWT) to predict endpoint

## 14. Local Development Cycle

```
# Edit code, then rebuild just backend
docker compose build api
# Restart container
docker compose up -d api
# View logs
docker compose logs -f api
```

## 15. Testing (Suggested)

- Add pytest suite: mock PIL image, assert probability vector length
- Add contract tests hitting `/predict` with sample image

---

Use this document as the single source of truth for operating the platform.
