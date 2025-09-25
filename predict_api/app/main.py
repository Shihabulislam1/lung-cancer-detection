"""FastAPI application exposing prediction endpoints for the Lung Cancer ViT.

Simplified surface:
  GET  /          -> meta info
  GET  /healthz   -> lightweight health (no model load)
  POST /predict   -> single image prediction (multipart/form-data)
  POST /predict/logits -> single image prediction with logits (debug)
  GET  /labels    -> list of class labels (loads model if needed)
  GET  /debug/analyze?limit=N -> (optional) aggregate stats over training_data

All responses follow a consistent envelope: { "ok": bool, ... } on success.
Errors return { "ok": false, "error": "message" } with appropriate status.
"""

from __future__ import annotations

import io
import os
import glob
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from app.model_loader import (
    predict,
    predict_with_logits,
    batch_predict,
    analyze_directory,
    get_class_labels,
    get_model,
)

app = FastAPI(title="Lung Cancer Detection API", version="1.1.0")

# ---------------------------------------------------------------------------
# CORS (allow local frontend & configurable origins)
# ---------------------------------------------------------------------------
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root():
    return {"ok": True, "name": app.title, "version": app.version, "docs": "/docs"}


@app.get("/healthz", tags=["health"])
def healthz():
    return {"ok": True, "status": "ready"}


def _load_image(upload: UploadFile) -> Image.Image:
    try:
        contents = upload.file.read()
        if not contents:
            raise ValueError("empty file")
        img = Image.open(io.BytesIO(contents))
        img.verify()  # quick integrity check
        img = Image.open(io.BytesIO(contents))  # reopen after verify resets fp
        return img
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.post("/predict", tags=["inference"])
async def predict_image(file: UploadFile = File(...)):
    image = _load_image(file)
    result = predict(image)
    return {"ok": True, "filename": file.filename, "prediction": result}


@app.post("/predict/logits", tags=["debug"])
async def predict_image_logits(file: UploadFile = File(...)):
    image = _load_image(file)
    result = predict_with_logits(image)
    return {"ok": True, "filename": file.filename, "debug": result}


@app.get("/labels", tags=["meta"])
def list_labels():
    # ensure model (and thus labels) loaded
    get_model()
    labels = get_class_labels()
    return {"ok": True, "labels": labels, "count": len(labels)}


@app.get("/debug/sample-preds", tags=["debug"])
def debug_sample_preds(limit: int = 3):
    base = os.path.join(os.path.dirname(__file__), "training_data")
    if not os.path.isdir(base):
        raise HTTPException(status_code=404, detail="training_data directory not found")
    classes = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    samples = []
    for cls in classes:
        pattern = os.path.join(base, cls, "*.jpg")
        files = sorted(glob.glob(pattern))[:limit]
        samples.extend(files)
    return {"ok": True, "classes": classes, "samples": batch_predict(samples)}


@app.get("/debug/analyze", tags=["debug"])
def debug_analyze(limit: int = 5):
    base = os.path.join(os.path.dirname(__file__), "training_data")
    return {"ok": True, **analyze_directory(base, limit_per_class=limit)}

