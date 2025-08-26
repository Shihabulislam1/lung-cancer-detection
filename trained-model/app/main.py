from fastapi import FastAPI, UploadFile, File
import os, glob
from PIL import Image
import io
from app.model_loader import predict, batch_predict, predict_with_logits, analyze_directory

app = FastAPI(title="Lung Cancer Detection API", version="1.0")

@app.get("/", tags=["meta"])
def root():
    return {"name": app.title, "version": app.version, "docs": "/docs"}

@app.get("/healthz", tags=["health"])
def healthz():
    # Lightweight health endpoint (does not touch the model to stay fast)
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict(image)
    return {"filename": file.filename, "prediction": result}


@app.get("/debug/sample-preds")
def debug_sample_preds(limit: int = 3):
    base = os.path.join(os.path.dirname(__file__), "training_data")
    classes = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    samples = []
    for cls in classes:
        pattern = os.path.join(base, cls, "*.jpg")
        files = sorted(glob.glob(pattern))[:limit]
        samples.extend(files)
    return {
        "classes": classes,
        "samples": batch_predict(samples)
    }


@app.post("/debug/logits")
async def debug_logits(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return predict_with_logits(image)


@app.get("/debug/analyze")
def debug_analyze(limit: int = 5):
    base = os.path.join(os.path.dirname(__file__), "training_data")
    return analyze_directory(base, limit_per_class=limit)
