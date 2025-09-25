# app/model_loader.py
"""Model loading & prediction utilities for Lung Cancer ViT model.

Key adjustments vs original version:
 - Lazy model loading (first prediction) to reduce import latency.
 - Preprocessing now scales pixels to 0-1 to match notebook training
     (the training notebook divided by 255 BEFORE Normalization/adapt).
 - Simpler public surface (predict / batch_predict) with consistent
     output schema and graceful fallbacks.
 - Removal of embedded FastAPI app (keep one app in main.py only).
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Iterable
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_FILENAME = "lung_cancer_vit_model.keras"  # matches file in same dir as main.py
MODEL_PATH = os.getenv("MODEL_FILE", DEFAULT_MODEL_FILENAME)
LABELS_FILE = os.path.join(os.path.dirname(__file__), "labels.json")
IMAGE_SIZE = 256
TRAIN_DATA_DIR = "training_data"  # optional (used only for label inference)

# ---------------------------------------------------------------------------
# Custom layers (needed for deserialization)
# ---------------------------------------------------------------------------

# --- Custom layers ---
@register_keras_serializable()
class Patches(layers.Layer):
    """Extract non‑overlapping patches (ViT style)."""

    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):  # type: ignore[override]
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):  # type: ignore[override]
        return {**super().get_config(), "patch_size": self.patch_size}


@register_keras_serializable()
class PatchEncoder(layers.Layer):
    """Linear projection + learnable positional embedding."""

    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):  # type: ignore[override]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):  # type: ignore[override]
        return {
            **super().get_config(),
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units,
        }


def _resolve_model_path(path: str) -> str:
    """Resolve model path: try absolute/relative, then same directory as this file."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # try relative to CWD
    if os.path.exists(path):
        return path
    # try next to this file
    alt = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Model file not found (tried '{path}' and '{alt}')")

_MODEL = None  # lazy singleton
CLASS_LABELS: List[str] = []


def _load_labels() -> List[str]:
    # Explicit labels.json
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise ValueError("labels.json must contain a list[str]")
        return labels
    except Exception as e:  # noqa: BLE001
        # Fallback: folder names inside training data dir
        td = os.path.join(os.path.dirname(__file__), TRAIN_DATA_DIR)
        if os.path.isdir(td):
            labels = sorted([d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))])
            if labels:
                print(f"[WARN] Using inferred labels (labels.json issue: {e}) -> {labels}")
            return labels
    return []


def get_class_labels() -> List[str]:
    global CLASS_LABELS
    if not CLASS_LABELS:
        CLASS_LABELS = _load_labels()
    return CLASS_LABELS


def get_model():
    """Return (and lazily load) the Keras model."""
    global _MODEL
    if _MODEL is None:
        path = _resolve_model_path(MODEL_PATH)
        _MODEL = keras.models.load_model(
            path,
            custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder},
        )
        get_class_labels()  # ensure labels loaded
        print(f"[INFO] Loaded model from: {path}")
    return _MODEL

def _preprocess(image: Image.Image) -> np.ndarray:
    """Convert PIL image -> float32 tensor (1, H, W, 3) scaled 0-1.

    Notebook training divided by 255 BEFORE Normalization(). Therefore we
    must replicate that here; the saved model (per notebook) does NOT have
    an additional Rescaling layer.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image) -> Dict[str, Any]:
    """Run prediction on a single PIL image.

    Returns a dict with keys:
      - label: str (if labels known)
      - confidence: float
      - probabilities: {label: prob} OR {index: prob}
    """
    mdl = get_model()
    labels = get_class_labels()
    tensor = _preprocess(image)
    logits = mdl.predict(tensor, verbose=0)
    if logits.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {logits.shape}")
    probs = tf.nn.softmax(logits[0]).numpy()
    top_idx = int(np.argmax(probs))
    if labels and len(labels) == probs.shape[0]:
        prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
        return {
            "label": labels[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": prob_map,
        }
    # fallback (no label names)
    return {
        "label_index": top_idx,
        "confidence": float(probs[top_idx]),
        "probabilities": {str(i): float(p) for i, p in enumerate(probs)},
    }


def predict_with_logits(image: Image.Image) -> Dict[str, Any]:
    """Return logits + probs (useful for debugging)."""
    mdl = get_model()
    labels = get_class_labels()
    tensor = _preprocess(image)
    logits = mdl(tensor, training=False).numpy()[0]
    probs = tf.nn.softmax(logits).numpy()
    out: Dict[str, Any] = {
        "logits": logits.tolist(),
        "probs": probs.tolist(),
        "pred_index": int(np.argmax(probs)),
    }
    if labels and len(labels) == probs.shape[0]:
        out["pred_label"] = labels[out["pred_index"]]
        out["mapped"] = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return out


def analyze_directory(base_dir: str, limit_per_class: int = 10) -> Dict[str, Any]:
    """Lightweight aggregate stats over a directory tree (optional helper)."""
    mdl = get_model()
    labels = get_class_labels()
    summary: Dict[str, Any] = {"base_dir": base_dir, "classes": []}
    if not os.path.isdir(base_dir):
        summary["error"] = "base_dir not found"
        return summary
    for cls in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, cls)
        if not os.path.isdir(p):
            continue
        imgs = [
            os.path.join(p, f)
            for f in os.listdir(p)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:limit_per_class]
        logits_accum = []
        for path in imgs:
            try:
                with Image.open(path) as im:
                    t = _preprocess(im)
                    logits_accum.append(mdl(t, training=False).numpy()[0])
            except Exception:  # noqa: BLE001
                pass
        if logits_accum:
            arr = np.vstack(logits_accum)
            mean_logits = arr.mean(axis=0)
            mean_probs = tf.nn.softmax(mean_logits).numpy()
            entry: Dict[str, Any] = {
                "class_dir": cls,
                "num_samples": len(logits_accum),
                "mean_logits": mean_logits.tolist(),
                "mean_probs": mean_probs.tolist(),
            }
            if labels and len(labels) == len(mean_probs):
                entry["mean_probs_mapped"] = {labels[i]: float(mean_probs[i]) for i in range(len(labels))}
        else:
            entry = {"class_dir": cls, "num_samples": 0, "mean_logits": [], "mean_probs": []}
        summary["classes"].append(entry)
    summary["label_order"] = labels
    return summary


def batch_predict(image_paths: Iterable[str]) -> List[Dict[str, Any]]:
    """Predict multiple image file paths."""
    results: List[Dict[str, Any]] = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                results.append({"path": os.path.basename(p), "prediction": predict(img)})
        except Exception as e:  # noqa: BLE001
            results.append({"path": os.path.basename(p), "error": str(e)})
    return results
__all__ = [
    "predict",
    "predict_with_logits",
    "batch_predict",
    "analyze_directory",
    "get_model",
    "get_class_labels",
]

