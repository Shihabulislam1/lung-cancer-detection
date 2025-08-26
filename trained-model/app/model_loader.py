# app/model_loader.py
import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers
import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_PATH = os.getenv("MODEL_FILE", "vit_lung_final.keras")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "labels.json")
IMAGE_SIZE = 256

# --- Custom layers ---
@register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
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

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


@register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units
        })
        return config


def _resolve_model_path(path: str) -> str:
    """Resolve model path trying current CWD first then module directory."""
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Model file not found at '{path}' or '{alt}'")


# --- Load model & labels at import time (single global instance) ---
_model_path = _resolve_model_path(MODEL_PATH)
model = keras.models.load_model(
    _model_path,
    custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder}
)

try:
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        CLASS_LABELS: List[str] = json.load(f)
    # Basic validation
    if not isinstance(CLASS_LABELS, list) or not all(isinstance(x, str) for x in CLASS_LABELS):
        raise ValueError("labels.json must contain a JSON string list")
except Exception as e:  # noqa: BLE001
    # Fallback: attempt to infer from training data directory order
    train_dir = os.path.join(os.path.dirname(__file__), "training_data")
    if os.path.isdir(train_dir):
        CLASS_LABELS = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    else:
        CLASS_LABELS = []
    print(f"[WARN] Could not load labels.json ({e}). Using inferred labels: {CLASS_LABELS}")

# --- Training dataset folder for reference ---
TRAIN_DATA_DIR = "training_data"  # <- change this to your training data root folder

def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize PIL image to model expected tensor.

    IMPORTANT: The saved model already contains a Rescaling(1./255) layer
    inside its data_augmentation pipeline. So we DO NOT divide by 255 here
    to avoid double scaling, which would shrink inputs and harm accuracy.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(image).astype("float32")  # raw 0-255; internal model rescales
    return np.expand_dims(arr, axis=0)  # add batch dim


def predict(image: Image.Image):
    """Run prediction on a PIL Image and return best label & probabilities.

    Returns:
        dict with top_label, confidence, probabilities (label->prob)
        or a fallback string if labels unavailable.
    """
    input_tensor = _preprocess(image)
    preds = model.predict(input_tensor)
    # If model returns logits, apply softmax
    if preds.ndim == 2:
        probs = tf.nn.softmax(preds[0]).numpy()
    else:  # unexpected shape
        probs = np.squeeze(preds)
        if probs.ndim != 1:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")
    top_idx = int(np.argmax(probs))
    if CLASS_LABELS and top_idx < len(CLASS_LABELS):
        top_label = CLASS_LABELS[top_idx]
        prob_map = {CLASS_LABELS[i]: float(p) for i, p in enumerate(probs[: len(CLASS_LABELS)])}
        return {
            "label": top_label,
            "confidence": float(probs[top_idx]),
            "probabilities": prob_map,
        }
    # Fallback: just return index & raw probabilities
    return {"label_index": top_idx, "confidence": float(probs[top_idx])}


def predict_with_logits(image: Image.Image) -> Dict[str, Any]:
    """Return logits along with probabilities for debugging."""
    input_tensor = _preprocess(image)
    logits = model(input_tensor, training=False).numpy()[0]
    probs = tf.nn.softmax(logits).numpy()
    data: Dict[str, Any] = {
        "logits": logits.tolist(),
        "probs": probs.tolist(),
    }
    if CLASS_LABELS:
        data["mapped"] = {CLASS_LABELS[i]: float(probs[i]) for i in range(min(len(CLASS_LABELS), len(probs)))}
        data["pred_label"] = CLASS_LABELS[int(np.argmax(probs))]
    else:
        data["pred_index"] = int(np.argmax(probs))
    return data


def analyze_directory(base_dir: str, limit_per_class: int = 10) -> Dict[str, Any]:
    """Compute mean logits & probs per class directory."""
    summary: Dict[str, Any] = {"base_dir": base_dir, "classes": []}
    if not os.path.isdir(base_dir):
        summary["error"] = "base_dir not found"
        return summary
    for cls in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, cls)
        if not os.path.isdir(p):
            continue
        imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))][:limit_per_class]
        logits_list = []
        pred_labels = []
        for img_path in imgs:
            try:
                with Image.open(img_path) as im:
                    input_tensor = _preprocess(im)
                    logits = model(input_tensor, training=False).numpy()[0]
                    logits_list.append(logits)
                    pred_labels.append(int(np.argmax(logits)))
            except Exception as e:  # noqa: BLE001
                pred_labels.append(f"error:{e}")
        if logits_list:
            arr_logits = np.vstack(logits_list)
            mean_logits = arr_logits.mean(axis=0)
            mean_probs = tf.nn.softmax(mean_logits).numpy()
        else:
            mean_logits = []
            mean_probs = []
        class_entry: Dict[str, Any] = {
            "class_dir": cls,
            "num_samples": len(logits_list),
            "pred_label_counts": {str(k): pred_labels.count(k) for k in set(pred_labels)},
            "mean_logits": mean_logits.tolist() if len(mean_logits) else [],
            "mean_probs": mean_probs.tolist() if len(mean_probs) else [],
        }
        if CLASS_LABELS and len(mean_probs) == len(CLASS_LABELS):
            class_entry["mean_probs_mapped"] = {CLASS_LABELS[i]: float(mean_probs[i]) for i in range(len(CLASS_LABELS))}
        summary["classes"].append(class_entry)
    summary["label_order"] = CLASS_LABELS
    return summary


def batch_predict(image_paths):
    """Run predictions for a list of image file paths.
    Returns list of dicts: {path, prediction}
    """
    out = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                out.append({"path": os.path.basename(p), "prediction": predict(img)})
        except Exception as e:  # noqa: BLE001
            out.append({"path": os.path.basename(p), "error": str(e)})
    return out


# --- FastAPI app (kept separate; main.py creates the public API) ---
app = FastAPI(title="Lung Cancer ViT Predictor (Internal)")

@app.on_event("startup")
def startup_event():
    # Print model expected class indices
    classes = sorted([d for d in os.listdir(TRAIN_DATA_DIR) 
                      if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))])
    print("\n[INFO] Model expected classes (from training folder structure):")
    for idx, cls in enumerate(classes):
        print(f"  {idx}: {cls}")
    print("[INFO] Server started successfully.\n")
