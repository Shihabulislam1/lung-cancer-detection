
#Training code in kaggle
```py
# Kaggle-ready ViT training on IQ-OTH/NCCD (3-class: Normal, Benign/Bengin, Malignant)

import os, sys, pathlib, itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TF:", tf.__version__, "Python:", sys.version)

# 0) Paths — replace with the exact slug you added via 'Add input' (check the right pane)
# Examples of common slugs; uncomment the one you actually added, or paste the path string.
# DATA_ROOT = "/kaggle/input/iqothnccd-lung-cancer-dataset"
# DATA_ROOT = "/kaggle/input/the-iq-othnccd-lung-cancer-dataset"
# DATA_ROOT = "/kaggle/input/iq-othnccd-lung-cancer-augmented-dataset"
DATA_ROOT = "/kaggle/input/the-iq-othnccd-lung-cancer-dataset"  # <-- change if needed

# If your dataset is a ZIP INSIDE the input, unzip it once to /kaggle/working and point DATA_ROOT there:
# !unzip -q "/kaggle/input/your-dataset/archive.zip" -d /kaggle/working
# DATA_ROOT = "/kaggle/working/path/to/unzipped/folder"

# 1) Training knobs
IMAGE_SIZE   = 256
PATCH_SIZE   = 16
NUM_PATCHES  = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJ_DIM     = 64
NUM_HEADS    = 6
TRANSFORMER_UNITS = [PROJ_DIM * 2, PROJ_DIM]
TRANSFORMER_LAYERS = 8
MLP_HEAD_UNITS = [2048, 1024]
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42
AUTOTUNE     = tf.data.AUTOTUNE

# 2) Build datasets from folders automatically (handles 'Bengin' vs 'Benign')
def find_data_root(root):
    # If the dataset has an extra nesting level, descend until we find class dirs.
    p = pathlib.Path(root)
    # Heuristic: find directory that contains any of these class dirs
    class_variants = [
        {"Normal cases", "Benign cases", "Malignant cases"},
        {"Normal cases", "Bengin cases", "Malignant cases"},
        {"Normal", "Benign", "Malignant"},
    ]
    candidates = [p] + [d for d in p.rglob("*") if d.is_dir() and d.parent == p]
    for cand in candidates:
        subdirs = {d.name for d in cand.iterdir() if d.is_dir()}
        for target in class_variants:
            if target.issubset(subdirs):
                return str(cand)
    return str(p)

DATA_DIR = find_data_root(DATA_ROOT)
print("Using data dir:", DATA_DIR)

# If the dataset also has a separate 'Test cases' folder, we’ll use it as test_ds.
test_cases_dir = None
for nm in ["Test cases", "test", "Test"]:
    path = pathlib.Path(DATA_DIR).parent / nm
    if path.exists():
        test_cases_dir = str(path)
        print("Found test cases at:", test_cases_dir)
        break

def make_ds(directory, subset=None, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        validation_split=0.2 if subset else None,
        subset=subset,                # 'training' or 'validation' or None
        seed=SEED,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb"              # ensures 3 channels even if source is grayscale
    )

if test_cases_dir is None:
    train_ds = make_ds(DATA_DIR, subset="training")
    val_ds   = make_ds(DATA_DIR, subset="validation")
    test_ds  = val_ds
else:
    train_ds = make_ds(DATA_DIR, subset="training")
    val_ds   = make_ds(DATA_DIR, subset="validation")
    test_ds  = tf.keras.utils.image_dataset_from_directory(
        test_cases_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb"
    )

NUM_CLASSES = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# Prefetch for performance
def prepare(ds, training=False):
    if training:
        ds = ds.shuffle(1024, seed=SEED)
    return ds.prefetch(AUTOTUNE)

train_ds = prepare(train_ds, training=True)
val_ds   = prepare(val_ds)
test_ds  = prepare(test_ds)

# 3) Data augmentation (use Rescaling instead of Normalization+adapt)
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.2, 0.2),
], name="data_augmentation")

# 4) ViT building blocks (fixed __init__ typos)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
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
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

def create_vit_classifier():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    patches = Patches(PATCH_SIZE)(x)
    encoded = PatchEncoder(NUM_PATCHES, PROJ_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJ_DIM, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attn, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        encoded = layers.Add()([x3, x2])

    # Head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = layers.Flatten()(representation)     # you can swap to GlobalAveragePooling1D()
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    logits = layers.Dense(NUM_CLASSES)(features)
    return keras.Model(inputs, logits, name="vit_lung")

model = create_vit_classifier()
model.summary()

# 5) Train
optimizer = tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
    ],
)

ckpt_path = "/kaggle/working/vit_lung_best.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# 6) Evaluate & save
test_metrics = model.evaluate(test_ds, return_dict=True)
print("Test:", test_metrics)
model.save("/kaggle/working/vit_lung_final.keras")

# 7) Plots
import matplotlib.pyplot as plt
plt.figure(); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"]); plt.title("accuracy"); plt.legend(["train","val"]); plt.xlabel("epoch"); plt.ylabel("acc"); plt.show()
plt.figure(); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"]); plt.title("loss"); plt.legend(["train","val"]); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()

```


#Training log 
```
TF: 2.18.0 Python: 3.11.13 (main, Jun  4 2025, 08:57:29) [GCC 11.4.0]
Using data dir: /kaggle/input/the-iq-othnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset
Found 1097 files belonging to 3 classes.
Using 878 files for training.
I0000 00:00:1756214195.360697      36 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13942 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1756214195.361396      36 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13942 MB memory:  -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5
Found 1097 files belonging to 3 classes.
Using 219 files for validation.
Classes: ['Bengin cases', 'Malignant cases', 'Normal cases']
Model: "vit_lung"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 256, 256,  │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ data_augmentation   │ (None, 256, 256,  │          0 │ input_layer[0][0] │
│ (Sequential)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ patches (Patches)   │ (None, None, 768) │          0 │ data_augmentatio… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ patch_encoder       │ (None, 256, 64)   │     65,600 │ patches[0][0]     │
│ (PatchEncoder)      │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalization │ (None, 256, 64)   │        128 │ patch_encoder[0]… │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (Add)           │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ patch_encoder[0]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add[0][0]         │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_1 (Dropout) │ (None, 256, 128)  │          0 │ dense_1[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_2 (Dense)     │ (None, 256, 64)   │      8,256 │ dropout_1[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_2 (Dropout) │ (None, 256, 64)   │          0 │ dense_2[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (Add)         │ (None, 256, 64)   │          0 │ dropout_2[0][0],  │
│                     │                   │            │ add[0][0]         │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_1[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_2 (Add)         │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ add_1[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_2[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_3 (Dense)     │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_4 (Dropout) │ (None, 256, 128)  │          0 │ dense_3[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_4 (Dense)     │ (None, 256, 64)   │      8,256 │ dropout_4[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_5 (Dropout) │ (None, 256, 64)   │          0 │ dense_4[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_3 (Add)         │ (None, 256, 64)   │          0 │ dropout_5[0][0],  │
│                     │                   │            │ add_2[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_3[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_4 (Add)         │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ add_3[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_4[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_5 (Dense)     │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_7 (Dropout) │ (None, 256, 128)  │          0 │ dense_5[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_6 (Dense)     │ (None, 256, 64)   │      8,256 │ dropout_7[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_8 (Dropout) │ (None, 256, 64)   │          0 │ dense_6[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_5 (Add)         │ (None, 256, 64)   │          0 │ dropout_8[0][0],  │
│                     │                   │            │ add_4[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_5[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_6 (Add)         │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ add_5[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_6[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_7 (Dense)     │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_10          │ (None, 256, 128)  │          0 │ dense_7[0][0]     │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_8 (Dense)     │ (None, 256, 64)   │      8,256 │ dropout_10[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_11          │ (None, 256, 64)   │          0 │ dense_8[0][0]     │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_7 (Add)         │ (None, 256, 64)   │          0 │ dropout_11[0][0], │
│                     │                   │            │ add_6[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_7[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_8 (Add)         │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ add_7[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_8[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_9 (Dense)     │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_13          │ (None, 256, 128)  │          0 │ dense_9[0][0]     │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_10 (Dense)    │ (None, 256, 64)   │      8,256 │ dropout_13[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_14          │ (None, 256, 64)   │          0 │ dense_10[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_9 (Add)         │ (None, 256, 64)   │          0 │ dropout_14[0][0], │
│                     │                   │            │ add_8[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_9[0][0]       │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_10 (Add)        │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ add_9[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_10[0][0]      │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_11 (Dense)    │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_16          │ (None, 256, 128)  │          0 │ dense_11[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_12 (Dense)    │ (None, 256, 64)   │      8,256 │ dropout_16[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_17          │ (None, 256, 64)   │          0 │ dense_12[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_11 (Add)        │ (None, 256, 64)   │          0 │ dropout_17[0][0], │
│                     │                   │            │ add_10[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_11[0][0]      │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_12 (Add)        │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ add_11[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_12[0][0]      │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_13 (Dense)    │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_19          │ (None, 256, 128)  │          0 │ dense_13[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_14 (Dense)    │ (None, 256, 64)   │      8,256 │ dropout_19[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_20          │ (None, 256, 64)   │          0 │ dense_14[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_13 (Add)        │ (None, 256, 64)   │          0 │ dropout_20[0][0], │
│                     │                   │            │ add_12[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_13[0][0]      │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 256, 64)   │     99,520 │ layer_normalizat… │
│ (MultiHeadAttentio… │                   │            │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_14 (Add)        │ (None, 256, 64)   │          0 │ multi_head_atten… │
│                     │                   │            │ add_13[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_14[0][0]      │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_15 (Dense)    │ (None, 256, 128)  │      8,320 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_22          │ (None, 256, 128)  │          0 │ dense_15[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_16 (Dense)    │ (None, 256, 64)   │      8,256 │ dropout_22[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_23          │ (None, 256, 64)   │          0 │ dense_16[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_15 (Add)        │ (None, 256, 64)   │          0 │ dropout_23[0][0], │
│                     │                   │            │ add_14[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ layer_normalizatio… │ (None, 256, 64)   │        128 │ add_15[0][0]      │
│ (LayerNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten (Flatten)   │ (None, 16384)     │          0 │ layer_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_24          │ (None, 16384)     │          0 │ flatten[0][0]     │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_17 (Dense)    │ (None, 2048)      │ 33,556,480 │ dropout_24[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_25          │ (None, 2048)      │          0 │ dense_17[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_18 (Dense)    │ (None, 1024)      │  2,098,176 │ dropout_25[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_26          │ (None, 1024)      │          0 │ dense_18[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_19 (Dense)    │ (None, 3)         │      3,075 │ dropout_26[0][0]  │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 36,654,275 (139.82 MB)
 Trainable params: 36,654,275 (139.82 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 52s 357ms/step - accuracy: 0.3985 - loss: 14.8497 - top2: 0.6741 - val_accuracy: 0.4155 - val_loss: 5.1721 - val_top2: 0.9041
Epoch 2/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 9s 271ms/step - accuracy: 0.4294 - loss: 3.8661 - top2: 0.8080 - val_accuracy: 0.4384 - val_loss: 1.0968 - val_top2: 0.9041
Epoch 3/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 9s 274ms/step - accuracy: 0.4333 - loss: 2.1800 - top2: 0.7939 - val_accuracy: 0.6347 - val_loss: 0.9494 - val_top2: 0.9041
Epoch 4/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 166ms/step - accuracy: 0.4777 - loss: 1.2056 - top2: 0.8102 - val_accuracy: 0.4886 - val_loss: 0.9904 - val_top2: 0.9041
Epoch 5/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 166ms/step - accuracy: 0.5390 - loss: 0.9686 - top2: 0.8752 - val_accuracy: 0.4886 - val_loss: 0.9347 - val_top2: 0.9041
Epoch 6/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 168ms/step - accuracy: 0.4725 - loss: 1.0712 - top2: 0.8426 - val_accuracy: 0.6073 - val_loss: 0.8522 - val_top2: 0.9041
Epoch 7/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 9s 272ms/step - accuracy: 0.5495 - loss: 0.9869 - top2: 0.8608 - val_accuracy: 0.7078 - val_loss: 0.8373 - val_top2: 0.9041
Epoch 8/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 168ms/step - accuracy: 0.5570 - loss: 0.9353 - top2: 0.8753 - val_accuracy: 0.6941 - val_loss: 0.8343 - val_top2: 0.9041
Epoch 9/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 167ms/step - accuracy: 0.5705 - loss: 0.8952 - top2: 0.8927 - val_accuracy: 0.6712 - val_loss: 0.8265 - val_top2: 0.9041
Epoch 10/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 168ms/step - accuracy: 0.5895 - loss: 0.8929 - top2: 0.9036 - val_accuracy: 0.6438 - val_loss: 0.8175 - val_top2: 0.9041
Epoch 11/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 169ms/step - accuracy: 0.6096 - loss: 0.8586 - top2: 0.8890 - val_accuracy: 0.6621 - val_loss: 0.8482 - val_top2: 0.9041
Epoch 12/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 175ms/step - accuracy: 0.6026 - loss: 0.8865 - top2: 0.9020 - val_accuracy: 0.6438 - val_loss: 0.8790 - val_top2: 0.9041
Epoch 13/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 8s 201ms/step - accuracy: 0.5305 - loss: 0.9126 - top2: 0.8759 - val_accuracy: 0.6347 - val_loss: 0.8169 - val_top2: 0.9041
Epoch 14/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 10s 276ms/step - accuracy: 0.5336 - loss: 0.8877 - top2: 0.8945 - val_accuracy: 0.7169 - val_loss: 0.7871 - val_top2: 0.9041
Epoch 15/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 172ms/step - accuracy: 0.6145 - loss: 0.8699 - top2: 0.8964 - val_accuracy: 0.6256 - val_loss: 0.8131 - val_top2: 0.9041
Epoch 16/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 9s 276ms/step - accuracy: 0.5826 - loss: 0.8792 - top2: 0.8588 - val_accuracy: 0.7352 - val_loss: 0.8046 - val_top2: 0.9041
Epoch 17/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 172ms/step - accuracy: 0.6356 - loss: 0.8033 - top2: 0.9006 - val_accuracy: 0.6849 - val_loss: 0.7865 - val_top2: 0.9041
Epoch 18/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 179ms/step - accuracy: 0.6322 - loss: 0.8256 - top2: 0.8873 - val_accuracy: 0.6804 - val_loss: 0.7906 - val_top2: 0.9041
Epoch 19/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 175ms/step - accuracy: 0.6258 - loss: 0.8299 - top2: 0.8844 - val_accuracy: 0.6530 - val_loss: 0.7941 - val_top2: 0.9041
Epoch 20/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 173ms/step - accuracy: 0.6079 - loss: 0.8167 - top2: 0.8940 - val_accuracy: 0.6667 - val_loss: 0.7327 - val_top2: 0.9041
Epoch 21/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 171ms/step - accuracy: 0.5749 - loss: 0.8480 - top2: 0.8796 - val_accuracy: 0.6438 - val_loss: 0.8190 - val_top2: 0.9041
Epoch 22/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 171ms/step - accuracy: 0.6358 - loss: 0.8277 - top2: 0.8702 - val_accuracy: 0.7260 - val_loss: 0.7638 - val_top2: 0.9041
Epoch 23/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 171ms/step - accuracy: 0.6409 - loss: 0.8135 - top2: 0.8809 - val_accuracy: 0.6301 - val_loss: 0.7965 - val_top2: 0.9041
Epoch 24/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 171ms/step - accuracy: 0.6176 - loss: 0.8430 - top2: 0.8829 - val_accuracy: 0.6667 - val_loss: 0.7323 - val_top2: 0.9041
Epoch 25/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 171ms/step - accuracy: 0.6377 - loss: 0.8082 - top2: 0.8826 - val_accuracy: 0.7260 - val_loss: 0.6391 - val_top2: 0.9224
Epoch 26/100
28/28 ━━━━━━━━━━━━━━━━━━━━ 6s 172ms/step - accuracy: 0.6706 - loss: 0.7671 - top2: 0.8873 - val_accuracy: 0.7260 - val_loss: 0.6868 - val_top2: 0.9132
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - accuracy: 0.7458 - loss: 0.8217 - top2: 0.8986
Test: {'accuracy': 0.7351598143577576, 'loss': 0.804610013961792, 'top2': 0.9041095972061157}

![alt text](image.png)
![alt text](image-1.png)
```