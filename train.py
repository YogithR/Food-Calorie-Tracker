import os
import sys
import shutil
import tensorflow as tf
# import tensorflow_datasets as tfds

# -----------------------------
# CONFIG
# -----------------------------
DATASET_NAME = "food101"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

# IMPORTANT: Put TFDS cache on D drive (change this if your D drive path is different)
TFDS_DIR = r"D:\tfds"

# Output files used by Streamlit app
OUT_DIR = "models"
MODEL_OUT_PATH = os.path.join(OUT_DIR, "food101_mobilenetv2.keras")
CLASS_NAMES_OUT_PATH = os.path.join(OUT_DIR, "class_names.txt")

# -----------------------------
# Utility: clean incomplete folders (common reason it hangs)
# -----------------------------
def cleanup_incomplete_folders(tfds_dir: str):
    """
    TFDS sometimes leaves incomplete folders if a previous run crashed or disk ran out.
    That can cause TFDS to hang or behave weird next day.
    """
    food_dir = os.path.join(tfds_dir, DATASET_NAME)
    if not os.path.exists(food_dir):
        return

    # Remove folders like incomplete.* inside D:\tfds\food101\
    for name in os.listdir(food_dir):
        if name.startswith("incomplete."):
            path = os.path.join(food_dir, name)
            print(f"[CLEANUP] Removing leftover folder: {path}", flush=True)
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"[CLEANUP] Could not remove {path}: {e}", flush=True)

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # transfer learning: freeze base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # IMPORTANT: use sparse_categorical_* because labels are integer class ids
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    print("TRAINING SCRIPT STARTED", flush=True)

    # Force TFDS to use D:\tfds
    os.environ["TFDS_DATA_DIR"] = TFDS_DIR
    print(f"[INFO] Using TFDS_DATA_DIR = {os.environ['TFDS_DATA_DIR']}", flush=True)

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Cleanup incomplete folders (big reason it works one day, fails next day)
    cleanup_incomplete_folders(TFDS_DIR)

    # Test TFDS import + builder
    print("[INFO] Checking TFDS builder...", flush=True)
    builder = tfds.builder(DATASET_NAME, data_dir=TFDS_DIR)
    print("[INFO] Builder OK", flush=True)

    # Download/prepare only if needed
    # If already downloaded, TFDS reuses it.
    print("[INFO] Downloading/preparing dataset if needed (first time takes long)...", flush=True)
    builder.download_and_prepare()
    print("[INFO] Dataset is ready.", flush=True)

    # Load dataset
    print("[INFO] Loading train/validation splits...", flush=True)
    train_ds = builder.as_dataset(split="train", as_supervised=True)
    val_ds = builder.as_dataset(split="validation", as_supervised=True)
    info = builder.info

    class_names = info.features["label"].names
    print(f"[INFO] Number of classes: {len(class_names)}", flush=True)
    print(f"[INFO] Example classes: {class_names[:10]}", flush=True)

    # Save class names for Streamlit app
    with open(CLASS_NAMES_OUT_PATH, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"[INFO] Saved class names → {CLASS_NAMES_OUT_PATH}", flush=True)

    # Prepare pipeline (fast)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        train_ds
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .shuffle(2000)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # Build + train model
    model = build_model(num_classes=len(class_names))
    print(model.summary(), flush=True)

    print("[INFO] Training started...", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save model
    model.save(MODEL_OUT_PATH)
    print(f"[INFO] Saved model → {MODEL_OUT_PATH}", flush=True)

    print("TRAINING COMPLETED ✅", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Training interrupted by user.", flush=True)
        sys.exit(0)
