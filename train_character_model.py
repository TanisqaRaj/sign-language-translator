# =============================================================================
# train_character_model.py
# Training Pipeline — Character Recognition Model (ISLRTC A-Z + 0-9)
# =============================================================================
# Usage:
#   python train_character_model.py
#
# Reads  : character_landmarks.csv   (output of preprocess_characters.py)
# Saves  :
#   models/character_model.keras
#   models/character_scaler.pkl
#   models/character_label_map.json
#
# WHY A DNN AND NOT LSTM:
#   The ISLRTC character dataset contains static images — each image is one
#   independent snapshot of a hand sign.  There is no temporal sequence.
#   A DNN on flat 42-feature landmark vectors is the correct architecture:
#     • Input is tabular / structured, not a time-series.
#     • 42 features have no 2-D spatial locality → CNN won't help.
#     • Same architecture that achieves >95% on word model — proven fit.
#
# FEATURE COUNT:
#   42 hand-only features (MediaPipe, same as app_cloud.py inference path).
#   A separate StandardScaler is fitted on THIS training set only.
#   It will NEVER be mixed with models/scaler.pkl (word model scaler).
#
# TRAINING/INFERENCE MISMATCH PREVENTION:
#   Scaler is saved with the feature count embedded in a sidecar JSON.
#   train_character_model.py and inference.py both validate the shape
#   at load time — a shape mismatch raises ValueError immediately.
# =============================================================================

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import (
    BATCH_SIZE,
    EPOCHS,
    LANDMARK_FEATURES,   # 42 — character model uses hand-only, same count
    CHAR_LANDMARKS_CSV,
    CHAR_LABEL_MAP_PATH,
    CHAR_SCALER_PATH,
    CHAR_MODEL_KERAS_PATH,
    MODEL_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    VALIDATION_SPLIT,
    LEARNING_RATE,
    NUM_CHARACTER_CLASSES,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Character model uses 42 hand features (identical to what app_cloud.py sends)
CHAR_FEATURE_COUNT = LANDMARK_FEATURES   # 42

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHAR_ACCURACY_PLOT_PATH    = os.path.join(DOCS_DIR, "char_accuracy_plot.png")
CHAR_LOSS_PLOT_PATH        = os.path.join(DOCS_DIR, "char_loss_plot.png")
CHAR_CONFUSION_MATRIX_PATH = os.path.join(DOCS_DIR, "char_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load character_landmarks.csv and validate the feature count.

    The CSV must have a 'label' column followed by exactly CHAR_FEATURE_COUNT
    numeric feature columns (42).

    Raises
    ------
    FileNotFoundError : csv_path does not exist
    ValueError        : wrong number of feature columns
    """
    logger.info("Loading character dataset: %s", csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"character_landmarks.csv not found at: {csv_path}\n"
            "Run preprocess_characters.py first."
        )

    df = pd.read_csv(csv_path)
    logger.info("Raw dataset shape: %s", df.shape)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    feature_cols = [c for c in df.columns if c != "label"]

    if len(feature_cols) != CHAR_FEATURE_COUNT:
        raise ValueError(
            f"Feature count mismatch!\n"
            f"  CSV has   : {len(feature_cols)} feature columns\n"
            f"  Expected  : {CHAR_FEATURE_COUNT}\n"
            "Re-run preprocess_characters.py to regenerate character_landmarks.csv."
        )

    X = df[feature_cols]
    y = df["label"]

    logger.info(
        "Dataset loaded — samples: %d, features: %d, classes: %s",
        len(X),
        len(feature_cols),
        sorted(y.unique().tolist()),
    )
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Label encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(y: pd.Series, label_map_path: str) -> tuple[np.ndarray, LabelEncoder, dict]:
    """
    Encode string character labels as integers and persist the mapping.

    The label map is saved to character_label_map.json — completely separate
    from models/label_map.json (word model).

    Returns
    -------
    y_encoded  : integer-encoded label array
    encoder    : fitted LabelEncoder
    label_map  : {class_name: int_index}
    """
    logger.info("Encoding character labels …")

    encoder   = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    label_map = {cls: int(i) for i, cls in enumerate(encoder.classes_)}

    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2)

    logger.info("Character label map saved: %s", label_map_path)
    logger.info("Classes (%d): %s", len(label_map), sorted(label_map.keys()))
    return y_encoded, encoder, label_map


# ─────────────────────────────────────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train / validation / test split (same ratios as word model).
    """
    logger.info("Splitting data — test: %.0f%%, val: %.0f%% …",
                TEST_SPLIT * 100, VALIDATION_SPLIT * 100)

    X_np = X.values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_np, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_temp
    )

    logger.info("Split sizes — train: %d, val: %d, test: %d",
                len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Feature normalisation  (separate scaler — never touches models/scaler.pkl)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the character training set only.

    The scaler is saved to character_scaler.pkl with a companion metadata
    file (.meta.json) recording the expected feature count.  inference.py
    validates this at load time to prevent silent shape mismatches.

    IMPORTANT: This scaler is completely independent of models/scaler.pkl.
    """
    logger.info("Fitting character StandardScaler on training data …")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    # Save companion metadata for validation at inference time
    meta_path = scaler_path + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump({"feature_count": CHAR_FEATURE_COUNT, "model": "character"}, fh)

    logger.info("Character scaler saved: %s", scaler_path)
    logger.info("Character scaler metadata: %s", meta_path)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_character_model(num_classes: int, input_dim: int = CHAR_FEATURE_COUNT) -> Model:
    """
    Build the Dense Neural Network for character classification.

    Architecture
    ────────────
    Input(42)
      → Dense(256, relu) → BatchNorm → Dropout(0.4)
      → Dense(128, relu) → BatchNorm → Dropout(0.3)
      → Dense(64,  relu) →            Dropout(0.2)
      → Dense(num_classes, softmax)

    This is identical in structure to the word model — proven architecture
    for static hand-landmark classification.

    36 output classes (A-Z + 0-9) vs 10 for word model — rest is the same.

    Why not LSTM?
    ─────────────
    The ISLRTC dataset is a collection of static images, one sign per image.
    There is no temporal dimension.  LSTM would require a sequence of frames
    per sample which is not how this dataset is structured.  DNN on flat
    landmark features is the architecturally correct choice.

    Parameters
    ----------
    num_classes : int  — typically 36 (A-Z + 0-9)
    input_dim   : int  — 42 (hand-only features)
    """
    logger.info(
        "Building character DNN — input_dim: %d, num_classes: %d",
        input_dim, num_classes,
    )

    inputs = Input(shape=(input_dim,), name="char_landmarks_input")

    # Block 1
    x = Dense(256, activation="relu", name="char_dense_256")(inputs)
    x = BatchNormalization(name="char_bn_256")(x)
    x = Dropout(0.4, name="char_dropout_256")(x)

    # Block 2
    x = Dense(128, activation="relu", name="char_dense_128")(x)
    x = BatchNormalization(name="char_bn_128")(x)
    x = Dropout(0.3, name="char_dropout_128")(x)

    # Block 3
    x = Dense(64, activation="relu", name="char_dense_64")(x)
    x = Dropout(0.2, name="char_dropout_64")(x)

    # Output
    outputs = Dense(num_classes, activation="softmax", name="char_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="character_dnn")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    return model


def get_callbacks(model_save_path: str) -> list:
    """Training callbacks — same strategy as word model."""
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    )
    return [checkpoint, early_stop, reduce_lr]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    callbacks: list,
) -> tf.keras.callbacks.History:
    """Train and return history."""
    logger.info("Starting character model training — max epochs: %d, batch: %d",
                EPOCHS, BATCH_SIZE)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Training complete — %d epochs run.", len(history.history["loss"]))
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: dict,
) -> np.ndarray:
    """Evaluate on held-out test set and print classification report."""
    logger.info("Evaluating character model on test set (%d samples) …", len(X_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Character test loss: %.4f — accuracy: %.4f", test_loss, test_acc)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    inv_map     = {v: k for k, v in label_map.items()}
    class_names = [inv_map[i] for i in sorted(inv_map.keys())]
    report      = classification_report(y_test, y_pred, target_names=class_names)

    logger.info("\n%s", report)
    print("\n" + "=" * 60)
    print("Character Model — Classification Report")
    print("=" * 60)
    print(report)

    return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy(history, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, history.history["accuracy"],     label="Train Accuracy",      linewidth=2)
    ax.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    ax.set_title("Character Model Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Character accuracy plot saved: %s", save_path)


def plot_loss(history, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs_range = range(1, len(history.history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, history.history["loss"],     label="Train Loss",      linewidth=2)
    ax.plot(epochs_range, history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    ax.set_title("Character Model Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Character loss plot saved: %s", save_path)


def plot_confusion_matrix(y_test, y_pred, label_map: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    inv_map     = {v: k for k, v in label_map.items()}
    class_names = [inv_map[i] for i in sorted(inv_map.keys())]
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    # 36 classes — use a larger figure
    fig_size = max(12, len(class_names) // 2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size - 2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45, values_format=".2f")
    ax.set_title("Character Model — Normalised Confusion Matrix",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Character confusion matrix saved: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline() -> None:
    """
    Full character model training pipeline.

    Pipeline steps
    ──────────────
    1. Load character_landmarks.csv  (42 features per sample)
    2. Encode labels → character_label_map.json
    3. Stratified train / val / test split
    4. Fit character StandardScaler → character_scaler.pkl
    5. Build DNN (Input=42, Output=36 classes)
    6. Train with callbacks
    7. Evaluate on test set
    8. Save training plots to docs/
    9. Save character_model.keras

    Safety checks
    ─────────────
    • Feature count validated before training (must == 42).
    • Scaler saved with companion .meta.json for inference-time validation.
    • Word model artefacts (gesture_model.*, scaler.pkl, label_map.json)
      are NEVER touched.
    """
    logger.info("=" * 60)
    logger.info("Character Model — Training Pipeline START")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("  Character Recognition Training Pipeline")
    print(f"  Input : {CHAR_FEATURE_COUNT} hand-only features (MediaPipe 42)")
    print(f"  Output: {NUM_CHARACTER_CLASSES} classes (A-Z + 0-9)")
    print("=" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y_raw = load_data(CHAR_LANDMARKS_CSV)

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    y_encoded, encoder, label_map = encode_labels(y_raw, CHAR_LABEL_MAP_PATH)
    num_classes = len(label_map)
    print(f"  Classes found in dataset: {sorted(label_map.keys())}\n")

    # ── 3. Split ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)

    # ── 4. Normalise (character-only scaler) ──────────────────────────────────
    X_train, X_val, X_test, _ = normalize_features(
        X_train, X_val, X_test, CHAR_SCALER_PATH
    )

    # ── 5. Build model ────────────────────────────────────────────────────────
    model = build_character_model(num_classes=num_classes)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = get_callbacks(CHAR_MODEL_KERAS_PATH)
    history   = train(model, X_train, y_train, X_val, y_val, callbacks)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    y_pred = evaluate_model(model, X_test, y_test, label_map)

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_accuracy(history, CHAR_ACCURACY_PLOT_PATH)
    plot_loss(history, CHAR_LOSS_PLOT_PATH)
    plot_confusion_matrix(y_test, y_pred, label_map, CHAR_CONFUSION_MATRIX_PATH)

    # ── 9. Save model ─────────────────────────────────────────────────────────
    model.save(CHAR_MODEL_KERAS_PATH)
    logger.info("Character Keras model saved: %s", CHAR_MODEL_KERAS_PATH)

    logger.info("=" * 60)
    logger.info("Character Training Pipeline COMPLETE")
    logger.info("  Model     : %s", CHAR_MODEL_KERAS_PATH)
    logger.info("  Scaler    : %s", CHAR_SCALER_PATH)
    logger.info("  Label map : %s", CHAR_LABEL_MAP_PATH)
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Model    → {CHAR_MODEL_KERAS_PATH}")
    print(f"  Scaler   → {CHAR_SCALER_PATH}")
    print(f"  LabelMap → {CHAR_LABEL_MAP_PATH}")
    print("  Next: run convert_character_tflite.py")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_training_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        print(f"\n  [ERROR]: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error in character training pipeline.")
        print(f"\n  [ERROR]: {exc}")
        sys.exit(1)
