# ─────────────────────────────────────────────────────────────────────────────
# train_model.py
# Training pipeline for the Real-Time Sign Language Translator.
#
# Architecture note — Dense Neural Network (DNN), NOT a CNN:
#   MediaPipe already distils each hand frame into 21 (x, y) landmark
#   coordinates, giving us a flat 42-feature vector.  That is tabular /
#   structured data — there is no 2-D spatial locality to exploit, so
#   convolutional layers would add parameters and complexity with no benefit.
#   A well-regularised DNN trains faster, generalises better on this data,
#   and converts to a tiny TFLite model suitable for real-time inference.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import pickle
import sys

# Ensure project root is on path regardless of where the script is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for all environments
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
    LABEL_MAP_PATH,
    LANDMARK_FEATURES,
    LANDMARKS_CSV,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_KERAS_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TEST_SPLIT,
    VALIDATION_SPLIT,
)
from utils.logger import get_logger

# ── Module-level logger ───────────────────────────────────────────────────────
logger = get_logger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
# SCALER_PATH is imported from config.py
ACCURACY_PLOT_PATH    = os.path.join(DOCS_DIR, "accuracy_plot.png")
LOSS_PLOT_PATH        = os.path.join(DOCS_DIR, "loss_plot.png")
CONFUSION_MATRIX_PATH = os.path.join(DOCS_DIR, "confusion_matrix.png")



# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the landmark dataset from a CSV file.

    The CSV is expected to have one column named ``label`` containing the
    gesture class name, followed by ``LANDMARK_FEATURES`` (42) numeric
    columns representing the normalised (x, y) coordinates of each
    MediaPipe hand landmark.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to ``landmarks.csv``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix of shape (n_samples, 42).
    y : pd.Series
        Raw string labels of shape (n_samples,).

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If the CSV does not contain a ``label`` column or has fewer than
        ``LANDMARK_FEATURES`` feature columns.
    """
    logger.info("Loading dataset from: %s", csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"landmarks.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Raw dataset shape: %s", df.shape)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) < LANDMARK_FEATURES:
        raise ValueError(
            f"Expected at least {LANDMARK_FEATURES} feature columns, "
            f"found {len(feature_cols)}."
        )

    # Use exactly the first LANDMARK_FEATURES columns as features
    feature_cols = feature_cols[:LANDMARK_FEATURES]

    X = df[feature_cols]
    y = df["label"]

    logger.info(
        "Dataset loaded — samples: %d, features: %d, classes: %s",
        len(X),
        len(feature_cols),
        sorted(y.unique().tolist()),
    )
    return X, y


def encode_labels(y: pd.Series, label_map_path: str) -> tuple[np.ndarray, LabelEncoder, dict]:
    """Encode string gesture labels as integers and persist the mapping.

    Parameters
    ----------
    y : pd.Series
        Raw string labels.
    label_map_path : str
        Path where ``label_map.json`` will be written.

    Returns
    -------
    y_encoded : np.ndarray
        Integer-encoded labels of shape (n_samples,).
    encoder : LabelEncoder
        Fitted :class:`~sklearn.preprocessing.LabelEncoder` instance.
    label_map : dict
        Mapping of ``{integer_index: class_name}`` — used at inference time.
    """
    logger.info("Encoding gesture labels …")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Build a JSON-serialisable str→int mapping  {"Hello": 0, "Yes": 1, …}
    # This is the canonical format used by inference.py at runtime.
    # Keys are gesture names (str); values are integer class indices.
    label_map = {cls: int(i) for i, cls in enumerate(encoder.classes_)}

    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2)

    logger.info("Label map saved to: %s", label_map_path)
    logger.info("Classes: %s", label_map)

    return y_encoded, encoder, label_map


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train, validation, and test subsets.

    The split ratios are read from ``config.py``:
    - ``TEST_SPLIT``       — fraction of the full dataset held out for testing.
    - ``VALIDATION_SPLIT`` — fraction of the *remaining* data used for
      validation (i.e., applied after the test split).

    Stratified splitting is used so that every class is proportionally
    represented in each subset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Integer-encoded labels.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Feature subsets.
    y_train, y_val, y_test : np.ndarray
        Label subsets.
    """
    logger.info(
        "Splitting data — test: %.0f%%, validation: %.0f%% of remainder …",
        TEST_SPLIT * 100,
        VALIDATION_SPLIT * 100,
    )

    X_np = X.values

    # First carve out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_np, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Then split the remainder into train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on the training set and transform all splits.

    The scaler is fitted *only* on the training data to prevent data
    leakage, then applied to validation and test sets.  The fitted scaler
    is serialised with ``pickle`` so it can be reloaded at inference time
    to apply identical normalisation to live webcam frames.

    Parameters
    ----------
    X_train : np.ndarray
        Raw training features.
    X_val : np.ndarray
        Raw validation features.
    X_test : np.ndarray
        Raw test features.
    scaler_path : str
        Path where ``scaler.pkl`` will be saved.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
        Normalised feature arrays.
    scaler : StandardScaler
        The fitted scaler instance.
    """
    logger.info("Fitting StandardScaler on training data …")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    logger.info("Scaler saved to: %s", scaler_path)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler



# ─────────────────────────────────────────────────────────────────────────────
# Model architecture & callbacks
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, input_dim: int = LANDMARK_FEATURES) -> Model:
    """Build and compile the Dense Neural Network classifier.

    Why a DNN and not a CNN?
    ─────────────────────────
    The input to this model is a flat 42-element vector of (x, y) landmark
    coordinates extracted by MediaPipe — pure tabular / structured data.
    CNNs are designed to learn spatial hierarchies from grid-like data
    (images, time-frequency spectrograms).  Feeding a 1-D landmark vector
    into a CNN would require artificial reshaping and would not yield any
    meaningful spatial patterns.  A DNN with BatchNormalization and Dropout
    achieves strong accuracy on this task with far fewer parameters and
    trains ~10× faster on CPU.

    Architecture
    ─────────────
    Input(42)
      → Dense(256, relu) → BatchNorm → Dropout(0.4)
      → Dense(128, relu) → BatchNorm → Dropout(0.3)
      → Dense(64,  relu) →            Dropout(0.2)
      → Dense(num_classes, softmax)

    Parameters
    ----------
    num_classes : int
        Number of gesture classes — determines the output layer width.
    input_dim : int, optional
        Number of input features (default: 42 from ``config.LANDMARK_FEATURES``).

    Returns
    -------
    model : tensorflow.keras.Model
        Compiled Keras model ready for training.
    """
    logger.info(
        "Building DNN — input_dim: %d, num_classes: %d", input_dim, num_classes
    )

    inputs = Input(shape=(input_dim,), name="landmarks_input")

    # Block 1
    x = Dense(256, activation="relu", name="dense_256")(inputs)
    x = BatchNormalization(name="bn_256")(x)
    x = Dropout(0.4, name="dropout_256")(x)

    # Block 2
    x = Dense(128, activation="relu", name="dense_128")(x)
    x = BatchNormalization(name="bn_128")(x)
    x = Dropout(0.3, name="dropout_128")(x)

    # Block 3
    x = Dense(64, activation="relu", name="dense_64")(x)
    x = Dropout(0.2, name="dropout_64")(x)

    # Output
    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="gesture_dnn")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    return model


def get_callbacks(model_save_path: str) -> list:
    """Build the list of Keras training callbacks.

    Callbacks used
    ──────────────
    * **ModelCheckpoint** — saves the epoch with the best ``val_accuracy``
      so the final artefact is always the best-performing checkpoint rather
      than the last epoch.
    * **EarlyStopping** — halts training when ``val_loss`` has not improved
      for ``patience=15`` consecutive epochs and restores the best weights
      automatically.
    * **ReduceLROnPlateau** — halves the learning rate when ``val_loss``
      plateaus for 7 epochs, helping the optimiser escape local minima.

    Parameters
    ----------
    model_save_path : str
        Path where the best model checkpoint will be written
        (e.g. ``models/gesture_model.keras``).

    Returns
    -------
    list
        Ordered list of instantiated Keras callback objects.
    """
    logger.info("Configuring training callbacks …")

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
    """Train the model and return the history object.

    Training runs for up to ``config.EPOCHS`` epochs but will stop early
    if the ``EarlyStopping`` callback triggers.  The best checkpoint is
    automatically saved via ``ModelCheckpoint``.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Compiled Keras model returned by :func:`build_model`.
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Integer training labels.
    X_val : np.ndarray
        Scaled validation features.
    y_val : np.ndarray
        Integer validation labels.
    callbacks : list
        List of Keras callbacks from :func:`get_callbacks`.

    Returns
    -------
    history : tf.keras.callbacks.History
        Training history containing per-epoch loss and accuracy values.
    """
    logger.info(
        "Starting training — max epochs: %d, batch size: %d", EPOCHS, BATCH_SIZE
    )

    # Set global random seeds for reproducibility
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

    actual_epochs = len(history.history["loss"])
    logger.info("Training complete — ran for %d epoch(s).", actual_epochs)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & reporting
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: dict,
) -> np.ndarray:
    """Evaluate the trained model on the held-out test set.

    Prints a full per-class classification report (precision, recall,
    F1-score, support) to both the console and the log file.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained (or best-checkpoint-restored) Keras model.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        Integer test labels.
    label_map : dict
        Mapping of ``{int: class_name}`` for human-readable report labels.

    Returns
    -------
    y_pred : np.ndarray
        Predicted integer class indices for ``X_test``.
    """
    logger.info("Evaluating model on test set (%d samples) …", len(X_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test loss: %.4f — Test accuracy: %.4f", test_loss, test_acc)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # label_map is {str: int}; build inverted {int: str} for display
    inv_map    = {v: k for k, v in label_map.items()}
    class_names = [inv_map[i] for i in sorted(inv_map.keys())]
    report = classification_report(y_test, y_pred, target_names=class_names)

    logger.info("\n%s", report)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy(history: tf.keras.callbacks.History, save_path: str) -> None:
    """Plot and save training vs. validation accuracy over epochs.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object returned by ``model.fit()``.
    save_path : str
        Full path where the PNG will be written
        (e.g. ``docs/accuracy_plot.png``).
    """
    logger.info("Saving accuracy plot → %s", save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, history.history["accuracy"],    label="Train Accuracy",      linewidth=2)
    ax.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    ax.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Accuracy plot saved.")


def plot_loss(history: tf.keras.callbacks.History, save_path: str) -> None:
    """Plot and save training vs. validation loss over epochs.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object returned by ``model.fit()``.
    save_path : str
        Full path where the PNG will be written
        (e.g. ``docs/loss_plot.png``).
    """
    logger.info("Saving loss plot → %s", save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, history.history["loss"],     label="Train Loss",      linewidth=2)
    ax.plot(epochs_range, history.history["val_loss"],  label="Validation Loss", linewidth=2, linestyle="--")
    ax.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Loss plot saved.")


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict,
    save_path: str,
) -> None:
    """Plot and save a normalised confusion matrix heatmap.

    Parameters
    ----------
    y_test : np.ndarray
        True integer labels.
    y_pred : np.ndarray
        Predicted integer labels from :func:`evaluate_model`.
    label_map : dict
        Mapping of ``{int: class_name}`` used for axis tick labels.
    save_path : str
        Full path where the PNG will be written
        (e.g. ``docs/confusion_matrix.png``).
    """
    logger.info("Saving confusion matrix → %s", save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # label_map is {str: int}; build inverted {int: str} for axis labels
    inv_map    = {v: k for k, v in label_map.items()}
    class_names = [inv_map[i] for i in sorted(inv_map.keys())]
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        xticks_rotation=45,
        values_format=".2f",
    )
    ax.set_title("Normalised Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved.")



# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline() -> None:
    """Execute the full model training pipeline end-to-end.

    Pipeline steps
    ──────────────
    1. Load landmark data from ``landmarks.csv``.
    2. Encode string labels to integers; save ``models/label_map.json``.
    3. Split into train / validation / test sets.
    4. Normalise features with StandardScaler; save ``models/scaler.pkl``.
    5. Build the DNN classifier.
    6. Train with ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
    7. Evaluate on the held-out test set; print classification report.
    8. Save accuracy, loss, and confusion-matrix plots to ``docs/``.
    9. Persist the final model as ``models/gesture_model.keras``.

    All intermediate artefacts and metrics are logged via the project
    logger so that every run leaves a full audit trail in ``logs/app.log``.

    Raises
    ------
    FileNotFoundError
        If ``landmarks.csv`` is missing.
    ValueError
        If the CSV schema does not match the expected format.
    """
    logger.info("=" * 60)
    logger.info("Sign Language Translator — Training Pipeline START")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y_raw = load_data(LANDMARKS_CSV)

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    y_encoded, encoder, label_map = encode_labels(y_raw, LABEL_MAP_PATH)
    num_classes = len(label_map)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)

    # ── 4. Normalise ──────────────────────────────────────────────────────────
    X_train, X_val, X_test, _ = normalize_features(
        X_train, X_val, X_test, SCALER_PATH
    )

    # ── 5. Build model ────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = get_callbacks(MODEL_KERAS_PATH)
    history = train(model, X_train, y_train, X_val, y_val, callbacks)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    y_pred = evaluate_model(model, X_test, y_test, label_map)

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_accuracy(history, ACCURACY_PLOT_PATH)
    plot_loss(history, LOSS_PLOT_PATH)
    plot_confusion_matrix(y_test, y_pred, label_map, CONFUSION_MATRIX_PATH)

    # ── 9. Save final model ───────────────────────────────────────────────────
    # ModelCheckpoint already saved the best epoch; this call ensures the
    # model object in memory (with restored best weights) is also on disk.
    model.save(MODEL_KERAS_PATH)
    logger.info("Final model saved → %s", MODEL_KERAS_PATH)

    logger.info("=" * 60)
    logger.info("Training Pipeline COMPLETE")
    logger.info("  Model      : %s", MODEL_KERAS_PATH)
    logger.info("  Scaler     : %s", SCALER_PATH)
    logger.info("  Label map  : %s", LABEL_MAP_PATH)
    logger.info("  Accuracy   : %s", ACCURACY_PLOT_PATH)
    logger.info("  Loss       : %s", LOSS_PLOT_PATH)
    logger.info("  Conf matrix: %s", CONFUSION_MATRIX_PATH)
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training_pipeline()
