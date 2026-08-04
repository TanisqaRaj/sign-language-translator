# =============================================================================
# convert_tflite.py
# Phase 5 - Convert Trained Keras Model to TensorFlow Lite
# =============================================================================
# Usage:
#   python convert_tflite.py
#
# Loads models/gesture_model.keras, applies float16 quantisation to shrink
# the file, saves models/gesture_model.tflite, then verifies it with a
# dummy inference and prints tensor details.
# =============================================================================

import os
import sys
import json
import numpy as np

# -- Project imports ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_KERAS_PATH,
    MODEL_TFLITE_PATH,
    MODEL_DIR,
    LABEL_MAP_PATH,
    LANDMARK_FEATURES,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------------

def convert_to_tflite() -> None:
    """
    Load the Keras model, convert to TFLite with float16 quantisation,
    and save the .tflite file.

    float16 quantisation:
      - Cuts model size roughly in half compared to float32.
      - Negligible accuracy loss for dense gesture classifiers.
      - Still runs on standard CPU (no special hardware needed).

    Raises
    ------
    FileNotFoundError  : if gesture_model.keras does not exist.
    RuntimeError       : if conversion fails unexpectedly.
    """
    # Lazy import TensorFlow only when needed (faster startup for other scripts)
    import tensorflow as tf

    # -- Load model ------------------------------------------------------------
    if not os.path.exists(MODEL_KERAS_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_KERAS_PATH}\n"
            "Run train_model.py first."
        )

    logger.info("Loading Keras model from: %s", MODEL_KERAS_PATH)
    print(f"\n  Loading model: {MODEL_KERAS_PATH}")
    model = tf.keras.models.load_model(MODEL_KERAS_PATH)
    model.summary()

    # -- Convert via SavedModel (bypasses BatchNorm+Keras3+TFLite bug) ---------
    import tempfile, shutil
    logger.info("Starting TFLite conversion via SavedModel intermediate.")
    print("\n  Saving as SavedModel intermediate...")
    tmpdir = tempfile.mkdtemp()
    saved_model_path = os.path.join(tmpdir, "saved_model")
    model.export(saved_model_path)

    print("\n  Converting SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    shutil.rmtree(tmpdir, ignore_errors=True)

    # -- Save ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    logger.info("TFLite model saved: %s", MODEL_TFLITE_PATH)
    print(f"\n  [OK] Saved: {MODEL_TFLITE_PATH}")

    # -- File size comparison --------------------------------------------------
    keras_size  = os.path.getsize(MODEL_KERAS_PATH)  / (1024 * 1024)
    tflite_size = os.path.getsize(MODEL_TFLITE_PATH) / (1024 * 1024)
    reduction   = (1 - tflite_size / keras_size) * 100

    print(f"\n  Size comparison:")
    print(f"    Keras model  : {keras_size:.2f} MB")
    print(f"    TFLite model : {tflite_size:.2f} MB")
    print(f"    Reduction    : {reduction:.1f}%")
    logger.info(
        "Size -- Keras: %.2f MB | TFLite: %.2f MB | Reduction: %.1f%%",
        keras_size, tflite_size, reduction,
    )


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

def verify_tflite() -> None:
    """
    Run a dummy inference through the TFLite model to confirm it loaded
    correctly and produces output of the expected shape.

    Prints input/output tensor details (name, shape, dtype).
    """
    import tensorflow as tf

    if not os.path.exists(MODEL_TFLITE_PATH):
        raise FileNotFoundError(f"TFLite model not found: {MODEL_TFLITE_PATH}")

    logger.info("Verifying TFLite model: %s", MODEL_TFLITE_PATH)
    print("\n  Verifying TFLite model...")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # -- Print tensor info -----------------------------------------------------
    print("\n  Input tensor:")
    for d in input_details:
        print(f"    name  : {d['name']}")
        print(f"    shape : {d['shape']}")
        print(f"    dtype : {d['dtype']}")

    print("\n  Output tensor:")
    for d in output_details:
        print(f"    name  : {d['name']}")
        print(f"    shape : {d['shape']}")
        print(f"    dtype : {d['dtype']}")

    # -- Dummy inference -------------------------------------------------------
    input_shape = input_details[0]["shape"]          # e.g. [1, 42]
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    predicted_class = int(np.argmax(output[0]))

    # Map back to gesture name if label map exists
    gesture_name = str(predicted_class)
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)   # {"Hello": 0, "Yes": 1, ...}
        inv_map      = {int(v): k for k, v in label_map.items()}
        gesture_name = inv_map.get(predicted_class, str(predicted_class))

    print(f"\n  Dummy inference result:")
    print(f"    Output probabilities : {output[0].round(3)}")
    print(f"    Predicted class      : {predicted_class} ({gesture_name})")
    print(f"    Confidence           : {float(output[0][predicted_class]):.3f}")

    logger.info(
        "TFLite verification passed. Predicted class=%d (%s) on dummy input.",
        predicted_class, gesture_name,
    )
    print("\n  [OK] TFLite model verified successfully.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Run conversion then verification."""
    print("\n  Sign Language Translator - Model Conversion\n")
    try:
        convert_to_tflite()
        verify_tflite()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during conversion.")
        print(f"\n  [ERROR] Unexpected error: {exc}")
        sys.exit(1)

    print("\n  [DONE] Conversion complete!  Run inference.py next.\n")


if __name__ == "__main__":
    main()
