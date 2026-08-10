# =============================================================================
# convert_character_tflite.py
# Convert character_model.keras → character_model.tflite
# =============================================================================
# Usage:
#   python convert_character_tflite.py
#
# Reads  : models/character_model.keras
# Writes : models/character_model.tflite
#
# Uses the same SavedModel-intermediate approach as convert_tflite.py to
# avoid the Keras 3 + TFLite BatchNormalization compatibility issue.
# =============================================================================

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHAR_MODEL_KERAS_PATH,
    CHAR_MODEL_TFLITE_PATH,
    CHAR_LABEL_MAP_PATH,
    CHAR_SCALER_PATH,
    MODEL_DIR,
    LANDMARK_FEATURES,   # 42 — character model input
)
from utils.logger import get_logger

logger = get_logger(__name__)

CHAR_FEATURE_COUNT = LANDMARK_FEATURES   # 42


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_tflite() -> None:
    """
    Load character_model.keras, convert to TFLite, save character_model.tflite.

    Uses SavedModel intermediate to bypass the BatchNorm + Keras3 bug.
    Quantisation: DEFAULT (float32 weights with possible int8 activations)
    — same approach as convert_tflite.py.
    """
    import tensorflow as tf
    import tempfile
    import shutil

    if not os.path.exists(CHAR_MODEL_KERAS_PATH):
        raise FileNotFoundError(
            f"Character Keras model not found: {CHAR_MODEL_KERAS_PATH}\n"
            "Run train_character_model.py first."
        )

    logger.info("Loading character Keras model: %s", CHAR_MODEL_KERAS_PATH)
    print(f"\n  Loading character model: {CHAR_MODEL_KERAS_PATH}")
    model = tf.keras.models.load_model(CHAR_MODEL_KERAS_PATH)
    model.summary()

    # Verify input shape
    expected_shape = (None, CHAR_FEATURE_COUNT)
    actual_shape   = tuple(model.input_shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Character model input shape mismatch!\n"
            f"  Expected : {expected_shape}\n"
            f"  Got      : {actual_shape}\n"
            "Ensure train_character_model.py was run with 42 features."
        )

    print(f"\n  ✅ Input shape verified: {actual_shape}")

    # Convert via SavedModel intermediate
    print("\n  Saving intermediate SavedModel…")
    tmpdir           = tempfile.mkdtemp()
    saved_model_path = os.path.join(tmpdir, "saved_model")
    model.export(saved_model_path)

    print("\n  Converting SavedModel → TFLite…")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    shutil.rmtree(tmpdir, ignore_errors=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CHAR_MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    logger.info("Character TFLite model saved: %s", CHAR_MODEL_TFLITE_PATH)
    print(f"\n  [OK] Saved: {CHAR_MODEL_TFLITE_PATH}")

    keras_size  = os.path.getsize(CHAR_MODEL_KERAS_PATH)  / (1024 * 1024)
    tflite_size = os.path.getsize(CHAR_MODEL_TFLITE_PATH) / (1024 * 1024)
    reduction   = (1 - tflite_size / keras_size) * 100 if keras_size > 0 else 0

    print(f"\n  Size comparison:")
    print(f"    Keras model  : {keras_size:.2f} MB")
    print(f"    TFLite model : {tflite_size:.2f} MB")
    print(f"    Reduction    : {reduction:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_tflite() -> None:
    """
    Run a dummy inference through character_model.tflite and verify:
      • Input tensor shape is (1, 42)
      • Output tensor shape is (1, num_classes)
      • num_classes matches character_label_map.json
    """
    import tensorflow as tf

    if not os.path.exists(CHAR_MODEL_TFLITE_PATH):
        raise FileNotFoundError(
            f"Character TFLite model not found: {CHAR_MODEL_TFLITE_PATH}"
        )

    print("\n  Verifying character TFLite model…")
    interpreter = tf.lite.Interpreter(model_path=CHAR_MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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

    # Validate input shape == (1, 42)
    input_shape = tuple(input_details[0]["shape"])
    if input_shape != (1, CHAR_FEATURE_COUNT):
        raise ValueError(
            f"Character TFLite input shape mismatch!\n"
            f"  Expected : (1, {CHAR_FEATURE_COUNT})\n"
            f"  Got      : {input_shape}"
        )
    print(f"\n  ✅ Input shape correct: {input_shape}")

    # Validate output shape against label map
    num_outputs = output_details[0]["shape"][1]
    if os.path.exists(CHAR_LABEL_MAP_PATH):
        with open(CHAR_LABEL_MAP_PATH) as f:
            label_map = json.load(f)
        num_classes = len(label_map)
        if num_outputs != num_classes:
            raise ValueError(
                f"Output shape mismatch with character_label_map.json!\n"
                f"  Model outputs   : {num_outputs} classes\n"
                f"  Label map has   : {num_classes} classes"
            )
        print(f"  ✅ Output classes correct: {num_outputs} (matches label_map.json)")

    # Dummy inference
    dummy_input = np.random.rand(1, CHAR_FEATURE_COUNT).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    predicted_idx  = int(np.argmax(output[0]))
    predicted_conf = float(output[0][predicted_idx])

    char_name = str(predicted_idx)
    if os.path.exists(CHAR_LABEL_MAP_PATH):
        inv_map   = {int(v): k for k, v in label_map.items()}
        char_name = inv_map.get(predicted_idx, char_name)

    print(f"\n  Dummy inference result:")
    print(f"    Predicted class : {predicted_idx} ({char_name})")
    print(f"    Confidence      : {predicted_conf:.3f}")
    print("\n  [OK] Character TFLite model verified successfully.")
    logger.info("Character TFLite verification passed — predicted class %d (%s).",
                predicted_idx, char_name)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n  Sign Language Translator — Character Model Conversion\n")
    try:
        convert_to_tflite()
        verify_tflite()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during character model conversion.")
        print(f"\n  [ERROR] Unexpected: {exc}")
        sys.exit(1)

    print("\n  [DONE] Character model conversion complete!")
    print("  Next: streamlit run app_cloud.py\n")


if __name__ == "__main__":
    main()
