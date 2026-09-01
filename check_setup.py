# =============================================================================
# check_setup.py
# Verify that all required files are in place before running the app.
# Usage:  python check_setup.py
#         python check_setup.py --character   (character pipeline only)
#         python check_setup.py --both        (word + character pipelines)
# =============================================================================

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    # Word model
    DATASET_DIR, LANDMARKS_CSV, MODEL_KERAS_PATH,
    MODEL_TFLITE_PATH, LABEL_MAP_PATH, SCALER_PATH, GESTURE_LABELS,
    LANDMARK_FEATURES,
    # Character model
    CHAR_DATASET_DIR, CHAR_LANDMARKS_CSV, CHAR_MODEL_KERAS_PATH,
    CHAR_MODEL_TFLITE_PATH, CHAR_LABEL_MAP_PATH, CHAR_SCALER_PATH,
    CHARACTER_LABELS,
)

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "


def check(label: str, condition: bool, required: bool = True) -> bool:
    icon = PASS if condition else (FAIL if required else WARN)
    print(f"{icon}  {label}")
    return condition


def _validate_tflite_shape(tflite_path: str, expected_features: int) -> tuple[bool, str]:
    """
    Load a TFLite model and validate its input shape.

    Returns (ok, message).
    """
    try:
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        interp = Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        shape = tuple(interp.get_input_details()[0]["shape"])

        if shape[1] == expected_features:
            return True, f"input shape {shape} ✓"
        else:
            return False, f"input shape {shape} — expected (1, {expected_features})"
    except Exception as exc:
        return False, str(exc)


def _validate_label_map(path: str, model_name: str) -> tuple[bool, str]:
    """Load and sanity-check a label_map.json. Returns (ok, message)."""
    try:
        with open(path) as f:
            lm = json.load(f)
        n = len(lm)
        classes = sorted(lm.keys())[:5]
        return True, f"{n} classes  (e.g. {classes}…)"
    except Exception as exc:
        return False, str(exc)


def _validate_scaler(path: str, expected_features: int) -> tuple[bool, str]:
    """Load scaler and check n_features_in_. Returns (ok, message)."""
    try:
        import pickle
        with open(path, "rb") as f:
            sc = pickle.load(f)
        n = getattr(sc, "n_features_in_", None)
        if n is None:
            return True, "loaded (feature count unknown)"
        if n == expected_features:
            return True, f"n_features_in_={n} ✓"
        else:
            return False, f"n_features_in_={n} — expected {expected_features}"
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Word pipeline check
# ─────────────────────────────────────────────────────────────────────────────

def check_word_pipeline() -> bool:
    print("\n" + "=" * 60)
    print("  WORD MODEL PIPELINE (6 gestures)")
    print("=" * 60)

    all_ok = True

    # Step 1: Dataset
    print("\n[Step 1] Dataset — collect_data.py")
    dataset_ok = True
    for gesture in GESTURE_LABELS:
        folder = os.path.join(DATASET_DIR, gesture)
        count  = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]) if os.path.isdir(folder) else 0
        ok = check(f"{gesture:<22} {count:>4} images", count >= 100)
        if not ok:
            dataset_ok = False
    if not dataset_ok:
        all_ok = False

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing — preprocess.py")
    check("landmarks.csv exists", os.path.exists(LANDMARKS_CSV))
    if not os.path.exists(LANDMARKS_CSV):
        all_ok = False

    # Step 3: Training artefacts
    print("\n[Step 3] Training — train_model.py")
    check("gesture_model.keras",  os.path.exists(MODEL_KERAS_PATH))
    check("label_map.json exists", os.path.exists(LABEL_MAP_PATH))
    if os.path.exists(LABEL_MAP_PATH):
        ok, msg = _validate_label_map(LABEL_MAP_PATH, "word")
        check(f"label_map.json valid  ({msg})", ok)
        if not ok:
            all_ok = False
    check("scaler.pkl exists", os.path.exists(SCALER_PATH))
    if os.path.exists(SCALER_PATH):
        ok, msg = _validate_scaler(SCALER_PATH, LANDMARK_FEATURES)
        check(f"scaler.pkl valid  ({msg})", ok)
        if not ok:
            all_ok = False

    # Step 4: TFLite
    print("\n[Step 4] Conversion — convert_tflite.py")
    check("gesture_model.tflite exists", os.path.exists(MODEL_TFLITE_PATH))
    if os.path.exists(MODEL_TFLITE_PATH):
        ok, msg = _validate_tflite_shape(MODEL_TFLITE_PATH, LANDMARK_FEATURES)
        check(f"TFLite shape valid  ({msg})", ok)
        if not ok:
            all_ok = False

    # Step 5: Docs plots (optional)
    print("\n[Step 5] Training plots — docs/ (optional)")
    for plot in ["accuracy_plot.png", "loss_plot.png", "confusion_matrix.png"]:
        path = os.path.join(os.path.dirname(__file__), "docs", plot)
        check(plot, os.path.exists(path), required=False)

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Character pipeline check
# ─────────────────────────────────────────────────────────────────────────────

def check_character_pipeline() -> bool:
    print("\n" + "=" * 60)
    print("  CHARACTER MODEL PIPELINE (A-Z + 0-9, 36 classes)")
    print("=" * 60)

    all_ok = True

    # Step 1: Dataset
    print("\n[Step 1] Character dataset — setup_character_dataset.py")
    dataset_ok = True
    total_images = 0
    for label in CHARACTER_LABELS:
        folder = os.path.join(CHAR_DATASET_DIR, label)
        count  = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]) if os.path.isdir(folder) else 0
        total_images += count

    if total_images == 0:
        check("dataset/characters/ images found", False)
        print(f"  → Run: python setup_character_dataset.py")
        print(f"  → Then download the ISLRTC Kaggle dataset and place images in:")
        print(f"    {CHAR_DATASET_DIR}")
        dataset_ok = False
        all_ok = False
    else:
        # Count per class
        empty_classes  = []
        sparse_classes = []
        ready_classes  = []
        for label in CHARACTER_LABELS:
            folder = os.path.join(CHAR_DATASET_DIR, label)
            count  = len([
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]) if os.path.isdir(folder) else 0
            if count == 0:
                empty_classes.append(label)
            elif count < 50:
                sparse_classes.append(label)
            else:
                ready_classes.append(label)

        check(f"Total images found: {total_images}", total_images > 0)
        check(f"Classes with ≥50 images: {len(ready_classes)}/{len(CHARACTER_LABELS)}",
              len(ready_classes) == len(CHARACTER_LABELS))
        if empty_classes:
            print(f"{WARN}  Empty classes: {empty_classes[:10]}"
                  + ("…" if len(empty_classes) > 10 else ""))
        if sparse_classes:
            print(f"{WARN}  Sparse classes (<50 images): {sparse_classes[:10]}"
                  + ("…" if len(sparse_classes) > 10 else ""))
        if empty_classes or sparse_classes:
            dataset_ok = False  # noqa: F841

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing — preprocess_characters.py")
    check("character_landmarks.csv exists", os.path.exists(CHAR_LANDMARKS_CSV))
    if not os.path.exists(CHAR_LANDMARKS_CSV):
        all_ok = False

    # Step 3: Training artefacts
    print("\n[Step 3] Training — train_character_model.py")
    check("character_model.keras exists",    os.path.exists(CHAR_MODEL_KERAS_PATH))
    check("character_label_map.json exists", os.path.exists(CHAR_LABEL_MAP_PATH))
    if os.path.exists(CHAR_LABEL_MAP_PATH):
        ok, msg = _validate_label_map(CHAR_LABEL_MAP_PATH, "character")
        check(f"character_label_map.json valid  ({msg})", ok)
        if not ok:
            all_ok = False
    check("character_scaler.pkl exists", os.path.exists(CHAR_SCALER_PATH))
    if os.path.exists(CHAR_SCALER_PATH):
        ok, msg = _validate_scaler(CHAR_SCALER_PATH, LANDMARK_FEATURES)
        check(f"character_scaler.pkl valid  ({msg})", ok)
        if not ok:
            all_ok = False
    # Check scaler meta file
    meta_path = CHAR_SCALER_PATH + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            fc = meta.get("feature_count")
            model_tag = meta.get("model")
            ok = (fc == LANDMARK_FEATURES and model_tag == "character")
            check(
                f"character_scaler.pkl.meta.json  (feature_count={fc}, model={model_tag!r})",
                ok,
            )
            if not ok:
                all_ok = False
        except Exception as exc:
            check(f"character_scaler.pkl.meta.json  ({exc})", False)
            all_ok = False
    else:
        check("character_scaler.pkl.meta.json", False, required=False)

    # Step 4: TFLite
    print("\n[Step 4] Conversion — convert_character_tflite.py")
    check("character_model.tflite exists", os.path.exists(CHAR_MODEL_TFLITE_PATH))
    if os.path.exists(CHAR_MODEL_TFLITE_PATH):
        ok, msg = _validate_tflite_shape(CHAR_MODEL_TFLITE_PATH, LANDMARK_FEATURES)
        check(f"TFLite input shape valid  ({msg})", ok)
        if not ok:
            all_ok = False

    # Step 5: Docs plots (optional)
    print("\n[Step 5] Training plots — docs/ (optional)")
    for plot in ["char_accuracy_plot.png", "char_loss_plot.png", "char_confusion_matrix.png"]:
        path = os.path.join(os.path.dirname(__file__), "docs", plot)
        check(plot, os.path.exists(path), required=False)

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(word_ok: bool | None, char_ok: bool | None) -> None:
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if word_ok is True:
        print(f"{PASS}  Word model pipeline — ready!")
    elif word_ok is False:
        print(f"{FAIL}  Word model pipeline — incomplete.")

    if char_ok is True:
        print(f"{PASS}  Character model pipeline — ready!")
    elif char_ok is False:
        print(f"{FAIL}  Character model pipeline — incomplete.")

    # Next steps
    print()
    if word_ok is True and (char_ok is True or char_ok is None):
        print("✅  Ready to run:")
        print("    streamlit run app_cloud.py")
        print("    python inference.py             (word mode, OpenCV window)")
        print("    python character_inference.py   (character mode, OpenCV window)")
    else:
        print("❌  Not ready. Follow the steps above in order.\n")
        if word_ok is False:
            print("  Word pipeline:")
            if not os.path.exists(LANDMARKS_CSV):
                print("    → python preprocess.py")
            if not os.path.exists(MODEL_KERAS_PATH):
                print("    → python train_model.py")
            if not os.path.exists(MODEL_TFLITE_PATH):
                print("    → python convert_tflite.py")
        if char_ok is False:
            print("  Character pipeline:")
            char_dataset_empty = not any(
                len([
                    f for f in os.listdir(os.path.join(CHAR_DATASET_DIR, label))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]) > 0
                for label in CHARACTER_LABELS
                if os.path.isdir(os.path.join(CHAR_DATASET_DIR, label))
            )
            if char_dataset_empty:
                print("    → python setup_character_dataset.py")
                print("    → (place Kaggle ISLRTC images in dataset/characters/)")
            if not os.path.exists(CHAR_LANDMARKS_CSV):
                print("    → python preprocess_characters.py")
            if not os.path.exists(CHAR_MODEL_KERAS_PATH):
                print("    → python train_character_model.py")
            if not os.path.exists(CHAR_MODEL_TFLITE_PATH):
                print("    → python convert_character_tflite.py")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the Sign Language Translator setup."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--word",      action="store_true",
                       help="Check word pipeline only (default)")
    group.add_argument("--character", action="store_true",
                       help="Check character pipeline only")
    group.add_argument("--both",      action="store_true",
                       help="Check both pipelines")
    args = parser.parse_args()

    print("\n🤙  Sign Language Translator — Setup Check\n" + "=" * 60)

    word_ok = None
    char_ok = None

    # Default behaviour: check both pipelines
    run_word = args.word or args.both or (not args.character)
    run_char = args.character or args.both or (not args.word)

    if run_word:
        word_ok = check_word_pipeline()

    if run_char:
        char_ok = check_character_pipeline()

    print_summary(word_ok, char_ok)


if __name__ == "__main__":
    main()
