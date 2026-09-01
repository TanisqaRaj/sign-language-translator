# =============================================================================
# preprocess.py
# Phase 3 – Hand Landmark Extraction (MediaPipe Hands only)
# =============================================================================
# Usage:
#   python preprocess.py
#
# For every image in dataset/ this script:
#   1. Runs MediaPipe Hands  → 42 normalised hand-landmark features (x,y × 21)
#   2. Saves landmarks.csv
#
# Images where no hand is detected are skipped.
# =============================================================================

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_DIR,
    LANDMARKS_CSV,
    NUM_LANDMARKS,
    GESTURE_LABELS,
    MODEL_DIR,
    LABEL_MAP_PATH,
    LANDMARK_FEATURES,
)
from utils.logger import get_logger
from utils.mediapipe_helper import HandDetector

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hand landmark normalisation  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_landmarks(landmarks_flat: list[float]) -> list[float]:
    """
    Normalise 42 raw hand-landmark values so predictions are position-invariant.

    Strategy
    --------
    1. Treat landmark 0 (wrist) as the origin → subtract it from all points.
    2. Scale by the maximum absolute value so all values lie in [-1, 1].

    Parameters
    ----------
    landmarks_flat : list of 42 floats  [x0, y0, x1, y1, ..., x20, y20]

    Returns
    -------
    list of 42 normalised floats
    """
    coords = np.array(landmarks_flat, dtype=np.float32).reshape(NUM_LANDMARKS, 2)

    # Translate: wrist (index 0) → origin
    wrist  = coords[0].copy()
    coords = coords - wrist

    # Scale: max absolute value → 1.0
    scale  = np.max(np.abs(coords))
    if scale > 0:
        coords = coords / scale

    return coords.flatten().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Per-image hybrid extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_hand_features(
    image_path: str,
    hand_detector: HandDetector,
) -> list[float] | None:
    """
    Extract 42 normalised hand-landmark features from one image.

    Parameters
    ----------
    image_path    : absolute path to a .jpg / .png image
    hand_detector : shared HandDetector instance

    Returns
    -------
    list[float] of length 42, or None if no hand was detected.
    """
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning("Cannot read image: %s", image_path)
            return None

        hand_lm, _ = hand_detector.find_landmarks(frame, draw=False)
        if hand_lm is None:
            return None

        return normalise_landmarks(hand_lm)   # 42 floats

    except Exception as exc:
        logger.error("Error processing %s: %s", image_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Column-name builder
# ─────────────────────────────────────────────────────────────────────────────

def build_column_names() -> list[str]:
    """
    Return CSV column names for the 42-feature hand landmark vector.

    Layout: label | hx0 hy0 … hx20 hy20

    Returns
    -------
    list[str] of length 43  (1 label + 42 hand features)
    """
    cols = ["label"]
    for i in range(NUM_LANDMARKS):
        cols.extend([f"hx{i}", f"hy{i}"])
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scanning
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset() -> pd.DataFrame:
    """
    Iterate over all gesture folders in DATASET_DIR and extract hybrid features.

    Returns
    -------
    pd.DataFrame
        Each row = one image's hybrid feature vector + its label.
        Columns: label, hx0, hy0, …, hx20, hy20, px0, py0, …, px16, py16
    """
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}\n"
            "Run collect_data.py first."
        )

    print("\n  Initialising detector...")
    hand_detector = HandDetector(static_image_mode=True)  # static mode for images

    rows  = []
    stats = {}

    gesture_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not gesture_dirs:
        raise RuntimeError(
            f"No gesture folders found inside {DATASET_DIR}.\n"
            "Run collect_data.py first."
        )

    logger.info("Found %d gesture classes: %s", len(gesture_dirs), gesture_dirs)
    print(f"\n{'='*60}")
    print(f"  Extracting hand features from {len(gesture_dirs)} gesture classes")
    print(f"  Feature vector: {LANDMARK_FEATURES} hand landmarks")
    print(f"{'='*60}\n")

    for gesture in gesture_dirs:
        folder      = os.path.join(DATASET_DIR, gesture)
        image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not image_files:
            logger.warning("No images found in: %s", folder)
            continue

        saved   = 0
        skipped = 0

        for img_file in tqdm(image_files, desc=f"  {gesture:<20}", unit="img", ncols=72):
            img_path = os.path.join(folder, img_file)
            features = extract_hand_features(img_path, hand_detector)

            if features is None:
                skipped += 1
                continue

            rows.append([gesture] + features)
            saved += 1

        stats[gesture] = {"saved": saved, "skipped": skipped}
        logger.info("'%s': %d saved, %d skipped", gesture, saved, skipped)

    hand_detector.close()


    if not rows:
        raise RuntimeError(
            "No features were extracted from any image.\n"
            "Check that images contain visible hands."
        )

    columns = build_column_names()
    df      = pd.DataFrame(rows, columns=columns)

    _print_stats(stats, len(df))
    return df


def _print_stats(stats: dict, total: int) -> None:
    """Print a formatted summary table after processing."""
    print(f"\n{'='*60}")
    print(f"  {'Gesture':<22} {'Saved':>8} {'Skipped':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*10}")
    for gesture, s in stats.items():
        print(f"  {gesture:<22} {s['saved']:>8} {s['skipped']:>10}")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<22} {total:>8}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Saving outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame) -> None:
    """
    Save the hybrid feature DataFrame as CSV and NumPy, and save the label map.

    Parameters
    ----------
    df : DataFrame with columns [label, hx0, hy0, …, px16, py16]
    """
    # ── CSV ───────────────────────────────────────────────────────────────────
    df.to_csv(LANDMARKS_CSV, index=False)
    logger.info("Saved landmarks CSV: %s  (%d rows)", LANDMARKS_CSV, len(df))
    print(f"  [OK] landmarks.csv saved  ({len(df)} samples, {LANDMARK_FEATURES} features)")

    # ── Label map ─────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    unique_labels = sorted(df["label"].unique().tolist())
    label_map     = {label: idx for idx, label in enumerate(unique_labels)}

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info("Saved label map: %s", LABEL_MAP_PATH)
    print(f"  [OK] label_map.json saved  ({len(label_map)} classes)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the hand landmark extraction pipeline."""
    logger.info("Starting hand landmark extraction pipeline.")
    print("\n  Sign Language Translator - Hand Feature Extraction\n")
    print("  Pipeline: MediaPipe Hands → 42 features\n")

    try:
        df = process_dataset()
        save_outputs(df)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        print(f"\n  [ERROR]: {exc}")
        sys.exit(1)

    print("\n  Preprocessing complete! Run train_model.py next.\n")
    logger.info("Preprocessing pipeline complete. %d total samples.", len(df))


if __name__ == "__main__":
    main()
