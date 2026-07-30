# =============================================================================
# preprocess.py
# Phase 3 – Landmark Extraction with MediaPipe Hands
# =============================================================================
# Usage:
#   python preprocess.py
#
# Scans every subfolder inside dataset/, treats the folder name as the gesture
# label, runs MediaPipe Hands on each image, extracts 21 hand landmarks
# (x, y per landmark = 42 features), normalises them relative to the wrist
# (landmark 0) so the model learns shape not position, and saves the result
# as both landmarks.csv (human-readable) and landmarks.npy (fast loading).
#
# Frames where no hand is detected are silently skipped.
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
)
from utils.logger import get_logger
from utils.mediapipe_helper import HandDetector

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_landmarks(landmarks_flat: list[float]) -> list[float]:
    """
    Normalise 42 raw landmark values so predictions are position-invariant.

    Strategy
    --------
    1. Treat landmark 0 (wrist) as the origin → subtract it from all points.
    2. Scale by the maximum absolute value → all values fall in [-1, 1].

    This means the same gesture is recognised whether the hand is at the top,
    bottom, left, or right of the frame.

    Parameters
    ----------
    landmarks_flat : list of 42 floats  [x0, y0, x1, y1, ..., x20, y20]

    Returns
    -------
    list of 42 normalised floats
    """
    coords = np.array(landmarks_flat, dtype=np.float32).reshape(NUM_LANDMARKS, 2)

    # Step 1: translate so wrist (index 0) is at origin
    wrist  = coords[0].copy()
    coords = coords - wrist

    # Step 2: scale so max absolute coordinate = 1.0 (avoid divide-by-zero)
    scale  = np.max(np.abs(coords))
    if scale > 0:
        coords = coords / scale

    return coords.flatten().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_image(
    image_path: str,
    detector: HandDetector,
) -> list[float] | None:
    """
    Read one image, run MediaPipe, return normalised landmarks or None.

    Parameters
    ----------
    image_path : absolute path to a .jpg / .png image
    detector   : shared HandDetector instance (avoids re-initialising per image)

    Returns
    -------
    list[float] | None
        42 normalised floats if a hand was found, otherwise None.
    """
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning("Cannot read image: %s", image_path)
            return None

        landmarks_flat, _ = detector.find_landmarks(frame, draw=False)

        if landmarks_flat is None:
            return None          # No hand detected in this image

        return normalise_landmarks(landmarks_flat)

    except Exception as exc:
        logger.error("Error processing %s: %s", image_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scanning
# ─────────────────────────────────────────────────────────────────────────────

def build_column_names() -> list[str]:
    """
    Build CSV column names: label, x0, y0, x1, y1, ..., x20, y20.

    Returns
    -------
    list[str] of length 43
    """
    cols = ["label"]
    for i in range(NUM_LANDMARKS):
        cols.extend([f"x{i}", f"y{i}"])
    return cols


def process_dataset() -> pd.DataFrame:
    """
    Iterate over all gesture folders in DATASET_DIR and extract landmarks.

    Returns
    -------
    pd.DataFrame
        Each row = one image's normalised landmarks + its label.
        Columns: label, x0, y0, ..., x20, y20
    """
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}\n"
            "Run collect_data.py first."
        )

    detector  = HandDetector()
    rows      = []
    stats     = {}   # { gesture: {"saved": int, "skipped": int} }

    # Scan subfolders – each folder = one gesture
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
    print(f"\n{'='*55}")
    print(f"  Extracting landmarks from {len(gesture_dirs)} gesture classes")
    print(f"{'='*55}\n")

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

        for img_file in tqdm(image_files, desc=f"  {gesture:<20}", unit="img", ncols=70):
            img_path   = os.path.join(folder, img_file)
            landmarks  = extract_landmarks_from_image(img_path, detector)

            if landmarks is None:
                skipped += 1
                continue

            rows.append([gesture] + landmarks)
            saved += 1

        stats[gesture] = {"saved": saved, "skipped": skipped}
        logger.info("'%s': %d saved, %d skipped", gesture, saved, skipped)

    detector.close()

    if not rows:
        raise RuntimeError(
            "No landmarks were extracted from any image.\n"
            "Check that images contain visible hands."
        )

    # Build DataFrame
    columns = build_column_names()
    df      = pd.DataFrame(rows, columns=columns)

    _print_stats(stats, len(df))
    return df


def _print_stats(stats: dict, total: int) -> None:
    """Print a formatted summary table after processing."""
    print(f"\n{'='*55}")
    print(f"  {'Gesture':<22} {'Saved':>8} {'Skipped':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*10}")
    for gesture, s in stats.items():
        print(f"  {gesture:<22} {s['saved']:>8} {s['skipped']:>10}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<22} {total:>8}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Saving outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame) -> None:
    """
    Save the landmark DataFrame as CSV and NumPy, and save the label map.

    Parameters
    ----------
    df : DataFrame with columns [label, x0, y0, ..., x20, y20]
    """
    # ── CSV ───────────────────────────────────────────────────────────────────
    df.to_csv(LANDMARKS_CSV, index=False)
    logger.info("Saved landmarks CSV: %s  (%d rows)", LANDMARKS_CSV, len(df))
    print(f"  ✅  landmarks.csv saved  ({len(df)} samples)")

    # ── NumPy ─────────────────────────────────────────────────────────────────
    npy_path = LANDMARKS_CSV.replace(".csv", ".npy")
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    np.save(npy_path, {"X": X, "y": y})
    logger.info("Saved landmarks NumPy: %s", npy_path)
    print(f"  ✅  landmarks.npy saved")

    # ── Label map (gesture → integer index) ───────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    unique_labels = sorted(df["label"].unique().tolist())
    label_map     = {label: idx for idx, label in enumerate(unique_labels)}

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info("Saved label map: %s", LABEL_MAP_PATH)
    print(f"  ✅  label_map.json saved  ({len(label_map)} classes)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full preprocessing pipeline."""
    logger.info("Starting landmark extraction pipeline.")
    print("\n🤙  Sign Language Translator – Landmark Extraction\n")

    try:
        df = process_dataset()
        save_outputs(df)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        print(f"\n❌  Error: {exc}")
        sys.exit(1)

    print("\n✅  Preprocessing complete!  Run train_model.py next.\n")
    logger.info("Preprocessing pipeline complete. %d total samples.", len(df))


if __name__ == "__main__":
    main()
