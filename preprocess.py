# =============================================================================
# preprocess.py
# Phase 3 – Hybrid Landmark Extraction (MediaPipe Hands + MoveNet Pose)
# =============================================================================
# Usage:
#   python preprocess.py
#
# For every image in dataset/ this script:
#   1. Runs MediaPipe Hands  → 42 normalised hand-landmark features (x,y × 21)
#   2. Runs MoveNet Lightning → 34 normalised body-pose features   (x,y × 17)
#   3. Concatenates both     → 76-feature hybrid vector per sample
#   4. Saves landmarks.csv and landmarks.npy
#
# Images where no hand is detected are skipped.
# Images where MoveNet finds no visible person yield zeroed pose features
# (the hand features still contribute — the sample is kept).
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
    HYBRID_FEATURES,
    LANDMARK_FEATURES,
    POSE_FEATURES,
    NUM_POSE_KEYPOINTS,
)
from utils.logger import get_logger
from utils.mediapipe_helper import HandDetector
from utils.movenet_helper   import MoveNetDetector

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

def extract_hybrid_features(
    image_path: str,
    hand_detector: HandDetector,
    pose_detector: MoveNetDetector,
) -> list[float] | None:
    """
    Extract the 76-feature hybrid vector from one image.

    Feature layout
    --------------
    [0  … 41]  42 normalised hand-landmark features  (MediaPipe Hands)
    [42 … 75]  34 normalised body-pose features       (MoveNet Lightning)

    Parameters
    ----------
    image_path     : absolute path to a .jpg / .png image
    hand_detector  : shared HandDetector instance
    pose_detector  : shared MoveNetDetector instance

    Returns
    -------
    list[float] of length 76, or None if no hand was detected.
    """
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning("Cannot read image: %s", image_path)
            return None

        # ── MediaPipe: hand landmarks ─────────────────────────────────────────
        hand_lm, _ = hand_detector.find_landmarks(frame, draw=False)

        # Skip if no hand — the hand signal is the primary classifier
        if hand_lm is None:
            return None

        hand_features = normalise_landmarks(hand_lm)   # 42 floats

        # ── MoveNet: body pose ────────────────────────────────────────────────
        pose_raw, _ = pose_detector.detect(frame, draw=False)

        if pose_raw is not None:
            pose_features = MoveNetDetector.normalise_pose(pose_raw)  # 34 floats
        else:
            # Person not visible → zero-fill pose block (hand features kept)
            pose_features = [0.0] * POSE_FEATURES

        # ── Concatenate ───────────────────────────────────────────────────────
        return hand_features + pose_features   # 76 floats

    except Exception as exc:
        logger.error("Error processing %s: %s", image_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Column-name builder
# ─────────────────────────────────────────────────────────────────────────────

def build_column_names() -> list[str]:
    """
    Return CSV column names for the hybrid feature vector.

    Layout: label | hx0 hy0 … hx20 hy20 | px0 py0 … px16 py16

    Returns
    -------
    list[str] of length 77  (1 label + 42 hand + 34 pose)
    """
    cols = ["label"]
    # Hand landmark columns  hx0, hy0 … hx20, hy20
    for i in range(NUM_LANDMARKS):
        cols.extend([f"hx{i}", f"hy{i}"])
    # Pose keypoint columns  px0, py0 … px16, py16
    for i in range(NUM_POSE_KEYPOINTS):
        cols.extend([f"px{i}", f"py{i}"])
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

    print("\n  Initialising detectors...")
    hand_detector = HandDetector(static_image_mode=True)  # static mode for images
    pose_detector = MoveNetDetector()

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
    print(f"  Extracting hybrid features from {len(gesture_dirs)} gesture classes")
    print(f"  Feature vector: {LANDMARK_FEATURES} hand + {POSE_FEATURES} pose = {HYBRID_FEATURES} total")
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
            features = extract_hybrid_features(img_path, hand_detector, pose_detector)

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
    print(f"  [OK] landmarks.csv saved  ({len(df)} samples, {HYBRID_FEATURES} features)")

    # ── NumPy ─────────────────────────────────────────────────────────────────
    npy_path     = LANDMARKS_CSV.replace(".csv", ".npy")
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    np.save(npy_path, {"X": X, "y": y})
    logger.info("Saved landmarks NumPy: %s", npy_path)
    print(f"  [OK] landmarks.npy saved  (shape: {X.shape})")

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
    """Run the full hybrid preprocessing pipeline."""
    logger.info("Starting hybrid landmark extraction pipeline.")
    print("\n  Sign Language Translator - Hybrid Feature Extraction\n")
    print("  Pipeline: MediaPipe Hands (42) + MoveNet Lightning (34) = 76 features\n")

    try:
        df = process_dataset()
        save_outputs(df)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        print(f"\n  [ERROR]: {exc}")
        sys.exit(1)

    print("\n  Preprocessing complete!  Run train_model.py next.\n")
    logger.info("Preprocessing pipeline complete. %d total samples.", len(df))


if __name__ == "__main__":
    main()
