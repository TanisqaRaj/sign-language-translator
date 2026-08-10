# =============================================================================
# preprocess_characters.py
# Character Landmark Extraction Pipeline — ISLRTC Dataset (A-Z + 0-9)
# =============================================================================
# Usage:
#   python preprocess_characters.py
#
# For every image in dataset/characters/ this script:
#   1. Runs MediaPipe Hands  → 42 normalised hand-landmark features (x,y × 21)
#   2. Saves character_landmarks.csv  (42 features + label per row)
#
# NOTE ON FEATURE COUNT:
#   The character model uses ONLY 42 hand features (no MoveNet pose).
#   Reasons:
#     • ISLRTC characters are static single-hand signs — pose adds no signal.
#     • app_cloud.py feeds 42 hand features to both models (same MediaPipe
#       pipeline, no extra MoveNet latency in the browser WebRTC loop).
#     • This perfectly matches the existing word model inference path in
#       app_cloud.py which also uses 42 features.
#   Training and inference will always use 42 features — mismatch is
#   impossible by construction.
#
# ISLRTC DATASET STRUCTURE EXPECTED:
#   dataset/characters/
#     A/       ← images of sign for "A"
#     B/
#     ...
#     Z/
#     0/
#     ...
#     9/
#
# The script auto-discovers all subdirectories — folder names become labels.
# If the ISLRTC dataset uses different casing (e.g. "a", "b") it is
# normalised to uppercase for consistency.
#
# Images where no hand is detected are skipped with a warning.
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
    CHAR_DATASET_DIR,
    CHAR_LANDMARKS_CSV,
    CHAR_LABEL_MAP_PATH,
    NUM_LANDMARKS,
    CHARACTER_LABELS,
    MODEL_DIR,
)
from utils.logger import get_logger
from utils.mediapipe_helper import HandDetector
from preprocess import normalise_landmarks   # reuse identical normalisation

logger = get_logger(__name__)

# Number of features produced by this pipeline — documented constant.
CHAR_FEATURE_COUNT = NUM_LANDMARKS * 2   # 42  (hand only, no pose)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_char_features(
    image_path: str,
    hand_detector: HandDetector,
) -> list[float] | None:
    """
    Extract the 42-feature hand-landmark vector from one image.

    Parameters
    ----------
    image_path    : absolute path to a .jpg / .png image
    hand_detector : shared HandDetector instance (static_image_mode=True)

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
            return None   # no hand detected — skip image

        return normalise_landmarks(hand_lm)   # 42 floats

    except Exception as exc:
        logger.error("Error processing %s: %s", image_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Column-name builder
# ─────────────────────────────────────────────────────────────────────────────

def build_column_names() -> list[str]:
    """
    Return CSV column names for the 42-feature character vector.

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

def _normalise_label(folder_name: str) -> str:
    """
    Normalise a folder name to a canonical label.

    ISLRTC dataset may use:
      - Uppercase:  "A", "B" ... "Z", "0" ... "9"
      - Lowercase:  "a", "b" ...
      - Mixed-case names in some forks

    We uppercase letter folders and keep digit folders as-is.
    Only labels that appear in CHARACTER_LABELS are kept.
    """
    label = folder_name.strip().upper()
    # If it is a single digit, use as-is (digits are already 0-9)
    if folder_name.isdigit() and 0 <= int(folder_name) <= 9:
        return folder_name
    return label


def discover_class_dirs(dataset_dir: str) -> dict[str, str]:
    """
    Walk dataset_dir and return {label: absolute_path} for recognised classes.

    Unrecognised folder names produce a warning and are skipped.

    Parameters
    ----------
    dataset_dir : path to dataset/characters/

    Returns
    -------
    dict mapping label string → full directory path
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Character dataset directory not found: {dataset_dir}\n"
            "Run setup_character_dataset.py and place ISLRTC images there first."
        )

    valid_labels = set(CHARACTER_LABELS)
    result: dict[str, str] = {}

    for entry in sorted(os.listdir(dataset_dir)):
        full = os.path.join(dataset_dir, entry)
        if not os.path.isdir(full):
            continue

        label = _normalise_label(entry)

        if label not in valid_labels:
            logger.warning(
                "Folder '%s' normalised to '%s' — not in CHARACTER_LABELS, skipping.",
                entry, label,
            )
            continue

        if label in result:
            logger.warning(
                "Duplicate label '%s' from folder '%s' (already have '%s'), skipping.",
                label, entry, result[label],
            )
            continue

        result[label] = full

    return result


def process_dataset() -> pd.DataFrame:
    """
    Iterate over all character folders and extract hand-landmark features.

    Returns
    -------
    pd.DataFrame
        Each row = one image's 42-feature vector + its label.
        Columns: label, hx0, hy0, …, hx20, hy20
    """
    print("\n  Sign Language Translator — Character Landmark Extraction\n")
    print("  Pipeline: MediaPipe Hands only → 42 features per image\n")
    print(f"  Dataset : {CHAR_DATASET_DIR}\n")

    class_dirs = discover_class_dirs(CHAR_DATASET_DIR)

    if not class_dirs:
        raise RuntimeError(
            "No valid character class folders found.\n"
            f"Expected folders named A–Z and 0–9 inside: {CHAR_DATASET_DIR}"
        )

    print(f"{'='*60}")
    print(f"  Found {len(class_dirs)} character classes: {sorted(class_dirs.keys())}")
    print(f"{'='*60}\n")

    hand_detector = HandDetector(static_image_mode=True)

    rows: list[list] = []
    stats: dict[str, dict] = {}

    for label in sorted(class_dirs.keys()):
        folder = class_dirs[label]
        image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not image_files:
            logger.warning("No images found in: %s", folder)
            stats[label] = {"saved": 0, "skipped": 0}
            continue

        saved   = 0
        skipped = 0

        for img_file in tqdm(image_files, desc=f"  {label:<6}", unit="img", ncols=72):
            img_path = os.path.join(folder, img_file)
            features = extract_char_features(img_path, hand_detector)

            if features is None:
                skipped += 1
                continue

            rows.append([label] + features)
            saved += 1

        stats[label] = {"saved": saved, "skipped": skipped}
        logger.info("'%s': %d saved, %d skipped", label, saved, skipped)

    hand_detector.close()

    if not rows:
        raise RuntimeError(
            "No features extracted from any character image.\n"
            "Check that images contain clearly visible hands."
        )

    columns = build_column_names()
    df      = pd.DataFrame(rows, columns=columns)

    _print_stats(stats, len(df))
    return df


def _print_stats(stats: dict, total: int) -> None:
    """Print a formatted summary after processing."""
    print(f"\n{'='*60}")
    print(f"  {'Class':<8} {'Saved':>8} {'Skipped':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10}")
    for label in sorted(stats.keys()):
        s = stats[label]
        print(f"  {label:<8} {s['saved']:>8} {s['skipped']:>10}")
    print(f"  {'-'*28}")
    print(f"  {'TOTAL':<8} {total:>8}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Saving outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame) -> None:
    """
    Save the character feature DataFrame as CSV and save the label map.

    Feature count validation is performed before saving to catch any
    accidental shape mismatches early.

    Parameters
    ----------
    df : DataFrame with columns [label, hx0, hy0, …, hx20, hy20]
    """
    # ── Validate feature count ────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) != CHAR_FEATURE_COUNT:
        raise ValueError(
            f"Feature count mismatch!\n"
            f"  Expected : {CHAR_FEATURE_COUNT}\n"
            f"  Got      : {len(feature_cols)}\n"
            "This should never happen — file a bug report."
        )

    # ── CSV ───────────────────────────────────────────────────────────────────
    df.to_csv(CHAR_LANDMARKS_CSV, index=False)
    logger.info("Saved character CSV: %s  (%d rows)", CHAR_LANDMARKS_CSV, len(df))
    print(f"  [OK] character_landmarks.csv saved  ({len(df)} samples, {CHAR_FEATURE_COUNT} features)")

    # ── Label map ─────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    unique_labels = sorted(df["label"].unique().tolist())
    label_map     = {label: idx for idx, label in enumerate(unique_labels)}

    with open(CHAR_LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info("Saved character label map: %s", CHAR_LABEL_MAP_PATH)
    print(f"  [OK] character_label_map.json saved  ({len(label_map)} classes)")
    print(f"  Classes : {sorted(label_map.keys())}")

    # ── Class distribution summary ────────────────────────────────────────────
    print("\n  Class distribution:")
    for label, count in sorted(df["label"].value_counts().items()):
        print(f"    {label:<4}: {count} samples")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full character preprocessing pipeline."""
    logger.info("Starting character landmark extraction pipeline.")
    print("\n" + "=" * 60)
    print("  Character Preprocessing Pipeline")
    print("  Feature count: 42 (MediaPipe hand landmarks only)")
    print("  No MoveNet pose — characters are static hand signs")
    print("=" * 60)

    try:
        df = process_dataset()
        save_outputs(df)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        print(f"\n  [ERROR]: {exc}")
        sys.exit(1)

    print("\n  Preprocessing complete!  Run train_character_model.py next.\n")
    logger.info(
        "Character preprocessing pipeline complete. %d total samples.", len(df)
    )


if __name__ == "__main__":
    main()
