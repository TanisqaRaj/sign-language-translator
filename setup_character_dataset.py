# =============================================================================
# setup_character_dataset.py
# One-time setup helper for the ISLRTC character dataset
# =============================================================================
# Usage:
#   python setup_character_dataset.py
#
# This script:
#   1. Creates dataset/characters/<label>/ for every class in CHARACTER_LABELS
#      (A–Z and 0–9, 36 folders total).
#   2. Prints instructions for placing the Kaggle dataset images.
#   3. Counts existing images in each folder and shows a readiness summary.
#   4. Optionally verifies that MediaPipe can detect a hand in a sample image
#      from each class (--verify flag).
#
# Kaggle dataset
# ──────────────
# Dataset : "Indian Sign Language ISLRTC referred"
# URL     : https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred
#
# Download and extract the dataset, then copy (or move) the class folders into:
#   dataset/characters/
#
# Expected structure after placement:
#   dataset/characters/
#     A/   ← images showing the ISL sign for letter A
#     B/
#     ...
#     Z/
#     0/
#     1/
#     ...
#     9/
#
# The Kaggle dataset may use uppercase folder names (A, B, …) or lowercase
# (a, b, …). Both are handled — preprocess_characters.py normalises them.
#
# After this script succeeds, run the full character pipeline:
#   python preprocess_characters.py
#   python train_character_model.py
#   python convert_character_tflite.py
#   streamlit run app_cloud.py
# =============================================================================

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHAR_DATASET_DIR, CHARACTER_LABELS, IMAGES_PER_CLASS
from utils.logger import get_logger

logger = get_logger(__name__)

PASS = "  ✅"
WARN = "  ⚠️ "
FAIL = "  ❌"
INFO = "  ℹ️ "

MIN_IMAGES_PER_CLASS = 50   # warn if fewer than this
TARGET_IMAGES        = IMAGES_PER_CLASS   # ideal target (from config)


# ─────────────────────────────────────────────────────────────────────────────
# Folder creation
# ─────────────────────────────────────────────────────────────────────────────

def create_class_folders() -> None:
    """
    Create dataset/characters/<label>/ for all 36 character classes.

    Safe to run multiple times — existing folders are not modified.
    """
    os.makedirs(CHAR_DATASET_DIR, exist_ok=True)
    created = 0
    existed = 0

    for label in CHARACTER_LABELS:
        folder = os.path.join(CHAR_DATASET_DIR, label)
        if not os.path.isdir(folder):
            os.makedirs(folder)
            created += 1
        else:
            existed += 1

    if created:
        print(f"{PASS} Created {created} new class folder(s) in: {CHAR_DATASET_DIR}")
    if existed:
        print(f"{INFO} {existed} folder(s) already existed (untouched).")


# ─────────────────────────────────────────────────────────────────────────────
# Image count summary
# ─────────────────────────────────────────────────────────────────────────────

def count_images() -> dict[str, int]:
    """
    Count .jpg / .jpeg / .png images in each class folder.

    Returns
    -------
    dict mapping label → image count
    """
    counts: dict[str, int] = {}
    for label in CHARACTER_LABELS:
        folder = os.path.join(CHAR_DATASET_DIR, label)
        if os.path.isdir(folder):
            imgs = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            counts[label] = len(imgs)
        else:
            counts[label] = 0
    return counts


def print_image_summary(counts: dict[str, int]) -> bool:
    """
    Print a formatted image-count table and return True if all classes have
    enough images to proceed with preprocessing.
    """
    total = sum(counts.values())
    empty = [l for l, c in counts.items() if c == 0]
    sparse = [l for l, c in counts.items() if 0 < c < MIN_IMAGES_PER_CLASS]
    ready  = [l for l, c in counts.items() if c >= MIN_IMAGES_PER_CLASS]

    print(f"\n{'='*64}")
    print(f"  {'Class':<6} {'Images':>8}  Status")
    print(f"  {'-'*6} {'-'*8}  {'-'*20}")

    for label in CHARACTER_LABELS:
        cnt = counts[label]
        if cnt == 0:
            icon = FAIL.strip()
            status = "Empty — add images"
        elif cnt < MIN_IMAGES_PER_CLASS:
            icon = WARN.strip()
            status = f"Sparse (need ≥{MIN_IMAGES_PER_CLASS})"
        elif cnt < TARGET_IMAGES:
            icon = WARN.strip()
            status = f"OK but low (target: {TARGET_IMAGES})"
        else:
            icon = PASS.strip()
            status = "Ready"
        print(f"  {label:<6} {cnt:>8}  {icon} {status}")

    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<6} {total:>8}")
    print(f"{'='*64}")
    print(f"\n  Ready   : {len(ready)}/{len(CHARACTER_LABELS)} classes")
    print(f"  Sparse  : {len(sparse)} classes")
    print(f"  Empty   : {len(empty)} classes")

    all_ready = len(empty) == 0 and len(sparse) == 0
    return all_ready


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe hand detection spot-check (optional)
# ─────────────────────────────────────────────────────────────────────────────

def verify_hand_detection(sample_per_class: int = 3) -> None:
    """
    Spot-check that MediaPipe can detect a hand in sample images.

    Reads up to `sample_per_class` images from each class folder and
    attempts MediaPipe hand detection.  Reports detection rate per class.

    This is optional — run with:  python setup_character_dataset.py --verify
    """
    import cv2
    from utils.mediapipe_helper import HandDetector

    print("\n  Running MediaPipe hand detection spot-check…")
    print(f"  Sampling {sample_per_class} image(s) per class.\n")

    detector = HandDetector(static_image_mode=True)
    results: dict[str, tuple[int, int]] = {}   # label → (detected, total)

    for label in CHARACTER_LABELS:
        folder = os.path.join(CHAR_DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:sample_per_class]

        detected = 0
        for img_file in images:
            path  = os.path.join(folder, img_file)
            frame = cv2.imread(path)
            if frame is None:
                continue
            lm, _ = detector.find_landmarks(frame, draw=False)
            if lm is not None:
                detected += 1

        results[label] = (detected, len(images))

    detector.close()

    print(f"  {'Class':<6}  Detection rate")
    print(f"  {'-'*6}  {'-'*16}")
    low_detection = []
    for label, (det, tot) in results.items():
        if tot == 0:
            continue
        rate = det / tot
        icon = PASS.strip() if rate >= 0.5 else WARN.strip()
        print(f"  {label:<6}  {icon} {det}/{tot} ({rate:.0%})")
        if rate < 0.5:
            low_detection.append(label)

    if low_detection:
        print(f"\n{WARN} Low detection rate for: {low_detection}")
        print("  → Check image quality, lighting, or hand visibility.")
    else:
        print(f"\n{PASS} All sampled classes passed hand detection check.")


# ─────────────────────────────────────────────────────────────────────────────
# Instructions
# ─────────────────────────────────────────────────────────────────────────────

def print_instructions(all_ready: bool) -> None:
    """Print next-step instructions based on dataset readiness."""
    print()
    if all_ready:
        print(f"{PASS} Dataset looks ready!  Run the pipeline:\n")
        print("    python preprocess_characters.py")
        print("    python train_character_model.py")
        print("    python convert_character_tflite.py")
        print("    streamlit run app_cloud.py\n")
    else:
        print(f"{WARN} Dataset is incomplete.  To add images:\n")
        print("  Option A — Kaggle dataset (recommended):")
        print("    1. Download from:")
        print("       https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred")
        print("    2. Extract the zip file.")
        print(f"    3. Copy each class folder (A/, B/, …, Z/, 0/, …, 9/) into:")
        print(f"       {CHAR_DATASET_DIR}")
        print()
        print("  Option B — Collect your own images:")
        print("    Run: python collect_data.py")
        print("    (Select character classes instead of word gestures)")
        print()
        print("  After adding images, re-run this script to verify:")
        print("    python setup_character_dataset.py\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup and verify the ISLRTC character dataset folder structure."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a MediaPipe hand-detection spot-check on sample images.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        metavar="N",
        help="Number of images to sample per class during --verify (default: 3).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("  Sign Language Translator — Character Dataset Setup")
    print(f"  Target directory: {CHAR_DATASET_DIR}")
    print(f"  Classes: {len(CHARACTER_LABELS)} (A–Z + 0–9)")
    print("=" * 64 + "\n")

    # Step 1: Create folders
    print("[Step 1] Creating class folders…")
    create_class_folders()

    # Step 2: Count images
    print("\n[Step 2] Counting existing images…")
    counts    = count_images()
    all_ready = print_image_summary(counts)

    # Step 3 (optional): MediaPipe spot-check
    if args.verify:
        print("\n[Step 3] MediaPipe hand detection spot-check…")
        try:
            verify_hand_detection(sample_per_class=args.sample)
        except Exception as exc:
            print(f"{WARN} Spot-check failed: {exc}")

    # Instructions
    print_instructions(all_ready)

    # Check word dataset is untouched
    from config import DATASET_DIR, GESTURE_LABELS
    print("[Safety check] Word dataset integrity…")
    word_ok = True
    for gesture in GESTURE_LABELS:
        folder = os.path.join(DATASET_DIR, gesture)
        if os.path.isdir(folder):
            cnt = len([
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            if cnt > 0:
                print(f"{PASS}  Word class '{gesture}': {cnt} images intact")
            else:
                print(f"{WARN}  Word class '{gesture}': folder exists but no images")
                word_ok = False
        else:
            print(f"{WARN}  Word class '{gesture}': folder missing (run collect_data.py)")
            word_ok = False

    if word_ok:
        print(f"\n{PASS} All word gesture images are intact — word model unaffected.\n")
    else:
        print(f"\n{WARN} Some word gesture folders are empty. "
              "Run collect_data.py to repopulate.\n")


if __name__ == "__main__":
    main()
