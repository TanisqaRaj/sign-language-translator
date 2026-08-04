# =============================================================================
# collect_data.py
# Phase 2 – Gesture Image Collection Script
# =============================================================================
# Usage:
#   python collect_data.py
#
# This script opens your webcam and lets you collect training images for each
# gesture label defined in config.py.  It saves only frames where a hand is
# detected, skips blurry frames, and shows real-time progress.
#
# Controls:
#   SPACE  – pause / resume collection
#   Q      – quit early (progress is saved)
# =============================================================================

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_DIR,
    GESTURE_LABELS,
    IMAGES_PER_CLASS,
    CAMERA_INDEX,
    MIN_DETECTION_CONF,
)
from utils.logger import get_logger
from utils.mediapipe_helper import HandDetector

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BLUR_THRESHOLD   = 100.0   # Laplacian variance below this = blurry → skip
COUNTDOWN_SECS   = 3       # Seconds of countdown before capture begins
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def is_blurry(frame: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    """
    Return True if the frame is too blurry to be useful.

    Uses the variance of the Laplacian as a focus measure.
    A low variance means the image has few edges → blurry.

    Parameters
    ----------
    frame     : BGR frame from OpenCV
    threshold : variance below this value is considered blurry
    """
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance  = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def draw_overlay(
    frame: np.ndarray,
    gesture: str,
    collected: int,
    total: int,
    paused: bool,
    status_msg: str = "",
) -> np.ndarray:
    """
    Draw a heads-up display on the frame with collection progress.

    Parameters
    ----------
    frame      : BGR frame to annotate
    gesture    : current gesture label being collected
    collected  : number of images collected so far
    total      : total images required
    paused     : whether collection is currently paused
    status_msg : extra status line shown at the bottom
    """
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Gesture name
    cv2.putText(frame, f"Gesture: {gesture}", (10, 25),
                FONT, 0.8, (0, 255, 255), 2)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 10, 40, w - 20, 18
    filled = int(bar_w * collected / max(total, 1))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 200, 0), -1)
    cv2.putText(frame, f"{collected}/{total}", (bar_x + bar_w // 2 - 30, bar_y + 14),
                FONT, 0.45, (255, 255, 255), 1)

    # Pause indicator
    if paused:
        cv2.putText(frame, "PAUSED – Press SPACE to resume", (10, h - 50),
                    FONT, 0.65, (0, 165, 255), 2)

    # Bottom status
    if status_msg:
        cv2.putText(frame, status_msg, (10, h - 20),
                    FONT, 0.5, (200, 200, 200), 1)

    # Controls hint
    cv2.putText(frame, "SPACE=pause  Q=quit", (w - 220, h - 10),
                FONT, 0.45, (150, 150, 150), 1)

    return frame


def countdown(cap: cv2.VideoCapture, gesture: str, seconds: int = COUNTDOWN_SECS) -> None:
    """
    Show a countdown on the webcam feed before collection starts.

    Parameters
    ----------
    cap     : OpenCV VideoCapture object
    gesture : gesture name shown during countdown
    seconds : number of seconds to count down
    """
    import time
    for remaining in range(seconds, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            # Dark overlay
            dark = frame.copy()
            cv2.rectangle(dark, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(dark, 0.4, frame, 0.6, 0, frame)
            # Countdown number
            cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2 + 20),
                        FONT, 4, (0, 255, 0), 6)
            cv2.putText(frame, f"Get ready: {gesture}", (w // 2 - 120, h // 2 - 60),
                        FONT, 0.8, (255, 255, 0), 2)
            cv2.imshow("Sign Language – Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return


# ─────────────────────────────────────────────────────────────────────────────
# Core collection logic
# ─────────────────────────────────────────────────────────────────────────────

def select_gesture() -> str:
    """
    Prompt the user to choose a gesture from the configured label list.

    Returns
    -------
    str : selected gesture label
    """
    print("\n" + "=" * 50)
    print("  Sign Language – Data Collection Tool")
    print("=" * 50)
    print("\nAvailable gestures:\n")
    for idx, label in enumerate(GESTURE_LABELS):
        folder   = os.path.join(DATASET_DIR, label)
        existing = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        status   = f"  [{existing}/{IMAGES_PER_CLASS}]"
        print(f"  {idx:>2}. {label:<20} {status}")

    print(f"\n  {len(GESTURE_LABELS)}. Collect ALL gestures sequentially")
    print()

    while True:
        try:
            choice = input(f"Enter number (0-{len(GESTURE_LABELS)}): ").strip()
            idx    = int(choice)
            if 0 <= idx < len(GESTURE_LABELS):
                return GESTURE_LABELS[idx]
            elif idx == len(GESTURE_LABELS):
                return "__ALL__"
            else:
                print("  Invalid choice. Try again.")
        except ValueError:
            print("  Please enter a number.")


def collect_for_gesture(gesture: str, detector: HandDetector) -> int:
    """
    Open the webcam and collect images for a single gesture.

    Parameters
    ----------
    gesture  : gesture label (folder will be dataset/<gesture>/)
    detector : shared HandDetector instance

    Returns
    -------
    int : number of images successfully saved
    """
    # ── Prepare output folder ─────────────────────────────────────────────────
    save_dir = os.path.join(DATASET_DIR, gesture)
    os.makedirs(save_dir, exist_ok=True)

    # Resume from where we left off
    existing_count = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    target         = IMAGES_PER_CLASS
    collected      = existing_count

    if collected >= target:
        logger.info("'%s' already has %d images. Skipping.", gesture, collected)
        print(f"\n  '{gesture}' already complete ({collected}/{target}). Skipping.\n")
        return collected

    logger.info("Collecting images for gesture: '%s' (%d/%d existing)",
                gesture, collected, target)

    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open webcam (index %d).", CAMERA_INDEX)
        raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Countdown before we start
    countdown(cap, gesture)

    paused        = False
    skipped_blur  = 0
    skipped_hand  = 0
    pbar          = tqdm(total=target, initial=collected,
                         desc=f"  {gesture}", unit="img", ncols=70)

    print(f"\n  Collecting '{gesture}'  –  SPACE=pause  Q=quit\n")

    try:
        while collected < target:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam.")
                continue

            frame = cv2.flip(frame, 1)   # Mirror so it feels natural

            # ── Hand detection ────────────────────────────────────────────────
            # Keep a clean copy for saving — draw only on display copy
            clean_frame = frame.copy()
            landmarks, annotated = detector.find_landmarks(frame, draw=True)
            hand_detected        = landmarks is not None

            status = ""
            if not paused and hand_detected:
                if is_blurry(clean_frame):
                    status       = "Blurry frame – skipped"
                    skipped_blur += 1
                else:
                    # ── Save CLEAN image (no landmark annotations) ────────────
                    filename = os.path.join(save_dir, f"img_{collected + 1:04d}.jpg")
                    cv2.imwrite(filename, clean_frame)
                    collected += 1
                    pbar.update(1)
            elif not hand_detected:
                status = "No hand detected"
                skipped_hand += 1

            # ── Draw HUD ──────────────────────────────────────────────────────
            annotated = draw_overlay(
                annotated, gesture, collected, target, paused, status
            )
            cv2.imshow("Sign Language – Data Collection", annotated)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Collection quit early by user at %d images.", collected)
                break
            elif key == ord(" "):
                paused = not paused

    finally:
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

    logger.info(
        "Gesture '%s' done: %d saved, %d blurry skipped, %d no-hand skipped.",
        gesture, collected, skipped_blur, skipped_hand,
    )
    print(f"\n  Done: {collected} images saved for '{gesture}'")
    print(f"  Skipped – blurry: {skipped_blur} | no hand: {skipped_hand}\n")
    return collected


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point: select gesture(s) and run collection."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    choice   = select_gesture()
    detector = HandDetector()

    try:
        if choice == "__ALL__":
            for gesture in GESTURE_LABELS:
                collect_for_gesture(gesture, detector)
        else:
            collect_for_gesture(choice, detector)
    finally:
        detector.close()

    print("\n✅  Collection complete!  Run preprocess.py next.\n")


if __name__ == "__main__":
    main()
