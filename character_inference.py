# =============================================================================
# character_inference.py
# Standalone Character Inference Module — ISLRTC A-Z + 0-9
# =============================================================================
# Usage (standalone OpenCV window):
#   python character_inference.py
#
# This module is the public interface for the character model (Model 2).
# It re-exports the four character-specific functions from inference.py
# so that external scripts can import from a single focused module:
#
#   from character_inference import (
#       load_character_tflite_model,
#       load_character_label_map,
#       load_character_scaler,
#       predict_character,
#   )
#
# The implementation stays in inference.py to keep a single source of truth.
# This file adds a standalone __main__ entry point for quick testing.
#
# Features
# --------
# • Completely independent from the word model (Model 1).
# • Uses models/character_model.tflite, character_scaler.pkl,
#   character_label_map.json — never touches word model artefacts.
# • Input: 42 normalised MediaPipe hand-landmark features (no MoveNet needed).
# • Output: one of 36 characters (A-Z + 0-9) or "Unknown" if below threshold.
# • Configurable debounce so repeated frames don't flood the sentence.
#
# Controls (standalone window)
# ----------------------------
#   Q – quit
#   Space – add a space to the character sentence
#   Backspace – remove last character
#   C – clear the entire character sentence
#   S – speak the current character sentence via pyttsx3 (local only)
# =============================================================================

import os
import sys
import time
import collections

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Re-export the four character functions from inference.py ──────────────────
# inference.py is the single source of truth; we delegate to it.
from inference import (
    load_character_tflite_model,
    load_character_label_map,
    load_character_scaler,
    predict_character,
)

from config import (
    CAMERA_INDEX,
    CHARACTER_CONFIDENCE_THRESHOLD,
    CHARACTER_STABLE_FRAME_COUNT,
    CHARACTER_DEBOUNCE_FRAMES,
    LANDMARK_FEATURES,
)
from utils.logger import get_logger
from utils.mediapipe_helper import HandDetector
from preprocess import normalise_landmarks

# Make the four functions part of the public API of this module
__all__ = [
    "load_character_tflite_model",
    "load_character_label_map",
    "load_character_scaler",
    "predict_character",
    "run_character_inference",
]

logger = get_logger(__name__)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 200,   0)
RED   = (0,   0, 200)
AMBER = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (  0,   0,   0)
GOLD  = (  0, 215, 255)


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hud(
    frame: np.ndarray,
    char_pred: str | None,
    confidence: float,
    char_sentence: list[str],
    fps: float,
    stable_frames: int,
    debounce_ready: bool,
) -> np.ndarray:
    """
    Render the character-mode HUD on the frame.

    Top panel   : current predicted character + confidence + stability bar
    Bottom panel: assembled character sentence
    """
    h, w = frame.shape[:2]

    # Top banner
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 100), BLACK, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    color = GREEN if char_pred else RED
    label = char_pred if char_pred else "No hand detected"
    cv2.putText(frame, f"Char: {label}", (10, 32),  FONT, 1.0, color, 2)

    if char_pred:
        conf_text = f"Confidence: {confidence:.0%}"
        cv2.putText(frame, conf_text, (10, 60), FONT, 0.65, WHITE, 1)

    # Debounce-ready dot (green = can accept, amber = cooling down)
    dot_color = GREEN if debounce_ready else AMBER
    cv2.circle(frame, (w - 20, 18), 7, dot_color, -1)
    cv2.putText(frame, "Ready" if debounce_ready else "Wait",
                (w - 75, 22), FONT, 0.42, dot_color, 1)

    # Stability bar
    stable_pct = min(stable_frames / max(CHARACTER_STABLE_FRAME_COUNT, 1), 1.0)
    bar_w = w - 24
    cv2.rectangle(frame, (12, 72), (12 + bar_w, 84), (50, 50, 50), -1)
    fill_w = int(bar_w * stable_pct)
    if fill_w > 0:
        bar_col = (0, int(200 * stable_pct), int(200 * (1 - stable_pct)))
        cv2.rectangle(frame, (12, 72), (12 + fill_w, 84), bar_col, -1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 58), FONT, 0.55, GOLD, 1)

    # Bottom sentence strip
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 64), (w, h), BLACK, -1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)

    sentence_txt = "".join(char_sentence) if char_sentence else "—"
    cv2.putText(frame, f"Text: {sentence_txt}", (10, h - 38),
                FONT, 0.65, (220, 220, 220), 1)
    cv2.putText(frame, "Q=quit  Space=space  Bksp=backspace  C=clear  S=speak",
                (10, h - 12), FONT, 0.40, (120, 120, 120), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Standalone inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_character_inference() -> None:
    """
    Open the webcam and run the character recognition loop.

    Feature path
    ────────────
    webcam frame → MediaPipe (42 hand landmarks) → normalise → character
    StandardScaler → TFLite DNN → top-1 class → debounce → sentence append

    Both the character TFLite model and the character scaler are loaded
    independently from the word model artefacts.  No word model files
    are touched.

    Debounce logic
    ──────────────
    A character is only appended when:
      1. The same character appears in CHARACTER_STABLE_FRAME_COUNT
         consecutive frames AND
      2. CHARACTER_DEBOUNCE_FRAMES frames have elapsed since the last
         accepted character (forces the user to lower their hand briefly
         between letters, preventing HHHEELLLOO repeats).

    Unknown handling
    ────────────────
    Predictions below CHARACTER_CONFIDENCE_THRESHOLD are reported as
    "Unknown" and are never appended to the sentence.
    """
    logger.info("Starting character inference pipeline…")
    print("\n🔤  Character Inference — ISLRTC A-Z + 0-9")
    print(f"   Model  : character_model.tflite")
    print(f"   Input  : {LANDMARK_FEATURES} hand-landmark features (MediaPipe only)")
    print(f"   Classes: A-Z + 0-9 (36 total)\n")

    # ── Load character model resources ────────────────────────────────────────
    interpreter, input_idx, output_idx = load_character_tflite_model()
    label_map = load_character_label_map()
    char_scaler = load_character_scaler()

    # ── Detector ──────────────────────────────────────────────────────────────
    hand_detector = HandDetector()

    # ── TTS (local only) ──────────────────────────────────────────────────────
    try:
        from utils.tts_engine import TTSEngine
        tts = TTSEngine()
    except Exception:
        tts = None

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # ── State ─────────────────────────────────────────────────────────────────
    char_sentence: list[str] = []
    stable_frames   = 0
    last_stable     = None
    debounce_count  = 0       # frames since last accepted character
    last_accepted   = None    # last character written to sentence

    pred_buffer = collections.deque(maxlen=10)

    fps    = 0.0
    t_prev = time.perf_counter()

    print("  Press Q to quit | Space = space | Bksp = backspace | C = clear | S = speak\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # ── Feature extraction ────────────────────────────────────────────
            raw_lm, _ = hand_detector.find_landmarks(frame, draw=True)

            char_pred  = None
            confidence = 0.0

            if raw_lm is not None:
                features = normalise_landmarks(raw_lm)   # 42 floats

                raw_char, conf = predict_character(
                    features, interpreter, input_idx, output_idx,
                    label_map, char_scaler,
                )

                if conf >= CHARACTER_CONFIDENCE_THRESHOLD:
                    # Rolling majority-vote smoother
                    pred_buffer.append(raw_char)
                    counter   = collections.Counter(pred_buffer)
                    best, cnt = counter.most_common(1)[0]
                    if cnt > len(pred_buffer) // 2:
                        char_pred  = best
                        confidence = conf
                else:
                    pred_buffer.append(None)
            else:
                pred_buffer.clear()

            # ── Stability + debounce tracking ─────────────────────────────────
            if char_pred is not None:
                if char_pred == last_stable:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    last_stable   = char_pred

                if stable_frames >= CHARACTER_STABLE_FRAME_COUNT:
                    if debounce_count >= CHARACTER_DEBOUNCE_FRAMES:
                        if last_accepted != char_pred:
                            char_sentence.append(char_pred)
                            last_accepted  = char_pred
                            logger.info("Character accepted: '%s'", char_pred)
                        debounce_count = 0
                    else:
                        debounce_count += 1
                else:
                    debounce_count += 1
            else:
                stable_frames = 0
                last_stable   = None
                debounce_count = min(debounce_count + 1, CHARACTER_DEBOUNCE_FRAMES + 5)

            debounce_ready = debounce_count >= CHARACTER_DEBOUNCE_FRAMES

            # ── FPS ───────────────────────────────────────────────────────────
            t_now  = time.perf_counter()
            fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # ── HUD ───────────────────────────────────────────────────────────
            frame = _draw_hud(
                frame, char_pred, confidence,
                char_sentence, fps, stable_frames, debounce_ready,
            )
            cv2.imshow("Character Inference — Press Q to quit", frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if char_sentence and char_sentence[-1] != " ":
                    char_sentence.append(" ")
                    last_accepted = " "
                    logger.info("Space added.")
            elif key == 8:  # Backspace
                if char_sentence:
                    removed = char_sentence.pop()
                    last_accepted = char_sentence[-1] if char_sentence else None
                    logger.info("Backspace: removed '%s'", removed)
            elif key == ord("c"):
                char_sentence.clear()
                last_accepted  = None
                debounce_count = 0
                pred_buffer.clear()
                logger.info("Sentence cleared.")
            elif key == ord("s"):
                text = "".join(char_sentence).strip()
                if text and tts:
                    tts.speak(text, force=True)
                    logger.info("Speaking: '%s'", text)
                elif text:
                    print(f"  Sentence: {text!r}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        logger.info("Character inference stopped.")
        print("\n  Inference stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point with top-level error handling."""
    try:
        run_character_inference()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(f"\n❌  {exc}")
        sys.exit(1)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        print(f"\n❌  {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error in character inference.")
        print(f"\n❌  Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
