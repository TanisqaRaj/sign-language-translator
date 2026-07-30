# =============================================================================
# inference.py
# Phase 6 – Real-Time Inference Pipeline (Standalone OpenCV Window)
# =============================================================================
# Usage:
#   python inference.py
#
# Opens the webcam, detects the hand with MediaPipe, extracts landmarks,
# runs the TFLite model, smooths predictions over a rolling buffer, and
# overlays the recognised gesture + confidence score on the live feed.
#
# Controls:
#   Q – quit
#   C – clear the sentence buffer
#   S – speak the current sentence (pyttsx3)
# =============================================================================

import os
import sys
import json
import pickle
import collections
import time

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    STABLE_FRAME_COUNT,
    PREDICTION_BUFFER_LEN,
    MODEL_TFLITE_PATH,
    LABEL_MAP_PATH,
    MODEL_DIR,
    LANDMARK_FEATURES,
    SCALER_PATH,
)
from utils.logger           import get_logger
from utils.mediapipe_helper import HandDetector
from utils.tts_engine       import TTSEngine
from preprocess             import normalise_landmarks

logger = get_logger(__name__)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 200,   0)
RED   = (0,   0, 200)
CYAN  = (255, 200,  0)
WHITE = (255, 255, 255)
BLACK = (  0,   0,   0)


# ─────────────────────────────────────────────────────────────────────────────
# Model & label-map loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tflite_model() -> tuple:
    """
    Load the TFLite interpreter and return (interpreter, input_idx, output_idx).

    Raises
    ------
    FileNotFoundError : if the .tflite model file is missing.
    """
    import tensorflow as tf

    if not os.path.exists(MODEL_TFLITE_PATH):
        raise FileNotFoundError(
            f"TFLite model not found: {MODEL_TFLITE_PATH}\n"
            "Run convert_tflite.py first."
        )

    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()

    input_idx  = interpreter.get_input_details()[0]["index"]
    output_idx = interpreter.get_output_details()[0]["index"]

    logger.info("TFLite model loaded: %s", MODEL_TFLITE_PATH)
    return interpreter, input_idx, output_idx


def load_label_map() -> dict[int, str]:
    """
    Load the integer → gesture-name mapping from label_map.json.

    label_map.json is saved as {"Hello": 0, "Yes": 1, …} (str → int).
    This function inverts it to {0: "Hello", 1: "Yes", …} (int → str)
    for fast lookup during inference.

    Returns
    -------
    dict mapping class index (int) to gesture name (str).
    """
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Label map not found: {LABEL_MAP_PATH}\n"
            "Run preprocess.py and train_model.py first."
        )
    with open(LABEL_MAP_PATH) as f:
        raw = json.load(f)          # { "Hello": 0, "Yes": 1, … }
    # Invert: { 0: "Hello", 1: "Yes", … }
    inv = {int(v): k for k, v in raw.items()}
    logger.info("Label map loaded (%d classes).", len(inv))
    return inv


def load_scaler():
    """
    Load the StandardScaler fitted during training.

    Returns None if the scaler file does not exist (older models may not
    have one — inference still works, just without scaling).
    """
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded: %s", SCALER_PATH)
        return scaler
    logger.warning("Scaler not found at %s — running without feature scaling.", SCALER_PATH)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    landmarks_flat: list[float],
    interpreter,
    input_idx: int,
    output_idx: int,
    label_map: dict[int, str],
    scaler,
) -> tuple[str, float]:
    """
    Run one forward pass through the TFLite model.

    Parameters
    ----------
    landmarks_flat : 42 normalised floats from HandDetector
    interpreter    : TFLite interpreter
    input_idx      : index of the input tensor
    output_idx     : index of the output tensor
    label_map      : {class_int: gesture_name}
    scaler         : fitted StandardScaler or None

    Returns
    -------
    (gesture_name, confidence)  –  best predicted class and its probability
    """
    features = np.array(landmarks_flat, dtype=np.float32).reshape(1, -1)

    # Apply the same scaling used during training
    if scaler is not None:
        features = scaler.transform(features).astype(np.float32)

    interpreter.set_tensor(input_idx, features)
    interpreter.invoke()

    probabilities   = interpreter.get_tensor(output_idx)[0]
    class_idx       = int(np.argmax(probabilities))
    confidence      = float(probabilities[class_idx])
    gesture_name    = label_map.get(class_idx, f"Class_{class_idx}")

    return gesture_name, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Prediction smoothing
# ─────────────────────────────────────────────────────────────────────────────

class PredictionSmoother:
    """
    Rolling majority-vote smoother to eliminate flickering predictions.

    Keeps a deque of the last N predictions and returns the most common one.
    A prediction is only accepted if it appears in at least half the buffer
    AND its confidence exceeds CONFIDENCE_THRESHOLD.
    """

    def __init__(self, buffer_len: int = PREDICTION_BUFFER_LEN) -> None:
        self._buffer = collections.deque(maxlen=buffer_len)

    def update(self, gesture: str, confidence: float) -> tuple[str | None, float]:
        """
        Add a new prediction and return the smoothed result.

        Returns
        -------
        (smoothed_gesture, confidence)
            smoothed_gesture is None if confidence is too low or no majority.
        """
        if confidence < CONFIDENCE_THRESHOLD:
            self._buffer.append(None)
            return None, confidence

        self._buffer.append(gesture)

        # Majority vote
        counts  = collections.Counter(g for g in self._buffer if g is not None)
        if not counts:
            return None, confidence

        best, votes = counts.most_common(1)[0]
        # Require votes > half of the full buffer capacity (not current length)
        # so a single prediction after reset cannot win the majority.
        required = self._buffer.maxlen // 2
        if votes > required:
            return best, confidence
        return None, confidence

    def reset(self) -> None:
        """Clear the prediction buffer."""
        self._buffer.clear()


# ─────────────────────────────────────────────────────────────────────────────
# HUD drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    gesture: str | None,
    confidence: float,
    sentence: list[str],
    fps: float,
    stable_frames: int,
) -> np.ndarray:
    """
    Render the heads-up display over the camera frame.

    Parameters
    ----------
    frame         : BGR frame
    gesture       : current smoothed prediction (None = no detection)
    confidence    : prediction confidence 0-1
    sentence      : list of confirmed words added to the sentence buffer
    fps           : current frames per second
    stable_frames : how many consecutive frames the gesture has been stable
    """
    h, w = frame.shape[:2]

    # ── Top panel ─────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if gesture:
        color = GREEN
        label_text = f"Gesture: {gesture}"
        conf_text  = f"Confidence: {confidence:.0%}"
        stable_pct = min(stable_frames / STABLE_FRAME_COUNT, 1.0)
    else:
        color      = RED
        label_text = "No gesture detected"
        conf_text  = ""
        stable_pct = 0.0

    cv2.putText(frame, label_text, (10, 30),  FONT, 0.9, color, 2)
    if conf_text:
        cv2.putText(frame, conf_text, (10, 58), FONT, 0.65, WHITE, 1)

    # Stability progress bar
    bar_w = w - 20
    cv2.rectangle(frame, (10, 68), (10 + bar_w, 82), (60, 60, 60), -1)
    fill  = int(bar_w * stable_pct)
    bar_c = (0, int(200 * stable_pct), int(200 * (1 - stable_pct)))
    if fill > 0:
        cv2.rectangle(frame, (10, 68), (10 + fill, 82), bar_c, -1)

    # ── FPS counter (top-right) ───────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 25), FONT, 0.6, CYAN, 1)

    # ── Sentence panel (bottom) ───────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 60), (w, h), BLACK, -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

    sentence_text = " ".join(sentence) if sentence else "—"
    cv2.putText(frame, f"Sentence: {sentence_text}", (10, h - 35),
                FONT, 0.65, (220, 220, 220), 1)
    cv2.putText(frame, "Q=quit  C=clear  S=speak", (10, h - 12),
                FONT, 0.45, (140, 140, 140), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference() -> None:
    """
    Open the webcam and run the full real-time inference loop.

    Flow per frame
    --------------
    1. Capture frame → flip (mirror)
    2. Detect hand with MediaPipe
    3. Normalise landmarks
    4. TFLite forward pass
    5. Smooth prediction via rolling buffer
    6. Track stable frames → add word to sentence when stable
    7. Draw HUD
    8. Handle key presses
    """
    # ── Load resources ────────────────────────────────────────────────────────
    logger.info("Initialising inference pipeline…")
    print("\n🤙  Sign Language Translator – Real-Time Inference\n")

    interpreter, input_idx, output_idx = load_tflite_model()
    label_map = load_label_map()
    scaler    = load_scaler()
    detector  = HandDetector()
    smoother  = PredictionSmoother()
    tts       = TTSEngine()

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    sentence       = []       # Words added to the running sentence
    stable_frames  = 0        # Consecutive frames with same gesture
    last_stable    = None     # The gesture currently being held

    # FPS calculation
    fps      = 0.0
    t_prev   = time.perf_counter()

    logger.info("Inference loop started.")
    print("  Press Q to quit | C to clear sentence | S to speak\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Webcam read failed.")
                continue

            frame = cv2.flip(frame, 1)

            # ── Hand detection + landmarks ────────────────────────────────────
            raw_lm, annotated = detector.find_landmarks(frame, draw=True)

            gesture    = None
            confidence = 0.0

            if raw_lm is not None:
                norm_lm          = normalise_landmarks(raw_lm)
                raw_pred, conf   = predict(norm_lm, interpreter, input_idx,
                                           output_idx, label_map, scaler)
                gesture, confidence = smoother.update(raw_pred, conf)
            else:
                smoother.reset()

            # ── Stability tracking → sentence building ────────────────────────
            if gesture is not None:
                if gesture == last_stable:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    last_stable   = gesture

                # Add to sentence when gesture held for STABLE_FRAME_COUNT frames
                if stable_frames == STABLE_FRAME_COUNT:
                    if not sentence or sentence[-1] != gesture:
                        sentence.append(gesture)
                        logger.info("Word added to sentence: '%s'", gesture)
            else:
                stable_frames = 0
                last_stable   = None

            # ── FPS ───────────────────────────────────────────────────────────
            t_now  = time.perf_counter()
            fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # ── Draw HUD ──────────────────────────────────────────────────────
            annotated = draw_hud(annotated, gesture, confidence,
                                 sentence, fps, stable_frames)
            cv2.imshow("Sign Language Translator – Press Q to quit", annotated)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                sentence = []
                smoother.reset()
                tts.reset_last_spoken()
                logger.info("Sentence cleared.")
            elif key == ord("s"):
                if sentence:
                    text = " ".join(sentence)
                    tts.speak(text, force=True)
                    logger.info("Speaking (forced): '%s'", text)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.info("Inference pipeline stopped.")
        print("\n  Inference stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point with top-level error handling."""
    try:
        run_inference()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(f"\n❌  {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error in inference pipeline.")
        print(f"\n❌  Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
