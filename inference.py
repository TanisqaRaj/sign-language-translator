# =============================================================================
# inference.py
# Phase 6 – Real-Time Hybrid Inference Pipeline (Standalone OpenCV Window)
# =============================================================================
# Usage:
#   python inference.py
#
# Opens the webcam, extracts the 76-feature hybrid vector per frame
# (42 MediaPipe hand landmarks + 34 MoveNet pose keypoints), runs the
# TFLite model, smooths predictions over a rolling buffer, and overlays
# the recognised gesture + confidence score on the live feed.
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
    HYBRID_FEATURES,
    LANDMARK_FEATURES,
    POSE_FEATURES,
    SCALER_PATH,
    # ── Character model paths (Model 2) ────────────────────────────────────────
    CHAR_MODEL_TFLITE_PATH,
    CHAR_LABEL_MAP_PATH,
    CHAR_SCALER_PATH,
    CHARACTER_CONFIDENCE_THRESHOLD,
)
from utils.logger           import get_logger
from utils.mediapipe_helper import HandDetector
from utils.movenet_helper   import MoveNetDetector
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

    Supports both:
      - tflite-runtime (lightweight, used in Docker/cloud via requirements_cloud.txt)
      - Full TensorFlow (used in local development via requirements.txt)

    Raises
    ------
    FileNotFoundError : if the .tflite model file is missing.
    """
    # Try lightweight tflite-runtime first (Docker/cloud), fall back to full TF
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    if not os.path.exists(MODEL_TFLITE_PATH):
        raise FileNotFoundError(
            f"TFLite model not found: {MODEL_TFLITE_PATH}\n"
            "Run convert_tflite.py first."
        )

    interpreter = Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()

    input_idx  = interpreter.get_input_details()[0]["index"]
    output_idx = interpreter.get_output_details()[0]["index"]

    logger.info("TFLite model loaded: %s", MODEL_TFLITE_PATH)
    return interpreter, input_idx, output_idx


def load_label_map() -> dict[int, str]:
    """
    Load the integer → gesture-name mapping from label_map.json.

    label_map.json stores {"Hello": 0, "Yes": 1, …} (str → int).
    This function inverts it to {0: "Hello", 1: "Yes", …} (int → str)
    for fast lookup during inference.
    """
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Label map not found: {LABEL_MAP_PATH}\n"
            "Run preprocess.py and train_model.py first."
        )
    with open(LABEL_MAP_PATH) as f:
        raw = json.load(f)
    inv = {int(v): k for k, v in raw.items()}
    logger.info("Label map loaded (%d classes).", len(inv))
    return inv


def load_scaler():
    """
    Load the StandardScaler fitted during training.

    Returns None if the scaler file does not exist.
    """
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded: %s", SCALER_PATH)
        return scaler
    logger.warning("Scaler not found at %s — running without feature scaling.", SCALER_PATH)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Character model loading  (Model 2 — completely independent)
# ─────────────────────────────────────────────────────────────────────────────

def load_character_tflite_model() -> tuple:
    """
    Load the character TFLite interpreter.

    Returns
    -------
    (interpreter, input_idx, output_idx)

    Raises
    ------
    FileNotFoundError : character_model.tflite is missing
    ValueError        : model input shape is not (1, 42)
    """
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    if not os.path.exists(CHAR_MODEL_TFLITE_PATH):
        raise FileNotFoundError(
            f"Character TFLite model not found: {CHAR_MODEL_TFLITE_PATH}\n"
            "Run train_character_model.py then convert_character_tflite.py first."
        )

    interpreter = Interpreter(model_path=CHAR_MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()

    input_idx  = interpreter.get_input_details()[0]["index"]
    output_idx = interpreter.get_output_details()[0]["index"]

    # Validate input shape — must be (1, 42)
    input_shape = tuple(interpreter.get_input_details()[0]["shape"])
    expected    = (1, LANDMARK_FEATURES)   # (1, 42)
    if input_shape != expected:
        raise ValueError(
            f"Character model input shape mismatch!\n"
            f"  Expected : {expected}\n"
            f"  Got      : {input_shape}\n"
            "Retrain character model with 42 features."
        )

    logger.info(
        "Character TFLite model loaded: %s  (input shape: %s)",
        CHAR_MODEL_TFLITE_PATH, input_shape,
    )
    return interpreter, input_idx, output_idx


def load_character_label_map() -> dict[int, str]:
    """
    Load the integer → character-name mapping from character_label_map.json.

    Returns
    -------
    dict {int: str}  e.g. {0: "A", 1: "B", …, 25: "Z", 26: "0", …}
    """
    if not os.path.exists(CHAR_LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Character label map not found: {CHAR_LABEL_MAP_PATH}\n"
            "Run preprocess_characters.py and train_character_model.py first."
        )
    with open(CHAR_LABEL_MAP_PATH) as f:
        raw = json.load(f)
    inv = {int(v): k for k, v in raw.items()}
    logger.info("Character label map loaded (%d classes).", len(inv))
    return inv


def load_character_scaler():
    """
    Load the character-model StandardScaler.

    Validates the companion .meta.json to confirm the scaler was fitted
    with the correct feature count (42).

    Returns
    -------
    StandardScaler or None if missing (with a warning)
    """
    if not os.path.exists(CHAR_SCALER_PATH):
        logger.warning(
            "Character scaler not found at %s — running without scaling.",
            CHAR_SCALER_PATH,
        )
        return None

    # Validate companion metadata
    meta_path = CHAR_SCALER_PATH + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        fc = meta.get("feature_count")
        if fc is not None and fc != LANDMARK_FEATURES:
            raise ValueError(
                f"Character scaler feature count mismatch!\n"
                f"  Scaler was fitted with : {fc} features\n"
                f"  Inference expects      : {LANDMARK_FEATURES} features\n"
                "Re-run train_character_model.py."
            )

    with open(CHAR_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Character scaler loaded: %s", CHAR_SCALER_PATH)
    return scaler


def predict_character(
    features: list[float],
    interpreter,
    input_idx: int,
    output_idx: int,
    label_map: dict[int, str],
    scaler,
) -> tuple[str, float]:
    """
    Run one forward pass through the character TFLite model.

    Parameters
    ----------
    features    : 42-element hand-landmark vector
    interpreter : character TFLite interpreter
    input_idx   : index of the input tensor
    output_idx  : index of the output tensor
    label_map   : {class_int: char_name}
    scaler      : fitted character StandardScaler or None

    Returns
    -------
    (char_name, confidence)

    Raises
    ------
    ValueError : if len(features) != 42
    """
    if len(features) != LANDMARK_FEATURES:
        raise ValueError(
            f"Character predict received {len(features)} features — expected {LANDMARK_FEATURES}."
        )

    x = np.array(features, dtype=np.float32).reshape(1, -1)

    if scaler is not None:
        x = scaler.transform(x).astype(np.float32)

    interpreter.set_tensor(input_idx, x)
    interpreter.invoke()

    probabilities = interpreter.get_tensor(output_idx)[0]
    class_idx     = int(np.argmax(probabilities))
    confidence    = float(probabilities[class_idx])
    char_name     = label_map.get(class_idx, f"Class_{class_idx}")

    return char_name, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_hybrid_features(
    frame: np.ndarray,
    hand_detector: HandDetector,
    pose_detector: MoveNetDetector,
) -> list[float] | None:
    """
    Extract the 76-feature hybrid vector from a single live frame.

    Feature layout
    --------------
    [0  … 41]  42 normalised hand-landmark features  (MediaPipe Hands)
    [42 … 75]  34 normalised body-pose features       (MoveNet Lightning)

    Parameters
    ----------
    frame         : BGR frame from the webcam (already flipped)
    hand_detector : shared HandDetector instance
    pose_detector : shared MoveNetDetector instance

    Returns
    -------
    list[float] of length 76, or None if no hand is detected.
    The frame is annotated in-place by both detectors when a signal is found.
    """
    # ── MediaPipe: hand landmarks ─────────────────────────────────────────────
    raw_lm, _ = hand_detector.find_landmarks(frame, draw=True)

    # If no hand is visible, skip this frame entirely
    if raw_lm is None:
        return None

    hand_features = normalise_landmarks(raw_lm)   # 42 floats

    # ── MoveNet: body pose ────────────────────────────────────────────────────
    pose_raw, _ = pose_detector.detect(frame, draw=True)

    if pose_raw is not None:
        pose_features = MoveNetDetector.normalise_pose(pose_raw)  # 34 floats
    else:
        # Person not in frame — zero-fill pose block, keep hand features
        pose_features = [0.0] * POSE_FEATURES

    return hand_features + pose_features   # 76 floats


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    features: list[float],
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
    features    : 76-element hybrid feature vector
    interpreter : TFLite interpreter
    input_idx   : index of the input tensor
    output_idx  : index of the output tensor
    label_map   : {class_int: gesture_name}
    scaler      : fitted StandardScaler or None

    Returns
    -------
    (gesture_name, confidence)
    """
    x = np.array(features, dtype=np.float32).reshape(1, -1)

    if scaler is not None:
        x = scaler.transform(x).astype(np.float32)

    interpreter.set_tensor(input_idx, x)
    interpreter.invoke()

    probabilities = interpreter.get_tensor(output_idx)[0]
    class_idx     = int(np.argmax(probabilities))
    confidence    = float(probabilities[class_idx])
    gesture_name  = label_map.get(class_idx, f"Class_{class_idx}")

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
        """Add a new prediction and return the smoothed result."""
        if confidence < CONFIDENCE_THRESHOLD:
            self._buffer.append(None)
            return None, confidence

        self._buffer.append(gesture)
        counts = collections.Counter(g for g in self._buffer if g is not None)
        if not counts:
            return None, confidence

        best, votes = counts.most_common(1)[0]
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
    pose_active: bool,
) -> np.ndarray:
    """
    Render the heads-up display over the camera frame.

    Parameters
    ----------
    frame         : BGR frame
    gesture       : current smoothed prediction (None = no detection)
    confidence    : prediction confidence 0-1
    sentence      : list of confirmed words
    fps           : current frames per second
    stable_frames : consecutive frames the gesture has been stable
    pose_active   : whether MoveNet detected a person this frame
    """
    h, w = frame.shape[:2]

    # ── Top panel ─────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if gesture:
        color      = GREEN
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

    # Pose indicator dot
    pose_dot_color = GREEN if pose_active else RED
    cv2.circle(frame, (w - 20, 15), 7, pose_dot_color, -1)
    cv2.putText(frame, "Pose", (w - 65, 20), FONT, 0.4,
                GREEN if pose_active else RED, 1)

    # Stability progress bar
    bar_w = w - 20
    cv2.rectangle(frame, (10, 72), (10 + bar_w, 86), (60, 60, 60), -1)
    fill  = int(bar_w * stable_pct)
    bar_c = (0, int(200 * stable_pct), int(200 * (1 - stable_pct)))
    if fill > 0:
        cv2.rectangle(frame, (10, 72), (10 + fill, 86), bar_c, -1)

    # ── FPS counter (top-right) ───────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 50), FONT, 0.6, CYAN, 1)

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
    Open the webcam and run the full real-time hybrid inference loop.

    Flow per frame
    --------------
    1. Capture frame → flip (mirror)
    2. Extract hybrid features: MediaPipe hand (42) + MoveNet pose (34) = 76
    3. TFLite forward pass on 76-feature vector
    4. Smooth prediction via rolling majority-vote buffer
    5. Track stable frames → add word to sentence when stable
    6. Draw HUD with gesture, confidence, pose indicator, FPS, sentence
    7. Handle key presses (Q / C / S)
    """
    logger.info("Initialising hybrid inference pipeline…")
    print("\n🤙  Sign Language Translator – Real-Time Hybrid Inference")
    print(f"   Features: {LANDMARK_FEATURES} hand + {POSE_FEATURES} pose = {HYBRID_FEATURES} total\n")

    # ── Load resources ────────────────────────────────────────────────────────
    interpreter, input_idx, output_idx = load_tflite_model()
    label_map     = load_label_map()
    scaler        = load_scaler()
    hand_detector = HandDetector()
    pose_detector = MoveNetDetector()   # downloads MoveNet on first run
    smoother      = PredictionSmoother()
    tts           = TTSEngine()

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    sentence      = []
    stable_frames = 0
    last_stable   = None

    fps    = 0.0
    t_prev = time.perf_counter()

    logger.info("Inference loop started.")
    print("  Press Q to quit | C to clear sentence | S to speak\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Webcam read failed.")
                continue

            frame = cv2.flip(frame, 1)

            # ── Hybrid feature extraction ─────────────────────────────────────
            features = extract_hybrid_features(frame, hand_detector, pose_detector)

            gesture     = None
            confidence  = 0.0
            pose_active = False

            if features is not None:
                # Detect whether pose was active (any non-zero in pose block)
                pose_block  = features[LANDMARK_FEATURES:]
                pose_active = any(v != 0.0 for v in pose_block)

                raw_pred, conf      = predict(features, interpreter, input_idx,
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
            frame = draw_hud(frame, gesture, confidence,
                             sentence, fps, stable_frames, pose_active)
            cv2.imshow("Sign Language Translator – Press Q to quit", frame)

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
        hand_detector.close()
        logger.info("Hybrid inference pipeline stopped.")
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
