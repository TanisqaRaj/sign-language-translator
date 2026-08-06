# =============================================================================
# app.py
# Phase 7 – Streamlit Web Application
# =============================================================================
# Usage:
#   streamlit run app.py
#
# Features:
#   • Live webcam feed with hand skeleton overlay
#   • Real-time gesture prediction + confidence bar
#   • Sentence builder – words are added when a gesture is held steady
#   • Speak button – reads the sentence aloud via pyttsx3
#   • Clear button – resets the sentence
#   • Modern dark-themed UI with custom CSS
# =============================================================================

import os
import sys
import json
import pickle
import time

import cv2
import numpy as np
import streamlit as st

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
)
from utils.logger           import get_logger
from utils.mediapipe_helper import HandDetector
from utils.movenet_helper   import MoveNetDetector
from utils.tts_engine       import TTSEngine
from preprocess             import normalise_landmarks
from inference              import load_tflite_model, load_label_map, load_scaler, predict, PredictionSmoother
from config                 import POSE_FEATURES

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – dark modern theme
# ─────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject custom CSS for a polished dark UI."""
    st.markdown("""
    <style>
    /* ── Global ── */
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #00d4ff; }

    /* ── Prediction card ── */
    .pred-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 16px;
        text-align: center;
    }
    .pred-gesture {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00ff88;
        letter-spacing: 2px;
    }
    .pred-conf {
        font-size: 1rem;
        color: #aaa;
        margin-top: 4px;
    }

    /* ── Sentence box ── */
    .sentence-box {
        background: #1a1f2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 1.25rem;
        color: #e0e0e0;
        min-height: 60px;
        word-break: break-word;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #141720;
        border-right: 1px solid #2a2f3f;
    }

    /* ── Status badges ── */
    .badge-green { color: #00ff88; font-weight: 700; }
    .badge-red   { color: #ff4444; font-weight: 700; }
    .badge-yellow{ color: #ffaa00; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialise all Streamlit session state variables on first load."""
    defaults = {
        "sentence":       [],          # List of words in the sentence
        "running":        False,       # Is the webcam loop active?
        "current_gesture": "—",       # Latest smoothed prediction
        "confidence":     0.0,        # Latest confidence score
        "stable_frames":  0,           # Frames gesture has been held
        "history":        [],          # Log of recent predictions
        "fps":            0.0,
        "model_loaded":   False,
        "error_msg":      "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Resource loading (cached so they aren't reloaded on every rerun)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_model_resources():
    try:
        interpreter, input_idx, output_idx = load_tflite_model()
        label_map    = load_label_map()
        scaler       = load_scaler()
        detector     = HandDetector()
        pose_detector = MoveNetDetector()
        tts          = TTSEngine()
        return interpreter, input_idx, output_idx, label_map, scaler, detector, pose_detector, tts
    except FileNotFoundError as exc:
        st.session_state["error_msg"] = str(exc)
        return None, None, None, None, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Frame processing
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    frame: np.ndarray,
    interpreter,
    input_idx: int,
    output_idx: int,
    label_map: dict,
    scaler,
    detector: HandDetector,
    pose_detector: MoveNetDetector,
    smoother: PredictionSmoother,
) -> tuple[np.ndarray, str | None, float]:
    raw_lm, annotated = detector.find_landmarks(frame, draw=True)

    if raw_lm is None:
        smoother.reset()
        return annotated, None, 0.0

    hand_features = normalise_landmarks(raw_lm)

    pose_raw, _ = pose_detector.detect(frame, draw=False)
    pose_features = MoveNetDetector.normalise_pose(pose_raw) if pose_raw is not None else [0.0] * POSE_FEATURES

    features = hand_features + pose_features
    raw_pred, conf = predict(features, interpreter, input_idx, output_idx, label_map, scaler)
    gesture, confidence = smoother.update(raw_pred, conf)
    return annotated, gesture, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the settings and info sidebar."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        # Model status
        if st.session_state.get("model_loaded"):
            st.markdown('<p class="badge-green">✅ Model loaded</p>', unsafe_allow_html=True)
        elif st.session_state.get("error_msg"):
            st.markdown('<p class="badge-red">❌ Model not found</p>', unsafe_allow_html=True)
            st.error(st.session_state["error_msg"])
        else:
            st.markdown('<p class="badge-yellow">⏳ Loading…</p>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Stats")
        st.metric("FPS",        f"{st.session_state['fps']:.1f}")
        st.metric("Confidence", f"{st.session_state['confidence']:.0%}")
        st.metric("Stable Frames",
                  f"{st.session_state['stable_frames']} / {STABLE_FRAME_COUNT}")

        st.markdown("---")
        st.markdown("### 📝 History")
        history = st.session_state["history"][-10:]  # Show last 10
        if history:
            for item in reversed(history):
                st.write(f"• {item}")
        else:
            st.caption("No gestures recognised yet.")

        st.markdown("---")
        st.markdown("### ℹ️ Controls")
        st.markdown("""
        - **Start / Stop** – toggle webcam
        - **Speak** – read the sentence aloud
        - **Clear** – reset the sentence
        - Hold a gesture steady for recognition
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

def render_main_ui(
    col_video,
    col_info,
    frame_placeholder,
    gesture_placeholder,
    sentence_placeholder,
) -> None:
    """Render prediction card and sentence builder in the info column."""

    gesture    = st.session_state["current_gesture"]
    confidence = st.session_state["confidence"]

    # ── Prediction card ────────────────────────────────────────────────────────
    with gesture_placeholder.container():
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-gesture">{gesture if gesture != "—" else "—"}</div>
            <div class="pred-conf">Confidence: {confidence:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence progress bar
        st.progress(confidence, text="Confidence")

        # Stability bar
        stable_pct = min(
            st.session_state["stable_frames"] / max(STABLE_FRAME_COUNT, 1), 1.0
        )
        st.progress(stable_pct, text=f"Stability ({st.session_state['stable_frames']}/{STABLE_FRAME_COUNT})")

    # ── Sentence builder ───────────────────────────────────────────────────────
    with sentence_placeholder.container():
        st.markdown("### 📝 Sentence Builder")
        words        = st.session_state["sentence"]
        sentence_str = " ".join(words) if words else "*(waiting for gestures…)*"
        st.markdown(f'<div class="sentence-box">{sentence_str}</div>',
                    unsafe_allow_html=True)

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🔊 Speak", key="btn_speak", use_container_width=True, type="primary"):
                if words:
                    resources = get_model_resources()
                    tts       = resources[7]
                    if tts:
                        tts.speak(" ".join(words), force=True)
        with btn_col2:
            if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
                st.session_state["sentence"]       = []
                st.session_state["current_gesture"] = "—"
                st.session_state["stable_frames"]  = 0
                resources = get_model_resources()
                if resources[7]:
                    resources[7].reset_last_spoken()
                st.rerun()
        with btn_col3:
            if st.button("⬅️ Undo", key="btn_undo", use_container_width=True):
                if st.session_state["sentence"]:
                    st.session_state["sentence"].pop()
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Webcam loop
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam_loop(
    frame_placeholder,
    interpreter,
    input_idx,
    output_idx,
    label_map,
    scaler,
    detector,
    pose_detector,
    tts,
    gesture_placeholder,
    sentence_placeholder,
    col_video,
    col_info,
) -> None:
    """
    Capture frames from the webcam and update the Streamlit UI in real-time.

    This runs inside Streamlit's execution model — each frame triggers a
    partial UI update using st.empty() placeholders.
    """
    cap     = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        st.error(f"Cannot open webcam at index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoother      = PredictionSmoother()
    last_stable   = None
    stable_frames = 0
    t_prev        = time.perf_counter()

    stop_placeholder = st.empty()
    stop_clicked     = stop_placeholder.button("⏹️ Stop Camera", key="stop_btn",
                                                type="secondary",
                                                use_container_width=True)

    try:
        while st.session_state["running"] and not stop_clicked:
            ret, frame = cap.read()
            if not ret:
                continue

            frame    = cv2.flip(frame, 1)
            annotated, gesture, confidence = process_frame(
                frame, interpreter, input_idx, output_idx,
                label_map, scaler, detector, pose_detector, smoother,
            )

            # ── FPS ───────────────────────────────────────────────────────────
            t_now  = time.perf_counter()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # ── Stability tracking ─────────────────────────────────────────────
            if gesture:
                if gesture == last_stable:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    last_stable   = gesture

                if stable_frames == STABLE_FRAME_COUNT:
                    words = st.session_state["sentence"]
                    if not words or words[-1] != gesture:
                        st.session_state["sentence"].append(gesture)
                        st.session_state["history"].append(gesture)
                        tts.speak(gesture)
            else:
                stable_frames = 0
                last_stable   = None

            # ── Update session state ──────────────────────────────────────────
            st.session_state["current_gesture"] = gesture or "—"
            st.session_state["confidence"]      = confidence
            st.session_state["stable_frames"]   = stable_frames
            st.session_state["fps"]             = fps

            # ── Display frame only (no buttons inside loop) ───────────────────
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

            # ── Update prediction card only (no buttons) ──────────────────────
            gesture_val = st.session_state["current_gesture"]
            conf_val    = st.session_state["confidence"]
            with gesture_placeholder.container():
                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-gesture">{gesture_val}</div>
                    <div class="pred-conf">Confidence: {conf_val:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf_val, text="Confidence")
                stable_pct = min(stable_frames / max(STABLE_FRAME_COUNT, 1), 1.0)
                st.progress(stable_pct, text=f"Stability ({stable_frames}/{STABLE_FRAME_COUNT})")

            # ── Update sentence display only (no buttons) ─────────────────────
            words        = st.session_state["sentence"]
            sentence_str = " ".join(words) if words else "*(waiting for gestures…)*"
            with sentence_placeholder.container():
                st.markdown("### 📝 Sentence Builder")
                st.markdown(f'<div class="sentence-box">{sentence_str}</div>',
                            unsafe_allow_html=True)

    finally:
        cap.release()
        st.session_state["running"] = False
        stop_placeholder.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Application entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Build and run the full Streamlit UI."""
    inject_css()
    init_session_state()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;'>🤙 Real-Time Sign Language Translator</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#888;'>Powered by MediaPipe · TensorFlow Lite · pyttsx3</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Load resources ──────────────────────────────────────────────────────────
    resources = get_model_resources()
    (interpreter, input_idx, output_idx,
     label_map, scaler, detector, pose_detector, tts) = resources

    if interpreter is not None:
        st.session_state["model_loaded"] = True

    render_sidebar()

    # ── Layout ─────────────────────────────────────────────────────────────────
    col_video, col_info = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown("### 📷 Live Camera Feed")
        frame_placeholder = st.empty()

        # Start / Stop button
        if not st.session_state["running"]:
            if st.button("▶️ Start Camera", key="btn_start", type="primary", use_container_width=True):
                if interpreter is None:
                    st.error("Model not loaded. Run train_model.py and convert_tflite.py first.")
                else:
                    st.session_state["running"] = True
                    st.rerun()
        # Show placeholder image when camera is off
        if not st.session_state["running"]:
            frame_placeholder.markdown(
                "<div style='background:#1a1f2e;border:2px dashed #333;"
                "border-radius:12px;height:360px;display:flex;"
                "align-items:center;justify-content:center;"
                "color:#555;font-size:1.1rem;'>"
                "📷 Camera feed will appear here</div>",
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown("### 🎯 Prediction")
        gesture_placeholder  = st.empty()
        sentence_placeholder = st.empty()

        # Initial render (camera off state)
        render_main_ui(col_video, col_info, frame_placeholder,
                       gesture_placeholder, sentence_placeholder)

    # ── Run loop when active ────────────────────────────────────────────────────
    if st.session_state["running"] and interpreter is not None:
        run_webcam_loop(
            frame_placeholder,
            interpreter, input_idx, output_idx,
            label_map, scaler, detector, pose_detector, tts,
            gesture_placeholder, sentence_placeholder,
            col_video, col_info,
        )


if __name__ == "__main__":
    main()
