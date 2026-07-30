# =============================================================================
# app_cloud.py  –  Cloud-compatible Streamlit UI
# =============================================================================
# Works on: Streamlit Cloud, Hugging Face Spaces, Render, Railway, Docker
#
# Key differences vs app.py:
#   • streamlit-webrtc  → browser webcam  (no cv2.VideoCapture)
#   • Web Speech API JS → browser TTS     (no pyttsx3)
#   • tflite-runtime    → lighter install (no full TensorFlow)
#
# Install extras:  pip install streamlit-webrtc aiortc
# Run:             streamlit run app_cloud.py
# =============================================================================

import os
import sys
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CONFIDENCE_THRESHOLD,
    STABLE_FRAME_COUNT,
    GESTURE_LABELS,
)
from utils.mediapipe_helper import HandDetector
from preprocess import normalise_landmarks
from inference import (
    load_tflite_model, load_label_map, load_scaler,
    predict, PredictionSmoother,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# WebRTC ICE config  (STUN server lets the browser punch through NAT)
# ─────────────────────────────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  –  full dark theme matching app.py
# ─────────────────────────────────────────────────────────────────────────────
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
    font-size: 2.6rem;
    font-weight: 700;
    color: #00ff88;
    letter-spacing: 2px;
}
.pred-conf { font-size: 1rem; color: #aaa; margin-top: 6px; }

/* ── Sentence box ── */
.sentence-box {
    background: #1a1f2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 1.25rem;
    color: #e0e0e0;
    min-height: 64px;
    word-break: break-word;
    line-height: 1.6;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
}
.stat-card {
    flex: 1;
    background: #1a1f2e;
    border: 1px solid #2a2f3f;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
}
.stat-label { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.stat-value { font-size: 1.3rem; font-weight: 700; color: #00d4ff; }

/* ── Gesture badge ── */
.gesture-badge {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #2a2f3f;
    border-radius: 8px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 0.85rem;
    color: #ccc;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #141720;
    border-right: 1px solid #2a2f3f;
}

/* ── Status badges ── */
.badge-green  { color: #00ff88; font-weight: 700; }
.badge-red    { color: #ff4444; font-weight: 700; }
.badge-yellow { color: #ffaa00; font-weight: 700; }

/* ── How-to steps ── */
.step {
    background: #1a1f2e;
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    color: #ccc;
}
.step-num { color: #00d4ff; font-weight: 700; margin-right: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "sentence":        [],
    "current_gesture": "—",
    "confidence":      0.0,
    "stable_frames":   0,
    "history":         [],
    "fps":             0.0,
    "model_loaded":    False,
    "error_msg":       "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached – loaded once per server session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading model…")
def get_resources():
    try:
        interp, in_idx, out_idx = load_tflite_model()
        label_map = load_label_map()
        scaler    = load_scaler()
        detector  = HandDetector()
        return interp, in_idx, out_idx, label_map, scaler, detector, None
    except FileNotFoundError as exc:
        return None, None, None, None, None, None, str(exc)


interp, in_idx, out_idx, label_map, scaler, detector, _load_err = get_resources()
if interp is not None:
    st.session_state["model_loaded"] = True
if _load_err:
    st.session_state["error_msg"] = _load_err

smoother = PredictionSmoother()

# ─────────────────────────────────────────────────────────────────────────────
# Shared mutable state for the WebRTC background thread
# (Streamlit session_state is NOT safe to write from background threads)
# ─────────────────────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "gesture":      "—",
    "confidence":   0.0,
    "stable_count": 0,
    "last_stable":  None,
    "sentence":     [],
    "history":      [],
    "fps":          0.0,
    "frame_count":  0,
}


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC frame callback  (runs in a background thread, ~30×/sec)
# ─────────────────────────────────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    if interp is None:
        return frame

    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    raw_lm, annotated = detector.find_landmarks(img, draw=True)

    gesture    = None
    confidence = 0.0

    if raw_lm is not None:
        norm_lm        = normalise_landmarks(raw_lm)
        raw_pred, conf = predict(norm_lm, interp, in_idx, out_idx, label_map, scaler)
        gesture, confidence = smoother.update(raw_pred, conf)
    else:
        smoother.reset()

    with _lock:
        # Stability tracking → sentence building
        if gesture:
            if gesture == _state["last_stable"]:
                _state["stable_count"] += 1
            else:
                _state["stable_count"] = 1
                _state["last_stable"]  = gesture

            if _state["stable_count"] == STABLE_FRAME_COUNT:
                words = _state["sentence"]
                if not words or words[-1] != gesture:
                    _state["sentence"].append(gesture)
                    _state["history"].append(gesture)
        else:
            _state["stable_count"] = 0
            _state["last_stable"]  = None

        _state["gesture"]    = gesture or "—"
        _state["confidence"] = confidence
        _state["frame_count"] += 1

    # ── Overlay on the video frame ────────────────────────────────────────────
    h, w = annotated.shape[:2]

    # Top banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    color = (0, 220, 0) if gesture else (0, 0, 220)
    label = gesture if gesture else "No gesture"
    cv2.putText(annotated, label, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    if gesture:
        cv2.putText(annotated, f"{confidence:.0%} confidence", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Stability bar
    stable_pct = min(_state["stable_count"] / max(STABLE_FRAME_COUNT, 1), 1.0)
    bar_w = w - 24
    cv2.rectangle(annotated, (12, 66), (12 + bar_w, 76), (50, 50, 50), -1)
    fill_w = int(bar_w * stable_pct)
    if fill_w > 0:
        bar_color = (0, int(200 * stable_pct), int(200 * (1 - stable_pct)))
        cv2.rectangle(annotated, (12, 66), (12 + fill_w, 76), bar_color, -1)

    # Bottom sentence strip
    overlay2 = annotated.copy()
    cv2.rectangle(overlay2, (0, h - 44), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, annotated, 0.45, 0, annotated)
    sentence_text = " ".join(_state["sentence"]) if _state["sentence"] else "—"
    cv2.putText(annotated, f"Sentence: {sentence_text}", (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings & Info")
    st.markdown("---")

    # Model status
    if st.session_state["model_loaded"]:
        st.markdown('<p class="badge-green">✅ Model loaded</p>', unsafe_allow_html=True)
    elif st.session_state["error_msg"]:
        st.markdown('<p class="badge-red">❌ Model not found</p>', unsafe_allow_html=True)
        st.error(st.session_state["error_msg"])
    else:
        st.markdown('<p class="badge-yellow">⏳ Loading…</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Live stats
    st.markdown("### 📊 Live Stats")
    with _lock:
        _g   = _state["gesture"]
        _c   = _state["confidence"]
        _sf  = _state["stable_count"]
        _fc  = _state["frame_count"]

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">Gesture</div>
            <div class="stat-value" style="font-size:1rem;">{_g}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Confidence</div>
            <div class="stat-value">{_c:.0%}</div>
        </div>
    </div>
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">Stability</div>
            <div class="stat-value">{_sf}/{STABLE_FRAME_COUNT}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Frames</div>
            <div class="stat-value">{_fc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # History
    st.markdown("### 📝 Recent Gestures")
    with _lock:
        _hist = list(_state["history"][-10:])
    if _hist:
        for item in reversed(_hist):
            st.write(f"• {item}")
    else:
        st.caption("No gestures recognised yet.")

    st.markdown("---")

    # Supported gestures reference
    st.markdown("### 🤙 Supported Gestures")
    badges = "".join(f'<span class="gesture-badge">{g}</span>' for g in GESTURE_LABELS)
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("---")

    # How to use
    st.markdown("### 📖 How to Use")
    st.markdown("""
    <div class="step"><span class="step-num">1</span>Click <b>START</b> and allow camera access</div>
    <div class="step"><span class="step-num">2</span>Hold a gesture steady in front of the camera</div>
    <div class="step"><span class="step-num">3</span>Wait for the stability bar to fill — word is added</div>
    <div class="step"><span class="step-num">4</span>Press <b>🔊 Speak</b> to hear the sentence</div>
    <div class="step"><span class="step-num">5</span>Press <b>⬅️ Undo</b> to remove the last word</div>
    <div class="step"><span class="step-num">6</span>Press <b>🗑️ Clear</b> to start a new sentence</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; margin-bottom:4px;'>🤙 Real-Time Sign Language Translator</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#888; margin-bottom:20px;'>"
    "Powered by MediaPipe · TensorFlow Lite · Web Speech API</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

col_video, col_info = st.columns([3, 2], gap="large")

# ── Left column: webcam ───────────────────────────────────────────────────────
with col_video:
    st.markdown("### 📷 Live Camera Feed")

    if interp is not None:
        webrtc_streamer(
            key="sign-lang",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
    else:
        st.markdown(
            "<div style='background:#1a1f2e; border:2px dashed #333; border-radius:12px;"
            "height:360px; display:flex; align-items:center; justify-content:center;"
            "color:#555; font-size:1.1rem;'>📷 Model not loaded — run the training pipeline first</div>",
            unsafe_allow_html=True,
        )

    # Tip below camera
    st.markdown(
        "<p style='color:#555; font-size:0.82rem; margin-top:8px; text-align:center;'>"
        "💡 Tip: Good lighting and a plain background improve accuracy</p>",
        unsafe_allow_html=True,
    )

# ── Right column: prediction + sentence ──────────────────────────────────────
with col_info:

    # ── Prediction card ───────────────────────────────────────────────────────
    st.markdown("### 🎯 Current Prediction")

    with _lock:
        _g  = _state["gesture"]
        _c  = _state["confidence"]
        _sf = _state["stable_count"]

    st.markdown(f"""
    <div class="pred-card">
        <div class="pred-gesture">{_g}</div>
        <div class="pred-conf">Confidence: {_c:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(_c), text="Confidence")
    stable_pct = min(_sf / max(STABLE_FRAME_COUNT, 1), 1.0)
    st.progress(stable_pct, text=f"Stability  {_sf} / {STABLE_FRAME_COUNT} frames")

    st.markdown("---")

    # ── Sentence builder ──────────────────────────────────────────────────────
    st.markdown("### 📝 Sentence Builder")

    with _lock:
        _words = list(_state["sentence"])

    sentence_str = " ".join(_words) if _words else "*(waiting for gestures…)*"
    st.markdown(f'<div class="sentence-box">{sentence_str}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        # Web Speech API speak button — works in any browser, no pyttsx3 needed
        speak_text = " ".join(_words).replace("'", "\\'") if _words else ""
        st.markdown(f"""
        <script>
        function speakSentence() {{
            if (!speak_text) return;
            var msg = new SpeechSynthesisUtterance('{speak_text}');
            msg.rate = 0.95;
            msg.pitch = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        }}
        </script>
        <button onclick="speakSentence()" {'disabled' if not _words else ''} style="
            width:100%; background:{'#00d4ff' if _words else '#333'};
            color:{'#000' if _words else '#666'};
            border:none; border-radius:10px; padding:10px 0;
            font-size:0.95rem; font-weight:700; cursor:{'pointer' if _words else 'default'};
            transition: all 0.2s;">
            🔊 Speak
        </button>
        """, unsafe_allow_html=True)

    with btn2:
        if st.button("⬅️ Undo", use_container_width=True, disabled=not _words):
            with _lock:
                if _state["sentence"]:
                    _state["sentence"].pop()
            st.rerun()

    with btn3:
        if st.button("🗑️ Clear", use_container_width=True, disabled=not _words):
            with _lock:
                _state["sentence"]     = []
                _state["last_stable"]  = None
                _state["stable_count"] = 0
            smoother.reset()
            st.rerun()

    st.markdown("---")

    # ── Word history chips ────────────────────────────────────────────────────
    st.markdown("### 🕐 Word History")
    with _lock:
        _hist = list(_state["history"][-20:])

    if _hist:
        chips = "".join(
            f'<span style="background:#1a1f2e; border:1px solid #2a2f3f; '
            f'border-radius:20px; padding:3px 12px; margin:3px; '
            f'font-size:0.85rem; color:#ccc; display:inline-block;">{w}</span>'
            for w in reversed(_hist)
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.caption("No words recognised yet — start signing!")
