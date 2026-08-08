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
# Import claymorphism design system
# ─────────────────────────────────────────────────────────────────────────────
from utils.ui_components import (
    inject_css, svg_hand_3d, svg_ai_robot, svg_speaker, svg_microphone,
)

inject_css(theme="dark")


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
    # Logo / title block
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 18px;">
      {svg_hand_3d()}
      <div style="font-size:1.1rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#60a5fa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:8px;">
        ISL Translator
      </div>
      <div style="font-size:0.72rem;color:#475569;font-weight:600;margin-top:2px;">
        Claymorphism Edition
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    # Model status badge
    if st.session_state["model_loaded"]:
        st.markdown('<div style="margin-bottom:6px;"><span class="badge badge-green">✅ Model Ready</span></div>', unsafe_allow_html=True)
    elif st.session_state["error_msg"]:
        st.markdown('<div style="margin-bottom:6px;"><span class="badge badge-red">❌ Model Missing</span></div>', unsafe_allow_html=True)
        st.error(st.session_state["error_msg"])
    else:
        st.markdown('<div style="margin-bottom:6px;"><span class="badge badge-yellow">⏳ Loading…</span></div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    # Live stats
    st.markdown('<div class="clay-section-title">📊 Live Stats</div>', unsafe_allow_html=True)
    with _lock:
        _g   = _state["gesture"]
        _c   = _state["confidence"]
        _sf  = _state["stable_count"]
        _fc  = _state["frame_count"]

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
      <div class="stat-card"><div class="stat-label">Gesture</div>
        <div class="stat-value" style="font-size:0.95rem;">{_g}</div></div>
      <div class="stat-card"><div class="stat-label">Confidence</div>
        <div class="stat-value">{_c:.0%}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
      <div class="stat-card"><div class="stat-label">Stability</div>
        <div class="stat-value">{_sf}/{STABLE_FRAME_COUNT}</div></div>
      <div class="stat-card"><div class="stat-label">Frames</div>
        <div class="stat-value">{_fc}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    # Recent gestures
    st.markdown('<div class="clay-section-title">📝 Recent Gestures</div>', unsafe_allow_html=True)
    with _lock:
        _hist = list(_state["history"][-10:])
    if _hist:
        chips = "".join(f'<span class="gesture-chip">{w}</span>' for w in reversed(_hist))
        st.markdown(f'<div style="line-height:2;">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;font-size:0.85rem;font-style:italic;">No gestures yet.</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    # Supported gestures
    st.markdown('<div class="clay-section-title">🤙 Supported Gestures</div>', unsafe_allow_html=True)
    badges = "".join(f'<span class="gesture-chip">{g}</span>' for g in GESTURE_LABELS)
    st.markdown(f'<div style="line-height:2.2;">{badges}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    # How to use
    st.markdown('<div class="clay-section-title">📖 How to Use</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Click <b>START</b> and allow camera access"),
        ("2", "Hold a gesture steady in front of the camera"),
        ("3", "Wait for the stability bar to fill — word is added"),
        ("4", "Press <b>🔊 Speak</b> to hear the sentence"),
        ("5", "Press <b>⬅️ Undo</b> to remove the last word"),
        ("6", "Press <b>🗑️ Clear</b> to start a new sentence"),
    ]
    for num, text in steps:
        st.markdown(
            f'<div class="step-item"><span class="step-num">{num}</span>{text}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:28px 0 8px;">
  <div style="display:flex;align-items:center;justify-content:center;gap:18px;flex-wrap:wrap;">
    <div style="flex-shrink:0;">{svg_hand_3d()}</div>
    <div>
      <div class="clay-title">Real-Time Sign Language Translator</div>
      <div class="clay-subtitle">Powered by MediaPipe · TensorFlow Lite · Web Speech API</div>
    </div>
    <div style="flex-shrink:0;">{svg_ai_robot()}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

col_video, col_info = st.columns([3, 2], gap="large")

# ── Left column: webcam ───────────────────────────────────────────────────────
with col_video:
    st.markdown('<div class="clay-section-title">📷 Live Camera Feed</div>', unsafe_allow_html=True)

    if interp is not None:
        st.markdown('<div class="camera-wrapper">', unsafe_allow_html=True)
        webrtc_streamer(
            key="sign-lang",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:linear-gradient(145deg,#1a2240,#141a34);"
            "border:2px dashed rgba(99,102,241,0.3);border-radius:24px;"
            "height:360px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:12px;"
            "box-shadow:8px 8px 20px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.05);'>"
            "<span style='font-size:2.5rem;'>📷</span>"
            "<span style='color:#475569;font-weight:600;'>Model not loaded — run training pipeline first</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='color:#334155;font-size:0.8rem;margin-top:10px;text-align:center;"
        "font-weight:500;'>💡 Good lighting and a plain background improve accuracy</div>",
        unsafe_allow_html=True,
    )

# ── Right column: prediction + sentence ──────────────────────────────────────
with col_info:

    # ── Prediction card ───────────────────────────────────────────────────────
    with _lock:
        _g  = _state["gesture"]
        _c  = _state["confidence"]
        _sf = _state["stable_count"]

    conf_pct = int(_c * 100)
    if _c >= 0.75:
        gesture_color = "linear-gradient(135deg,#34d399,#10b981)"
        fill_cls = "conf-fill-high"
    elif _c >= 0.5:
        gesture_color = "linear-gradient(135deg,#fbbf24,#f59e0b)"
        fill_cls = "conf-fill-medium"
    else:
        gesture_color = "linear-gradient(135deg,#818cf8,#6366f1)"
        fill_cls = "conf-fill-low"

    stable_pct = int(min(_sf / max(STABLE_FRAME_COUNT, 1), 1.0) * 100)

    st.markdown(f"""
    <div class="clay-card clay-card-violet" style="text-align:center;margin-bottom:14px;">
      <div style="font-size:0.7rem;color:#64748b;font-weight:700;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:10px;">🎯 Current Prediction</div>
      <div style="font-size:3rem;font-weight:800;
                  background:{gesture_color};
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  filter:drop-shadow(0 0 14px rgba(139,92,246,0.4));
                  line-height:1.1;margin-bottom:6px;">{_g}</div>
      <div style="color:#64748b;font-size:0.85rem;font-weight:600;margin-bottom:14px;">
        Confidence: <span style="color:#c7d2fe;font-weight:800;">{conf_pct}%</span>
      </div>
      <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:0.75rem;color:#64748b;font-weight:600;">CONFIDENCE</span>
          <span style="font-size:0.75rem;color:#94a3b8;">{conf_pct}%</span>
        </div>
        <div class="conf-track">
          <div class="{fill_cls}" style="width:{conf_pct}%;transition:width 0.4s;"></div>
        </div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:0.75rem;color:#64748b;font-weight:600;">STABILITY</span>
          <span style="font-size:0.75rem;color:#94a3b8;">{_sf}/{STABLE_FRAME_COUNT}</span>
        </div>
        <div class="conf-track">
          <div style="width:{stable_pct}%;height:100%;background:linear-gradient(90deg,#818cf8,#a78bfa);
                      border-radius:8px;transition:width 0.3s;"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.12);">', unsafe_allow_html=True)

    # ── Sentence builder ──────────────────────────────────────────────────────
    with _lock:
        _words = list(_state["sentence"])

    sentence_str = " ".join(_words) if _words else "<span style='color:#334155;font-style:italic;'>Waiting for gestures…</span>"

    st.markdown(f"""
    <div style="margin-bottom:6px;">
      <div class="clay-section-title">📝 Sentence Builder</div>
    </div>
    <div class="sentence-box">{sentence_str}</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        speak_text = " ".join(_words).replace("'", "\\'") if _words else ""
        disabled_style = "opacity:0.4;cursor:not-allowed;" if not _words else "cursor:pointer;"
        st.markdown(f"""
        <button onclick="(function(){{if(!'{speak_text}')return;
            var m=new SpeechSynthesisUtterance('{speak_text}');
            m.rate=0.95;m.pitch=1.0;window.speechSynthesis.cancel();
            window.speechSynthesis.speak(m);}})();"
          style="width:100%;background:linear-gradient(135deg,#4f46e5,#7c3aed);
            color:#fff;border:none;border-radius:16px;padding:11px 0;
            font-size:0.9rem;font-weight:700;{disabled_style}
            box-shadow:0 5px 14px rgba(99,102,241,0.4),inset 0 1px 0 rgba(255,255,255,0.12);
            transition:all 0.2s;font-family:'Plus Jakarta Sans',sans-serif;">
          {svg_speaker()[:30].replace('<svg','<svg style="width:16px;height:16px;display:inline;"') if False else '🔊'} Speak
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

    st.markdown('<hr style="border-color:rgba(99,102,241,0.12);">', unsafe_allow_html=True)

    # ── Word history chips ────────────────────────────────────────────────────
    st.markdown('<div class="clay-section-title">🕐 Word History</div>', unsafe_allow_html=True)
    with _lock:
        _hist = list(_state["history"][-20:])

    if _hist:
        chips = "".join(
            f'<span class="gesture-chip">{w}</span>'
            for w in reversed(_hist)
        )
        st.markdown(f'<div style="line-height:2.2;">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#334155;font-size:0.88rem;font-style:italic;font-weight:500;">'
            'No words recognised yet — start signing!</div>',
            unsafe_allow_html=True,
        )
