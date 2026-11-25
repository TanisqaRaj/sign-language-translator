# =============================================================================
# app_cloud.py  –  Cloud-compatible Streamlit entry point
# Sign Language Translator – Professional Edition
# =============================================================================
# Works on: Streamlit Cloud, Hugging Face Spaces, Render, Railway, Docker
#
# Key differences vs app.py:
#   • streamlit-webrtc  → browser webcam  (no cv2.VideoCapture)
#   • Web Speech API JS → browser TTS     (no pyttsx3)
#   • tflite-runtime    → lighter install (no full TensorFlow)
#
# Run:  streamlit run app_cloud.py
# =============================================================================

from __future__ import annotations

import os
import sys
import time
import threading

# Suppress TensorFlow / MediaPipe / absl noise before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["GLOG_minloglevel"]        = "2"
os.environ["MEDIAPIPE_DISABLE_GPU"]   = "1"

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
from config import (
    CONFIDENCE_THRESHOLD,
    STABLE_FRAME_COUNT,
    GESTURE_LABELS,
    GESTURE_MEANINGS,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_THEME,
    DEFAULT_FONT_SIZE,
    DEFAULT_HIGH_CONTRAST,
    AUTO_SPEAK_DEFAULT,
    SUBTITLE_ENABLED_DEFAULT,
    LANDMARK_FEATURES,
    CHARACTER_CONFIDENCE_THRESHOLD,
    CHARACTER_STABLE_FRAME_COUNT,
    CHARACTER_DEBOUNCE_FRAMES,
    WORD_CONFIDENCE_THRESHOLD,
    CHARACTER_LABELS,
)

# ── Shared utilities ──────────────────────────────────────────────────────────
from utils.mediapipe_helper import HandDetector, MultiHandResult
from utils.analytics        import SessionAnalytics
from utils.gesture_history  import GestureHistory
from utils.translator       import translate, translate_sentence

# ── ML ────────────────────────────────────────────────────────────────────────
from preprocess import normalise_landmarks
from inference  import (
    load_tflite_model, load_label_map, load_scaler,
    predict, PredictionSmoother,
    load_character_tflite_model, load_character_label_map, load_character_scaler,
    predict_character,
)

# ── UI ────────────────────────────────────────────────────────────────────────
from utils.ui_components import (
    inject_css,
    svg_hand_3d,
    svg_ai_robot,
    svg_speaker,
    svg_analytics,
)

# ── Page modules (shared pages only) ─────────────────────────────────────────
from app_pages.home      import show_home
from app_pages.history   import show_history
from app_pages.analytics import show_analytics
from app_pages.settings  import show_settings_cloud
from app_pages.about     import show_about

# Navigation pages shown in the sidebar (matches app.py exactly)
_NAV = [
    "🏠 Home",
    "📷 Live Translator",
    "📜 History",
    "📊 Analytics",
    "⚙️ Settings",
    "ℹ️ About",
]

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
# WebRTC ICE / STUN config
# ─────────────────────────────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    # Navigation
    "page":              _NAV[0],
    # Translator state
    "sentence":          [],
    "current_gesture":   "—",
    "confidence":        0.0,
    "stable_frames":     0,
    # Model
    "model_loaded":      False,
    "char_model_loaded": False,
    "error_msg":         "",
    # Recognition mode: "Word" or "Character"
    "recognition_mode":  "Word",
    # Settings
    "theme":             DEFAULT_THEME,
    "font_size":         DEFAULT_FONT_SIZE,
    "high_contrast":     DEFAULT_HIGH_CONTRAST,
    "auto_speak":        AUTO_SPEAK_DEFAULT,
    "subtitles":         SUBTITLE_ENABLED_DEFAULT,
    "language":          DEFAULT_LANGUAGE,
    "conf_threshold":    CONFIDENCE_THRESHOLD,
    "stable_frames_cfg": STABLE_FRAME_COUNT,
    # Character mode settings
    "char_conf_threshold":    CHARACTER_CONFIDENCE_THRESHOLD,
    "char_stable_frames_cfg": CHARACTER_STABLE_FRAME_COUNT,
    # Auto-speak tracking
    "_last_auto_spoken": "",
    # Speech
    "speech_rate":       150,
    # Objects
    "analytics":         None,
    "history":           None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state["analytics"] is None:
    _a = SessionAnalytics()
    _a.start_session()
    st.session_state["analytics"] = _a

if st.session_state["history"] is None:
    st.session_state["history"] = GestureHistory()

# ── Key aliases so shared pages work identically to app.py ───────────────────
# app.py uses: target_language, large_font, subtitles_on, history_obj, analytics_obj
# app_cloud.py uses: language, font_size, subtitles, history, analytics
# We keep both in sync so every shared page (Home, History, Analytics, Settings)
# reads the same values regardless of which app is running.
if "target_language" not in st.session_state:
    st.session_state["target_language"] = st.session_state["language"]
if "large_font" not in st.session_state:
    st.session_state["large_font"] = (st.session_state["font_size"] == "large")
if "subtitles_on" not in st.session_state:
    st.session_state["subtitles_on"] = st.session_state["subtitles"]
if "history_obj" not in st.session_state:
    st.session_state["history_obj"] = st.session_state["history"]
if "analytics_obj" not in st.session_state:
    st.session_state["analytics_obj"] = st.session_state["analytics"]
# Keep aliases in sync on every rerun
st.session_state["target_language"] = st.session_state["language"]
st.session_state["large_font"]       = (st.session_state["font_size"] == "large")
st.session_state["subtitles_on"]     = st.session_state["subtitles"]
st.session_state["history_obj"]      = st.session_state["history"]
st.session_state["analytics_obj"]    = st.session_state["analytics"]

inject_css(
    theme=st.session_state["theme"],
    large_font=(st.session_state["font_size"] == "large"),
    high_contrast=st.session_state["high_contrast"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached once per server process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading word model…")
def get_resources():
    try:
        interp, in_idx, out_idx = load_tflite_model()
        label_map = load_label_map()
        scaler    = load_scaler()
        input_shape = tuple(interp.get_input_details()[0]["shape"])
        if input_shape[1] != LANDMARK_FEATURES:
            raise ValueError(
                f"Word model input shape mismatch!\n"
                f"  Expected : (1, {LANDMARK_FEATURES})\n"
                f"  Got      : {input_shape}\n"
                "  Re-run train_model.py and convert_tflite.py to fix this."
            )
        return interp, in_idx, out_idx, label_map, scaler, None
    except FileNotFoundError as exc:
        return None, None, None, None, None, str(exc)
    except ValueError as exc:
        return None, None, None, None, None, str(exc)


@st.cache_resource(show_spinner="Initialising hand detector…")
def get_hand_detector():
    return HandDetector()


@st.cache_resource(show_spinner="Loading character model…")
def get_char_resources():
    try:
        interp, in_idx, out_idx = load_character_tflite_model()
        label_map = load_character_label_map()
        scaler    = load_character_scaler()
        return interp, in_idx, out_idx, label_map, scaler, None
    except FileNotFoundError as exc:
        return None, None, None, None, None, str(exc)
    except ValueError as exc:
        return None, None, None, None, None, str(exc)


# ── Load models ───────────────────────────────────────────────────────────────
interp, in_idx, out_idx, label_map, scaler, _load_err = get_resources()
if interp is not None:
    st.session_state["model_loaded"] = True
if _load_err and not st.session_state["error_msg"]:
    st.session_state["error_msg"] = _load_err

detector = get_hand_detector()

(
    char_interp, char_in_idx, char_out_idx,
    char_label_map, char_scaler, _char_load_err,
) = get_char_resources()
if char_interp is not None:
    st.session_state["char_model_loaded"] = True

smoother      = PredictionSmoother()
char_smoother = PredictionSmoother()

# ─────────────────────────────────────────────────────────────────────────────
# Shared mutable state for the WebRTC background thread
# ─────────────────────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state: dict = {
    "gesture":        "—",
    "confidence":     0.0,
    "stable_count":   0,
    "last_stable":    None,
    "fps":            0.0,
    "frame_count":    0,
    "t_last":         time.time(),
    "hands_count":    0,
    "left_detected":  False,
    "right_detected": False,
    "sentence":       [],
    "char_sentence":        [],
    "char_debounce_count":  0,
    "char_last_accepted":   None,
    "mode":                 "Word",
}


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_hand_state_locked(mhr: MultiHandResult) -> None:
    with _lock:
        _state["hands_count"]    = mhr.count
        _state["left_detected"]  = mhr.left_detected
        _state["right_detected"] = mhr.right_detected


def _update_fps() -> None:
    now = time.time()
    dt  = now - _state["t_last"]
    _state["t_last"] = now
    fps = 1.0 / dt if dt > 0 else 0.0
    _state["fps"] = round(fps * 0.1 + _state["fps"] * 0.9, 1)


def _draw_overlay(
    frame: np.ndarray,
    gesture: str | None,
    confidence: float,
    stable_count: int,
    stab_thr: int,
    sentence: list[str],
    mode: str,
    mhr: MultiHandResult | None = None,
) -> None:
    """Draw the HUD overlay onto the frame in-place."""
    h, w = frame.shape[:2]

    # Top banner
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 82), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    mode_color = (255, 140, 0) if mode == "Character" else (100, 220, 200)
    cv2.putText(frame, f"[{mode}]", (w - 120, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

    color = (0, 220, 160) if gesture else (100, 160, 200)
    label = gesture if gesture else "No gesture"
    cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    if gesture:
        cv2.putText(frame, f"{confidence:.0%} confidence", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 210), 1)

    # Stability bar
    stable_pct = min(stable_count / max(stab_thr, 1), 1.0)
    bar_w = w - 24
    cv2.rectangle(frame, (12, 66), (12 + bar_w, 76), (30, 50, 50), -1)
    fill_w = int(bar_w * stable_pct)
    if fill_w > 0:
        bar_col = (0, int(180 * stable_pct + 40), int(120 * stable_pct + 80))
        cv2.rectangle(frame, (12, 66), (12 + fill_w, 76), bar_col, -1)

    # Two-hand debug strip
    if mhr is not None:
        n     = mhr.count
        l_txt = "L:OK" if mhr.left_detected  else "L:--"
        r_txt = "R:OK" if mhr.right_detected else "R:--"
        n_col = (0, 220, 160) if n > 0              else (80, 130, 130)
        l_col = (0, 220, 160) if mhr.left_detected  else (80, 130, 130)
        r_col = (0, 220, 160) if mhr.right_detected else (80, 130, 130)
        strip_y = 38
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.40
        thick = 1
        cv2.putText(frame, f"Hands:{n}", (w - 216, strip_y), font, scale, n_col, thick)
        cv2.putText(frame, l_txt,        (w - 148, strip_y), font, scale, l_col, thick)
        cv2.putText(frame, r_txt,        (w - 90,  strip_y), font, scale, r_col, thick)

    # Bottom sentence strip
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 44), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)

    if mode == "Character":
        sentence_txt = "".join(sentence) if sentence else "—"
        cv2.putText(frame, f"Text: {sentence_txt}", (10, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 220, 210), 1)
    else:
        sentence_txt = " ".join(sentence) if sentence else "—"
        cv2.putText(frame, f"Sentence: {sentence_txt}", (10, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 220, 210), 1)


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC frame callback  (background thread, ~30×/sec)
# ─────────────────────────────────────────────────────────────────────────────

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Process one video frame from the browser WebRTC stream.
    Both models receive exactly 42 normalised MediaPipe hand-landmark features.
    """
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    with _lock:
        current_mode = _state["mode"]

    if detector is None:
        return frame

    mhr, annotated = detector.find_all_landmarks(img, draw=True)
    has_hand = mhr.dominant is not None
    norm_lm  = normalise_landmarks(mhr.dominant.landmarks) if has_hand else None

    gesture    = None
    confidence = 0.0

    # ── WORD MODE ──────────────────────────────────────────────────────────────
    if current_mode == "Word":
        if interp is None:
            _write_hand_state_locked(mhr)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        _conf_thr = st.session_state.get("conf_threshold", WORD_CONFIDENCE_THRESHOLD)
        _stab_thr = st.session_state.get("stable_frames_cfg", STABLE_FRAME_COUNT)

        if norm_lm is not None:
            raw_pred, conf = predict(norm_lm, interp, in_idx, out_idx, label_map, scaler)
            if conf >= _conf_thr:
                gesture, confidence = smoother.update(raw_pred, conf)
            else:
                smoother.reset()
                gesture = None
        else:
            smoother.reset()

        with _lock:
            _update_fps()
            _state["frame_count"] += 1
            _state["hands_count"]    = mhr.count
            _state["left_detected"]  = mhr.left_detected
            _state["right_detected"] = mhr.right_detected

            if gesture:
                if gesture == _state["last_stable"]:
                    _state["stable_count"] += 1
                else:
                    _state["stable_count"] = 1
                    _state["last_stable"]  = gesture

                if _state["stable_count"] == _stab_thr:
                    words = _state["sentence"]
                    if not words or words[-1] != gesture:
                        _state["sentence"].append(gesture)
            else:
                _state["stable_count"] = 0
                _state["last_stable"]  = None

            _state["gesture"]    = gesture or "—"
            _state["confidence"] = confidence

        with _lock:
            _word_sent = list(_state["sentence"])

        _draw_overlay(annotated, gesture, confidence,
                      _state["stable_count"], _stab_thr,
                      _word_sent, mode="Word", mhr=mhr)

    # ── CHARACTER MODE ─────────────────────────────────────────────────────────
    else:
        if char_interp is None:
            _write_hand_state_locked(mhr)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        _conf_thr = st.session_state.get("char_conf_threshold", CHARACTER_CONFIDENCE_THRESHOLD)
        _stab_thr = st.session_state.get("char_stable_frames_cfg", CHARACTER_STABLE_FRAME_COUNT)

        if norm_lm is not None:
            raw_pred, conf = predict_character(
                norm_lm, char_interp, char_in_idx, char_out_idx,
                char_label_map, char_scaler,
            )
            if conf >= _conf_thr:
                gesture, confidence = char_smoother.update(raw_pred, conf)
            else:
                char_smoother.reset()
                gesture = None
        else:
            char_smoother.reset()
            gesture = None

        with _lock:
            _update_fps()
            _state["frame_count"] += 1
            _state["hands_count"]    = mhr.count
            _state["left_detected"]  = mhr.left_detected
            _state["right_detected"] = mhr.right_detected

            if gesture:
                if gesture == _state["last_stable"]:
                    _state["stable_count"] += 1
                else:
                    _state["stable_count"] = 1
                    _state["last_stable"]  = gesture

                if _state["stable_count"] >= _stab_thr:
                    if _state["char_debounce_count"] >= CHARACTER_DEBOUNCE_FRAMES:
                        if _state["char_last_accepted"] != gesture:
                            _state["char_sentence"].append(gesture)
                            _state["char_last_accepted"] = gesture
                        _state["char_debounce_count"] = 0
                    else:
                        _state["char_debounce_count"] += 1
                else:
                    _state["char_debounce_count"] += 1
            else:
                _state["stable_count"] = 0
                _state["last_stable"]  = None
                _state["char_debounce_count"] = min(
                    _state["char_debounce_count"] + 1,
                    CHARACTER_DEBOUNCE_FRAMES + 5,
                )

            _state["gesture"]    = gesture or "—"
            _state["confidence"] = confidence

        with _lock:
            _char_sent = list(_state["char_sentence"])

        _draw_overlay(annotated, gesture, confidence,
                      _state["stable_count"], _stab_thr,
                      _char_sent, mode="Character", mhr=mhr)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─────────────────────────────────────────────────────────────────────────────
# Cloud-only TTS helpers (Web Speech API)
# ─────────────────────────────────────────────────────────────────────────────

def _tts_js(text: str) -> str:
    """Return an HTML button that triggers Web Speech API TTS on click."""
    safe = text.replace("'", "\\'").replace('"', '\\"')
    return f"""
    <button onclick="(function(){{
        if(!'{safe}')return;
        var u=new SpeechSynthesisUtterance('{safe}');
        u.rate=0.95;u.pitch=1.0;u.volume=1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
    }})();"
      style="width:100%;background:linear-gradient(135deg,#83A6CE,#C48CB3);
        color:#fff;border:none;border-radius:12px;padding:11px 0;
        font-size:0.9rem;font-weight:700;cursor:pointer;
        box-shadow:0 4px 16px rgba(131,166,206,0.4);
        transition:all 0.15s;font-family:'Inter',sans-serif;
        letter-spacing:0.2px;">
      🔊 Speak
    </button>"""


def _disabled_btn_html(label: str) -> str:
    return f"""
    <button disabled style="width:100%;background:rgba(40,65,85,0.05);
      color:#6b8aaa;border:1px solid rgba(131,166,206,0.2);border-radius:12px;
      padding:11px 0;font-size:0.9rem;font-weight:600;cursor:not-allowed;opacity:0.55;
      font-family:'Inter',sans-serif;">{label}</button>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (identical to app.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0 24px;">
          {svg_hand_3d()}
          <div style="font-size:1.2rem;font-weight:900;color:#FFFFFF;
                      margin-top:12px;letter-spacing:-0.2px;">
            SignFlow
          </div>
          <div style="font-size:0.72rem;font-weight:600;margin-top:4px;
                      background:linear-gradient(135deg,#83A6CE,#C48CB3);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Sign Language Translator
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-top:1px solid rgba(131,166,206,0.2);">', unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────────────────────
        for page in _NAV:
            selected = st.session_state["page"] == page
            btn_type = "primary" if selected else "secondary"
            if st.button(page, key=f"nav_{page}", use_container_width=True, type=btn_type):
                st.session_state["page"] = page
                st.rerun()

        st.markdown('<hr style="border-top:1px solid rgba(131,166,206,0.2);">', unsafe_allow_html=True)

        # ── Live stats ─────────────────────────────────────────────────────────
        with _lock:
            _g   = _state["gesture"]
            _c   = _state["confidence"]
            _fps = _state["fps"]

        analytics: SessionAnalytics = st.session_state["analytics"]
        snap = analytics.snapshot()

        st.markdown('<div class="clay-section-title">Live Stats</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:8px;">
              <div class="stat-label">FPS</div>
              <div class="stat-value">{_fps:.1f}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Gestures</div>
              <div class="stat-value">{snap.total_gestures}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:8px;">
              <div class="stat-label">Conf</div>
              <div class="stat-value">{_c:.0%}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Time</div>
              <div class="stat-value" style="font-size:0.82rem;">{analytics.format_duration()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr style="border-top:1px solid rgba(131,166,206,0.2);">', unsafe_allow_html=True)

        # ── Camera + model status ──────────────────────────────────────────────
        with _lock:
            _running = len(_state["sentence"]) > 0 or _state["gesture"] != "—"

        if _running:
            st.markdown('<span class="badge badge-green">● Camera Active</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">○ Camera Off</span>',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        if st.session_state["model_loaded"]:
            st.markdown('<span class="badge badge-green">✓ Model Ready</span>',
                        unsafe_allow_html=True)
        elif st.session_state["error_msg"]:
            st.markdown('<span class="badge badge-red">✕ Model Error</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">… Loading</span>',
                        unsafe_allow_html=True)

        st.markdown('<hr style="border-top:1px solid rgba(131,166,206,0.2);">', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:rgba(131,166,206,0.5);font-size:0.68rem;'
            'font-weight:600;text-align:center;letter-spacing:0.5px;">v2.0 · AI Edition</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Live Translator page  (cloud-specific — WebRTC + Web Speech API)
# ─────────────────────────────────────────────────────────────────────────────

def page_live_translator() -> None:
    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:24px;">
      <div style="font-size:0.72rem;font-weight:700;letter-spacing:2.5px;
                  text-transform:uppercase;color:#83A6CE;margin-bottom:8px;">
        AI Recognition
      </div>
      <h1 style="font-size:2rem;font-weight:900;color:#284155;margin:0 0 6px;
                 letter-spacing:-0.3px;">Live Translator</h1>
      <p style="color:#6b8aaa;font-size:0.9rem;margin:0;">
        Real-time hand gesture recognition — hold a sign steady to add it to your sentence.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Error banner ───────────────────────────────────────────────────────────
    if st.session_state["error_msg"]:
        st.markdown(
            '<div class="clay-error">⚠ Word model unavailable: ' +
            st.session_state["error_msg"].replace("\n", " ") + '</div>',
            unsafe_allow_html=True,
        )

    # ── Mode selector ──────────────────────────────────────────────────────────
    mode_col1, mode_col2, _ = st.columns([2, 2, 4], gap="small")

    with mode_col1:
        if st.button(
            "🤙 Word Mode", use_container_width=True, key="btn_mode_word",
            type="primary" if st.session_state["recognition_mode"] == "Word" else "secondary",
        ):
            st.session_state["recognition_mode"] = "Word"
            with _lock:
                _state["mode"]         = "Word"
                _state["stable_count"] = 0
                _state["last_stable"]  = None
            smoother.reset()
            st.rerun()

    with mode_col2:
        char_avail = char_interp is not None
        if st.button(
            "🔤 Character Mode", use_container_width=True, key="btn_mode_char",
            type="primary" if st.session_state["recognition_mode"] == "Character" else "secondary",
            disabled=not char_avail,
        ):
            st.session_state["recognition_mode"] = "Character"
            with _lock:
                _state["mode"]         = "Character"
                _state["stable_count"] = 0
                _state["last_stable"]  = None
            char_smoother.reset()
            st.rerun()

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with _lock:
        _state["mode"] = st.session_state["recognition_mode"]

    _mode_now = st.session_state["recognition_mode"]

    # ── Camera — full width ─────────────────────────────────────────────────────
    model_ready = (interp is not None) or (char_interp is not None)

    st.markdown(
        '<div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.08em;color:#83A6CE;margin-bottom:10px;">Camera Feed</div>',
        unsafe_allow_html=True,
    )

    if model_ready:
        st.markdown('<div class="camera-wrapper">', unsafe_allow_html=True)
        webrtc_streamer(
            key="sign-lang",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": 1280, "height": 720},
                "audio": False,
            },
            async_processing=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:linear-gradient(135deg,#f0f4f8,#e8eef5);"
            "border:2px dashed rgba(131,166,206,0.3);"
            "border-radius:22px;height:400px;display:flex;align-items:center;"
            "justify-content:center;flex-direction:column;gap:14px;"
            "box-shadow:0 4px 20px rgba(40,65,85,0.07);'>"
            "<span style='font-size:3rem;'>📷</span>"
            "<span style='color:#284155;font-weight:700;font-size:1rem;'>"
            "No model loaded — run the training pipeline first</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='color:#6b8aaa;font-size:0.8rem;margin-top:8px;"
        "text-align:center;font-weight:500;'>"
        "💡 Good lighting and a plain background improve accuracy</div>",
        unsafe_allow_html=True,
    )

    # Subtitle strip below camera
    if st.session_state["subtitles"]:
        with _lock:
            _mode_sub = _state["mode"]
            subtitle_text = (
                "".join(_state["char_sentence"])
                if _mode_sub == "Character"
                else " ".join(_state["sentence"])
            )
        if subtitle_text:
            st.markdown(
                f'<div class="subtitle-strip">📝 {subtitle_text}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── Translation panel below camera ─────────────────────────────────────────
    with _lock:
        _g  = _state["gesture"]
        _c  = _state["confidence"]
        _sf = _state["stable_count"]

        if _mode_now == "Character":
            _chars    = list(_state["char_sentence"])
            _stab_cfg = st.session_state.get("char_stable_frames_cfg", CHARACTER_STABLE_FRAME_COUNT)
        else:
            _words    = list(_state["sentence"])
            _stab_cfg = st.session_state.get("stable_frames_cfg", STABLE_FRAME_COUNT)

    conf_pct   = int(_c * 100)
    stable_pct = int(min(_sf / max(_stab_cfg, 1), 1.0) * 100)

    if _c >= 0.75:
        fill_cls = "conf-fill-high"
    elif _c >= 0.5:
        fill_cls = "conf-fill-medium"
    else:
        fill_cls = "conf-fill-low"

    meaning = GESTURE_MEANINGS.get(_g, "") if _mode_now == "Word" else ""

    mode_badge_html = (
        '<span class="mode-badge-word">Word Mode</span>'
        if _mode_now == "Word"
        else '<span class="mode-badge-char">Character Mode</span>'
    )

    # Two-column: prediction card | sentence builder
    pred_col, sent_col = st.columns([1, 2], gap="large")

    with pred_col:
        st.markdown(f"""
        <div class="prediction-card">
          <div class="prediction-label">Current Prediction</div>
          {mode_badge_html}
          <div class="prediction-gesture">{_g}</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);
                      font-style:italic;margin:4px 0 16px;">{meaning}</div>
          <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
              <span style="font-size:0.7rem;color:rgba(255,255,255,0.45);
                           font-weight:700;text-transform:uppercase;
                           letter-spacing:1.2px;">Confidence</span>
              <span style="font-size:0.85rem;color:#FFFFFF;font-weight:800;">{conf_pct}%</span>
            </div>
            <div class="conf-track">
              <div class="{fill_cls}" style="width:{conf_pct}%;transition:width 0.4s;"></div>
            </div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
              <span style="font-size:0.7rem;color:rgba(255,255,255,0.45);
                           font-weight:700;text-transform:uppercase;
                           letter-spacing:1.2px;">Stability</span>
              <span style="font-size:0.85rem;color:#FFFFFF;font-weight:800;">{_sf}/{_stab_cfg}</span>
            </div>
            <div class="conf-track">
              <div style="width:{stable_pct}%;height:100%;
                          background:linear-gradient(90deg,rgba(229,201,215,0.7),#E5C9D7);
                          border-radius:8px;transition:width 0.3s;"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with sent_col:
        # ── WORD MODE sentence builder ─────────────────────────────────────────
        if _mode_now == "Word":
            with _lock:
                _words = list(_state["sentence"])

            sentence_str = (
                " ".join(_words) if _words
                else "<span style='color:#6b8aaa;font-style:italic;'>Waiting for gestures…</span>"
            )
            st.markdown(f"""
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#83A6CE;margin-bottom:8px;">Sentence Builder</div>
            <div class="sentence-box">{sentence_str}</div>
            """, unsafe_allow_html=True)

            if _words and st.session_state["language"] != DEFAULT_LANGUAGE:
                tl = translate_sentence(_words, st.session_state["language"])
                st.markdown(f"""
                <div class="trans-card" style="margin-top:8px;">
                  <div class="trans-lang">{st.session_state["language"]}</div>
                  <div class="trans-text">{tl}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            speak_text = " ".join(_words) if _words else ""
            btn_speak, btn_undo, btn_clear = st.columns(3)

            with btn_speak:
                if speak_text:
                    st.markdown(_tts_js(speak_text), unsafe_allow_html=True)
                else:
                    st.markdown(_disabled_btn_html("🔊 Speak"), unsafe_allow_html=True)

            with btn_undo:
                if st.button("⬅ Undo", use_container_width=True,
                             disabled=not _words, key="word_btn_undo"):
                    with _lock:
                        if _state["sentence"]:
                            _state["sentence"].pop()
                    st.rerun()

            with btn_clear:
                if st.button("🗑 Clear", use_container_width=True,
                             disabled=not _words, key="word_btn_clear"):
                    with _lock:
                        _state["sentence"]     = []
                        _state["last_stable"]  = None
                        _state["stable_count"] = 0
                    smoother.reset()
                    st.session_state["_last_auto_spoken"] = ""
                    st.rerun()

            # Auto-speak
            if st.session_state["auto_speak"] and _words:
                latest = _words[-1]
                if st.session_state.get("_last_auto_spoken") != latest:
                    st.session_state["_last_auto_spoken"] = latest
                    tts_word = latest.replace("'", "\\'")
                    st.markdown(f"""
                    <script>
                    (function(){{
                      var u=new SpeechSynthesisUtterance('{tts_word}');
                      u.rate=1.0;u.pitch=1.0;u.volume=1.0;
                      window.speechSynthesis.cancel();
                      window.speechSynthesis.speak(u);
                    }})();
                    </script>
                    """, unsafe_allow_html=True)

            # Word chips
            if _words:
                st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
                chips = "".join(
                    f'<span style="display:inline-block;background:rgba(131,166,206,0.12);'
                    f'border:1px solid rgba(131,166,206,0.28);border-radius:50px;'
                    f'padding:4px 14px;margin:3px;font-size:0.82rem;color:#284155;'
                    f'font-weight:600;">{w}</span>'
                    for w in reversed(_words[-20:])
                )
                st.markdown(f'<div style="line-height:2.4;">{chips}</div>', unsafe_allow_html=True)

        # ── CHARACTER MODE sentence builder ───────────────────────────────────
        else:
            with _lock:
                _chars = list(_state["char_sentence"])

            char_display = "".join(_chars) if _chars else ""
            char_box_content = (
                f'<span style="letter-spacing:0.12em;font-size:1.2rem;">{char_display}</span>'
                if char_display
                else "<span style='color:#6b8aaa;font-style:italic;'>Waiting for characters…</span>"
            )
            st.markdown(f"""
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#83A6CE;margin-bottom:8px;">Character Builder</div>
            <div class="sentence-box" style="min-height:52px;font-family:monospace;">
              {char_box_content}
            </div>
            """, unsafe_allow_html=True)

            if char_display.strip() and st.session_state["language"] != DEFAULT_LANGUAGE:
                tl = translate(char_display.strip(), st.session_state["language"], "en")
                st.markdown(f"""
                <div class="trans-card" style="margin-top:8px;">
                  <div class="trans-lang">{st.session_state["language"]}</div>
                  <div class="trans-text">{tl}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            speak_text_char = char_display.strip()
            c_speak, c_space, c_back = st.columns(3)

            with c_speak:
                if speak_text_char:
                    st.markdown(_tts_js(speak_text_char), unsafe_allow_html=True)
                else:
                    st.markdown(_disabled_btn_html("🔊 Speak"), unsafe_allow_html=True)

            with c_space:
                if st.button("⎵ Space", use_container_width=True, key="char_btn_space"):
                    with _lock:
                        if _state["char_sentence"] and _state["char_sentence"][-1] != " ":
                            _state["char_sentence"].append(" ")
                            _state["char_last_accepted"] = " "
                    st.rerun()

            with c_back:
                if st.button("⌫ Backspace", use_container_width=True,
                             disabled=not _chars, key="char_btn_back"):
                    with _lock:
                        if _state["char_sentence"]:
                            _state["char_sentence"].pop()
                            _state["char_last_accepted"] = (
                                _state["char_sentence"][-1]
                                if _state["char_sentence"] else None
                            )
                    st.rerun()

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            if st.button("🗑 Clear All", use_container_width=True,
                         disabled=not _chars, key="char_btn_clear"):
                with _lock:
                    _state["char_sentence"]       = []
                    _state["char_last_accepted"]  = None
                    _state["char_debounce_count"] = 0
                    _state["stable_count"]        = 0
                    _state["last_stable"]         = None
                char_smoother.reset()
                st.rerun()

            if _chars:
                st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
                chips = "".join(
                    f'<span style="display:inline-block;background:rgba(196,140,179,0.12);'
                    f'border:1px solid rgba(196,140,179,0.28);border-radius:50px;'
                    f'padding:4px 14px;margin:3px;font-size:0.82rem;color:#284155;'
                    f'font-weight:600;">{c if c != " " else "⎵"}</span>'
                    for c in _chars[-30:]
                )
                st.markdown(f'<div style="line-height:2.4;">{chips}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main router
# ─────────────────────────────────────────────────────────────────────────────
render_sidebar()

page = st.session_state["page"]

if page == "🏠 Home":
    show_home()
elif page == "📷 Live Translator":
    page_live_translator()
elif page == "📜 History":
    show_history(history_key="history")
elif page == "ℹ️ About":
    show_about()
# Fallback routes (not in nav but accessible programmatically)
elif page == "📊 Analytics":
    show_analytics(
        analytics_key="analytics",
        history_key="history",
        use_plotly=False,
    )
elif page == "⚙️ Settings":
    show_settings_cloud()
else:
    show_home()
