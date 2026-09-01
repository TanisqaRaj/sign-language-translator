# =============================================================================
# app_cloud.py  –  Cloud-compatible Streamlit UI  (Claymorphism Edition)
# =============================================================================
# Works on: Streamlit Cloud, Hugging Face Spaces, Render, Railway, Docker
#
# Key differences vs app.py:
#   • streamlit-webrtc  → browser webcam  (no cv2.VideoCapture)
#   • Web Speech API JS → browser TTS     (no pyttsx3)
#   • tflite-runtime    → lighter install (no full TensorFlow)
#
# Two independent models:
#   Model 1 (Word)      : 42 hand-landmark features → 6 words
#   Model 2 (Character) : 42 hand-landmark features → 36 chars (A-Z + 0-9)
#
# Pages: Home · Live Translator · History · Analytics · Settings · About
#
# Install extras:  pip install streamlit-webrtc aiortc
# Run:             streamlit run app_cloud.py
# =============================================================================

from __future__ import annotations

import os
import sys
import time
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
    GESTURE_MEANINGS,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    NAV_PAGES,
    DEFAULT_THEME,
    DEFAULT_FONT_SIZE,
    DEFAULT_HIGH_CONTRAST,
    AUTO_SPEAK_DEFAULT,
    SUBTITLE_ENABLED_DEFAULT,
    LANDMARK_FEATURES,
    # ── Character model config ────────────────────────────────────────────────
    CHARACTER_CONFIDENCE_THRESHOLD,
    CHARACTER_STABLE_FRAME_COUNT,
    CHARACTER_DEBOUNCE_FRAMES,
    WORD_CONFIDENCE_THRESHOLD,
    CHARACTER_LABELS,
)
from utils.mediapipe_helper import HandDetector, MultiHandResult
from utils.analytics import SessionAnalytics
from utils.gesture_history import GestureHistory
from utils.translator import translate, translate_sentence
from preprocess import normalise_landmarks
from inference import (
    load_tflite_model, load_label_map, load_scaler,
    predict, PredictionSmoother,
    # ── Character model (Model 2) ─────────────────────────────────────────────
    load_character_tflite_model, load_character_label_map, load_character_scaler,
    predict_character,
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
# WebRTC ICE / STUN config
# ─────────────────────────────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ─────────────────────────────────────────────────────────────────────────────
# Claymorphism design system
# ─────────────────────────────────────────────────────────────────────────────
from utils.ui_components import (
    inject_css,
    svg_hand_3d, svg_ai_robot, svg_speaker, svg_analytics,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    # Navigation
    "page":              NAV_PAGES[0],
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
    "char_conf_threshold":    CHARACTER_CONFIDENCE_THRESHOLD,   # 0.60
    "char_stable_frames_cfg": CHARACTER_STABLE_FRAME_COUNT,     # 12
    # Auto-speak tracking (prevents speaking same word twice)
    "_last_auto_spoken": "",
    # Objects (created below after cache)
    "analytics":         None,
    "history":           None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Lazy-init object instances
if st.session_state["analytics"] is None:
    _a = SessionAnalytics()
    _a.start_session()
    st.session_state["analytics"] = _a

if st.session_state["history"] is None:
    st.session_state["history"] = GestureHistory()

# Inject CSS (reads settings from session_state)
inject_css(
    theme=st.session_state["theme"],
    large_font=(st.session_state["font_size"] == "large"),
    high_contrast=st.session_state["high_contrast"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached once per server process)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading word model…")
def get_resources():
    """
    Load word model (Model 1) — cached once. Never loads or touches character model.

    Word model uses 42 hand-landmark features (MediaPipe only).
    The word scaler (models/scaler.pkl) was fitted on 42 features.
    No MoveNet / hybrid features are needed in the cloud app.

    NOTE: The HandDetector is NOT created here — it is shared and created
    separately in get_hand_detector() so that character mode works even if
    the word model fails to load.
    """
    try:
        interp, in_idx, out_idx = load_tflite_model()
        label_map = load_label_map()
        scaler    = load_scaler()

        # Validate word model input shape
        input_shape = tuple(interp.get_input_details()[0]["shape"])
        if input_shape[1] != LANDMARK_FEATURES:
            raise ValueError(
                f"Word model input shape mismatch!\n"
                f"  Expected : (1, {LANDMARK_FEATURES})\n"
                f"  Got      : {input_shape}\n"
                f"  The existing gesture_model.tflite was trained on "
                f"{input_shape[1]} features (hybrid MediaPipe+MoveNet).\n"
                "  Re-run train_model.py (hand-only, 42 features) and "
                "convert_tflite.py to fix this."
            )
        return interp, in_idx, out_idx, label_map, scaler, None
    except FileNotFoundError as exc:
        return None, None, None, None, None, str(exc)
    except ValueError as exc:
        return None, None, None, None, None, str(exc)


@st.cache_resource(show_spinner="⏳ Initialising hand detector…")
def get_hand_detector():
    """
    Shared HandDetector instance — loaded once, independent of both models.

    Creating this separately from get_resources() means character mode works
    even when the word model is not trained / shape-mismatched.
    """
    return HandDetector()


@st.cache_resource(show_spinner="⏳ Loading character model…")
def get_char_resources():
    """
    Load character model (Model 2) — cached once, completely independent.

    Character model uses 42 hand-landmark features (MediaPipe only).
    Uses character_scaler.pkl — never the word scaler.

    Returns
    -------
    (interp, in_idx, out_idx, label_map, scaler, error_msg)
    """
    try:
        interp, in_idx, out_idx = load_character_tflite_model()
        label_map = load_character_label_map()
        scaler    = load_character_scaler()
        return interp, in_idx, out_idx, label_map, scaler, None
    except FileNotFoundError as exc:
        return None, None, None, None, None, str(exc)
    except ValueError as exc:
        return None, None, None, None, None, str(exc)


# ── Load word model ────────────────────────────────────────────────────────────
interp, in_idx, out_idx, label_map, scaler, _load_err = get_resources()
if interp is not None:
    st.session_state["model_loaded"] = True
if _load_err and not st.session_state["error_msg"]:
    st.session_state["error_msg"] = _load_err

# ── Shared hand detector (independent of both models) ────────────────────────
detector = get_hand_detector()

# ── Load character model ───────────────────────────────────────────────────────
(
    char_interp, char_in_idx, char_out_idx,
    char_label_map, char_scaler, _char_load_err,
) = get_char_resources()
if char_interp is not None:
    st.session_state["char_model_loaded"] = True

# Independent smoothers — one per model, never mixed
smoother      = PredictionSmoother()
char_smoother = PredictionSmoother()

# ─────────────────────────────────────────────────────────────────────────────
# Shared mutable state for the WebRTC background thread
# (st.session_state is NOT safe to write from background threads)
# ─────────────────────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state: dict = {
    # ── Shared ────────────────────────────────────────────────────────────────
    "gesture":        "—",
    "confidence":     0.0,
    "stable_count":   0,
    "last_stable":    None,
    "fps":            0.0,
    "frame_count":    0,
    "t_last":         time.time(),
    # ── Two-hand detection status (updated every frame) ───────────────────────
    "hands_count":    0,
    "left_detected":  False,
    "right_detected": False,
    # ── Word mode sentence (list of words) ────────────────────────────────────
    "sentence":       [],
    # ── Character mode state ───────────────────────────────────────────────────
    # char_sentence: list of characters, e.g. ["H", "E", " ", "L", "L", "O"]
    # Space characters are stored as " " for joined display.
    "char_sentence":        [],
    # Debounce: frames since last accepted character (must reach DEBOUNCE_FRAMES
    # before a new character can be accepted — prevents HHHEELLLOO duplicates).
    "char_debounce_count":  0,
    # The last character that was accepted (prevents re-accepting same char
    # without a break in detection).
    "char_last_accepted":   None,
    # Recognition mode — mirrors session_state but safe for background thread
    "mode":                 "Word",
}


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC frame callback  (background thread, ~30×/sec)
# ─────────────────────────────────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Process one video frame from the browser WebRTC stream.

    Both models receive exactly 42 normalised MediaPipe hand-landmark features.
    The word model scaler (models/scaler.pkl) was fitted on 42 features.
    The character model scaler (models/character_scaler.pkl) was also fitted
    on 42 features.  Neither requires MoveNet pose data in this app.

    Two-hand behaviour
    ------------------
    Both hand skeletons are drawn when visible.  Inference always runs on the
    *dominant* hand (right preferred, left fallback) so neither model ever
    receives a different input shape.  Hand-detection status (count, L/R) is
    written to _state and shown on the HUD overlay.

    Below-threshold predictions are discarded (→ "Unknown") per
    WORD_CONFIDENCE_THRESHOLD and CHARACTER_CONFIDENCE_THRESHOLD from config.
    """
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    with _lock:
        current_mode = _state["mode"]

    if detector is None:
        return frame

    # ── Two-hand-aware MediaPipe extraction ───────────────────────────────────
    # find_all_landmarks() draws ALL detected skeletons onto the frame and
    # returns a MultiHandResult with .dominant, .left, .right, .count etc.
    # Inference only uses mhr.dominant.landmarks (42 floats) — model unchanged.
    mhr, annotated = detector.find_all_landmarks(img, draw=True)
    has_hand = mhr.dominant is not None
    norm_lm  = normalise_landmarks(mhr.dominant.landmarks) if has_hand else None

    gesture    = None
    confidence = 0.0

    # ── WORD MODE ─────────────────────────────────────────────────────────────
    if current_mode == "Word":
        if interp is None:
            _write_hand_state_locked(mhr)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        _conf_thr = st.session_state.get("conf_threshold", WORD_CONFIDENCE_THRESHOLD)
        _stab_thr = st.session_state.get("stable_frames_cfg", STABLE_FRAME_COUNT)

        if norm_lm is not None:
            raw_pred, conf = predict(
                norm_lm, interp, in_idx, out_idx, label_map, scaler
            )
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
            # Hand detection status
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

        _draw_overlay(
            annotated, gesture, confidence,
            _state["stable_count"], _stab_thr,
            _word_sent, mode="Word", mhr=mhr,
        )

    # ── CHARACTER MODE ────────────────────────────────────────────────────────
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
            # Hand detection status
            _state["hands_count"]    = mhr.count
            _state["left_detected"]  = mhr.left_detected
            _state["right_detected"] = mhr.right_detected

            if gesture:
                if gesture == _state["last_stable"]:
                    _state["stable_count"] += 1
                else:
                    _state["stable_count"] = 1
                    _state["last_stable"]  = gesture

                # Character accepted once stability + debounce cooldown are met
                if _state["stable_count"] >= _stab_thr:
                    if _state["char_debounce_count"] >= CHARACTER_DEBOUNCE_FRAMES:
                        # Accept only if different from last accepted char
                        # (user must lower hand + re-sign to repeat same letter)
                        if _state["char_last_accepted"] != gesture:
                            _state["char_sentence"].append(gesture)
                            _state["char_last_accepted"] = gesture
                        _state["char_debounce_count"] = 0
                    else:
                        _state["char_debounce_count"] += 1
                else:
                    _state["char_debounce_count"] += 1
            else:
                # Hand absent or low confidence — accumulate debounce "cooldown"
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

        _draw_overlay(
            annotated, gesture, confidence,
            _state["stable_count"], _stab_thr,
            _char_sent, mode="Character", mhr=mhr,
        )

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def _write_hand_state_locked(mhr: "MultiHandResult") -> None:
    """Write hand-detection counts into _state under the lock. DRY helper."""
    with _lock:
        _state["hands_count"]    = mhr.count
        _state["left_detected"]  = mhr.left_detected
        _state["right_detected"] = mhr.right_detected


def _update_fps() -> None:
    """Update EMA FPS in _state. Must be called inside _lock."""
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
    mhr: "MultiHandResult | None" = None,
) -> None:
    """Draw the HUD overlay onto the frame in-place."""
    h, w = frame.shape[:2]

    # Top banner
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 82), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # Mode badge (top-right)
    mode_color = (255, 140, 0) if mode == "Character" else (100, 180, 255)
    cv2.putText(frame, f"[{mode}]", (w - 120, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

    color = (0, 220, 0) if gesture else (80, 80, 220)
    label = gesture if gesture else "No gesture"
    cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    if gesture:
        cv2.putText(frame, f"{confidence:.0%} confidence", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Stability bar
    stable_pct = min(stable_count / max(stab_thr, 1), 1.0)
    bar_w = w - 24
    cv2.rectangle(frame, (12, 66), (12 + bar_w, 76), (40, 40, 40), -1)
    fill_w = int(bar_w * stable_pct)
    if fill_w > 0:
        bar_col = (0, int(200 * stable_pct), int(200 * (1 - stable_pct)))
        cv2.rectangle(frame, (12, 66), (12 + fill_w, 76), bar_col, -1)

    # ── Two-hand detection debug strip ───────────────────────────────────────
    # Draws: "Hands:N  L:OK/--  R:OK/--" at the top-right below the mode badge.
    # Green = detected, blue-grey = not detected.
    if mhr is not None:
        n     = mhr.count
        l_txt = "L:OK" if mhr.left_detected  else "L:--"
        r_txt = "R:OK" if mhr.right_detected else "R:--"
        n_col = (0, 220, 0) if n > 0              else (80, 80, 200)
        l_col = (0, 220, 0) if mhr.left_detected  else (80, 80, 200)
        r_col = (0, 220, 0) if mhr.right_detected else (80, 80, 200)
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
    else:
        sentence_txt = " ".join(sentence) if sentence else "—"
        cv2.putText(frame, f"Sentence: {sentence_txt}", (10, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        # Brand block
        st.markdown(f"""
        <div style="text-align:center;padding:14px 0 20px;">
          {svg_hand_3d()}
          <div style="font-size:1.15rem;font-weight:800;
                      background:linear-gradient(135deg,#818cf8,#60a5fa);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      margin-top:8px;">ISL Translator</div>
          <div style="font-size:0.72rem;color:#475569;font-weight:600;margin-top:2px;">
            Claymorphism Edition
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

        # Model status badges
        if st.session_state["model_loaded"]:
            st.markdown('<span class="badge badge-green">✅ Word Model Ready</span>',
                        unsafe_allow_html=True)
        elif st.session_state["error_msg"]:
            st.markdown(
                '<span class="badge badge-red" title="Word model failed to load — '
                'character mode still works">⚠️ Word Model Error</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="badge badge-yellow">⏳ Loading…</span>',
                        unsafe_allow_html=True)

        if st.session_state["char_model_loaded"]:
            st.markdown('<span class="badge badge-green">✅ Char Model Ready</span>',
                        unsafe_allow_html=True)
        elif _char_load_err:
            st.markdown('<span class="badge badge-yellow">⚠️ Char Model Not Trained</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">⏳ Loading…</span>',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

        # Navigation
        st.markdown('<div class="clay-section-title">Navigation</div>',
                    unsafe_allow_html=True)
        for page in NAV_PAGES:
            if st.button(page, key=f"nav_{page}", use_container_width=True):
                st.session_state["page"] = page
                st.rerun()

        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

        # Live mini-stats
        with _lock:
            _g    = _state["gesture"]
            _c    = _state["confidence"]
            _sf   = _state["stable_count"]
            _fps  = _state["fps"]
            _hn   = _state["hands_count"]
            _l_ok = _state["left_detected"]
            _r_ok = _state["right_detected"]

        _mode_sb = st.session_state.get("recognition_mode", "Word")
        _stab_cfg_sb = (
            st.session_state.get("char_stable_frames_cfg", CHARACTER_STABLE_FRAME_COUNT)
            if _mode_sb == "Character"
            else st.session_state.get("stable_frames_cfg", STABLE_FRAME_COUNT)
        )
        st.markdown('<div class="clay-section-title">📊 Live Stats</div>',
                    unsafe_allow_html=True)
        _l_badge = "🟢 L" if _l_ok else "⚫ L"
        _r_badge = "🟢 R" if _r_ok else "⚫ R"
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
          <div class="stat-card"><div class="stat-label">Gesture</div>
            <div class="stat-value" style="font-size:0.9rem;">{_g}</div></div>
          <div class="stat-card"><div class="stat-label">Conf</div>
            <div class="stat-value">{_c:.0%}</div></div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
          <div class="stat-card"><div class="stat-label">Stability</div>
            <div class="stat-value">{_sf}/{_stab_cfg_sb}</div></div>
          <div class="stat-card"><div class="stat-label">FPS</div>
            <div class="stat-value">{_fps:.0f}</div></div>
        </div>
        <div class="stat-card" style="margin-bottom:4px;">
          <div class="stat-label">Hands Detected: {_hn}</div>
          <div style="display:flex;gap:12px;margin-top:4px;">
            <span style="font-size:0.82rem;font-weight:700;
              color:{'#34d399' if _l_ok else '#475569'};">{_l_badge}</span>
            <span style="font-size:0.82rem;font-weight:700;
              color:{'#34d399' if _r_ok else '#475569'};">{_r_badge}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

        # Supported gestures / characters
        if _mode_sb == "Word":
            st.markdown('<div class="clay-section-title">🤙 Gestures</div>',
                        unsafe_allow_html=True)
            chips = "".join(f'<span class="gesture-chip">{g}</span>' for g in GESTURE_LABELS)
        else:
            st.markdown('<div class="clay-section-title">🔤 Characters</div>',
                        unsafe_allow_html=True)
            chips = "".join(f'<span class="gesture-chip">{c}</span>' for c in CHARACTER_LABELS)
        st.markdown(f'<div style="line-height:2.3;">{chips}</div>',
                    unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — HOME
# =============================================================================
def page_home() -> None:
    # Hero
    st.markdown(f"""
    <div style="text-align:center;padding:40px 0 28px;">
      <div style="display:flex;align-items:center;justify-content:center;
                  gap:24px;flex-wrap:wrap;margin-bottom:16px;">
        <div>{svg_hand_3d()}</div>
        <div>
          <div class="clay-title">Real-Time Sign Language Translator</div>
          <div class="clay-subtitle" style="margin-top:6px;">
            MediaPipe · TensorFlow Lite · Web Speech API · Claymorphism UI
          </div>
        </div>
        <div>{svg_ai_robot()}</div>
      </div>
      <div style="margin-top:16px;">
        <span class="badge badge-green">✅ Cloud Ready</span>&nbsp;
        <span class="badge badge-violet">🎨 Claymorphism</span>&nbsp;
        <span class="badge badge-blue">🌍 13 Languages</span>&nbsp;
        <span class="badge badge-yellow">⚡ Real-Time</span>
      </div>
    </div>
    <hr style="border-color:rgba(99,102,241,0.15);">
    """, unsafe_allow_html=True)

    # Feature cards row
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("""
        <div class="clay-card clay-card-violet" style="text-align:center;min-height:200px;">
          <div style="font-size:2.8rem;margin-bottom:10px;">📷</div>
          <div style="font-size:1.05rem;font-weight:800;color:#a78bfa;margin-bottom:8px;">Browser Camera</div>
          <div style="font-size:0.88rem;color:#64748b;line-height:1.6;">
            Uses WebRTC via <code>streamlit-webrtc</code> — works in any browser.
            No webcam device passthrough required.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="clay-card clay-card-blue" style="text-align:center;min-height:200px;">
          <div style="font-size:2.8rem;margin-bottom:10px;">🧠</div>
          <div style="font-size:1.05rem;font-weight:800;color:#60a5fa;margin-bottom:8px;">Dual TFLite Models</div>
          <div style="font-size:0.88rem;color:#64748b;line-height:1.6;">
            Model 1: 6 words (Hello, No, Please…).
            Model 2: 36 characters (A–Z + 0–9). Both on 42 MediaPipe features.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="clay-card clay-card-cyan" style="text-align:center;min-height:200px;">
          <div style="font-size:2.8rem;margin-bottom:10px;">🔊</div>
          <div style="font-size:1.05rem;font-weight:800;color:#06b6d4;margin-bottom:8px;">Web Speech API TTS</div>
          <div style="font-size:0.88rem;color:#64748b;line-height:1.6;">
            TTS runs in the browser. Speaks completed words or full character strings —
            never individual letters.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Stats row
    c4, c5, c6, c7 = st.columns(4, gap="medium")
    stats = [
        ("6", "Word Gestures", "#818cf8"),
        ("36", "Characters (A-Z+0-9)", "#60a5fa"),
        (">95%", "Accuracy", "#34d399"),
        ("25+ FPS", "Real-Time", "#fbbf24"),
    ]
    for col, (val, label, color) in zip([c4, c5, c6, c7], stats):
        with col:
            st.markdown(f"""
            <div class="clay-card" style="text-align:center;padding:18px 12px;">
              <div style="font-size:2rem;font-weight:800;color:{color};">{val}</div>
              <div style="font-size:0.8rem;color:#64748b;font-weight:600;
                          text-transform:uppercase;letter-spacing:1px;margin-top:4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Supported gestures showcase
    st.markdown('<div class="clay-section-title">🤙 Supported Word Gestures</div>',
                unsafe_allow_html=True)
    emoji_map = {
        "Hello": "👋", "Thank You": "🙏", "Yes": "✅", "No": "❌",
        "Please": "🤲", "Sorry": "😔", "Good": "👍", "Bad": "👎",
        "Help": "🆘", "I Love You": "🤟",
    }
    cols = st.columns(5, gap="small")
    for i, gesture in enumerate(GESTURE_LABELS):
        with cols[i % 5]:
            meaning = GESTURE_MEANINGS.get(gesture, "")
            emoji   = emoji_map.get(gesture, "🤙")
            st.markdown(f"""
            <div class="clay-card" style="text-align:center;padding:16px 10px;margin-bottom:8px;">
              <div style="font-size:1.8rem;">{emoji}</div>
              <div style="font-size:0.9rem;font-weight:700;color:#c7d2fe;margin-top:6px;">{gesture}</div>
              <div style="font-size:0.72rem;color:#475569;margin-top:4px;line-height:1.4;">{meaning}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # How to use
    st.markdown('<div class="clay-section-title">📖 How to Use</div>', unsafe_allow_html=True)
    steps = [
        ("1", "📷", "Go to <b>Live Translator</b> → click <b>START</b> and allow camera access."),
        ("2", "🔄", "Select <b>Word</b> mode (6 gestures) or <b>Character</b> mode (A–Z, 0–9)."),
        ("3", "✋", "Hold a sign steady — wait for the stability bar to fill."),
        ("4", "📝", "Words are added automatically. Use <b>Space / Backspace</b> in character mode."),
        ("5", "🔊", "Click <b>Speak</b> to hear the sentence via your browser's TTS."),
        ("6", "🌍", "Change the target language in <b>Settings</b> to translate."),
    ]
    scols = st.columns(2, gap="large")
    for i, (num, icon, text) in enumerate(steps):
        with scols[i % 2]:
            st.markdown(f"""
            <div class="step-item" style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">
              <span style="font-size:1.3rem;">{icon}</span>
              <span><span class="step-num">Step {num}</span> {text}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("🚀 Open Live Translator", use_container_width=False):
        st.session_state["page"] = "📷 Live Translator"
        st.rerun()


# =============================================================================
# PAGE 2 — LIVE TRANSLATOR
# =============================================================================
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
      style="width:100%;background:linear-gradient(135deg,#4f46e5,#7c3aed);
        color:#fff;border:none;border-radius:16px;padding:11px 0;
        font-size:0.92rem;font-weight:700;cursor:pointer;
        box-shadow:0 5px 14px rgba(99,102,241,0.4),
                   inset 0 1px 0 rgba(255,255,255,0.12);
        transition:all 0.2s;font-family:'Plus Jakarta Sans',sans-serif;
        letter-spacing:0.2px;">
      🔊 Speak
    </button>"""


def _disabled_btn_html(label: str) -> str:
    """Render a disabled HTML button."""
    return f"""
    <button disabled style="width:100%;background:#1a2240;color:#334155;
      border:none;border-radius:16px;padding:11px 0;font-size:0.92rem;
      font-weight:700;cursor:not-allowed;opacity:0.5;
      font-family:'Plus Jakarta Sans',sans-serif;">{label}</button>
    """


def page_live_translator() -> None:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 8px;">
      <div class="clay-title" style="font-size:2rem;">📷 Live Translator</div>
      <div class="clay-subtitle">Real-time gesture recognition with sentence builder</div>
    </div>
    <hr style="border-color:rgba(99,102,241,0.15);">
    """, unsafe_allow_html=True)

    # ── Mode availability warning banners ────────────────────────────────────
    if st.session_state["error_msg"]:
        st.markdown(
            '<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
            'border-radius:14px;padding:10px 16px;margin-bottom:10px;color:#fca5a5;font-size:0.85rem;">'
            '<b>⚠️ Word model unavailable:</b> ' + st.session_state["error_msg"].replace("\n", " ") +
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Recognition Mode Selector ─────────────────────────────────────────────
    st.markdown('<div class="clay-section-title">🔄 Recognition Mode</div>',
                unsafe_allow_html=True)
    mode_col1, mode_col2, mode_col3 = st.columns([2, 2, 4], gap="small")

    with mode_col1:
        if st.button(
            "🤙 Word Recognition",
            use_container_width=True,
            key="btn_mode_word",
            type="primary" if st.session_state["recognition_mode"] == "Word" else "secondary",
        ):
            st.session_state["recognition_mode"] = "Word"
            with _lock:
                _state["mode"] = "Word"
                _state["stable_count"] = 0
                _state["last_stable"]  = None
            smoother.reset()
            st.rerun()

    with mode_col2:
        char_avail = char_interp is not None
        if st.button(
            "🔤 Alphabet / Character",
            use_container_width=True,
            key="btn_mode_char",
            type="primary" if st.session_state["recognition_mode"] == "Character" else "secondary",
            disabled=not char_avail,
        ):
            st.session_state["recognition_mode"] = "Character"
            with _lock:
                _state["mode"] = "Character"
                _state["stable_count"] = 0
                _state["last_stable"]  = None
            char_smoother.reset()
            st.rerun()

    with mode_col3:
        current_mode = st.session_state["recognition_mode"]
        if current_mode == "Word":
            st.markdown(
                '<div style="padding:8px 12px;border-radius:12px;'
                'background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(99,102,241,0.05));'
                'color:#c7d2fe;font-size:0.85rem;font-weight:600;">'
                '🤙 <b>Word mode</b> — recognises: Hello · No · Please · Sorry · Thank You · Yes</div>',
                unsafe_allow_html=True,
            )
        else:
            if char_avail:
                st.markdown(
                    '<div style="padding:8px 12px;border-radius:12px;'
                    'background:linear-gradient(135deg,rgba(251,191,36,0.15),rgba(251,191,36,0.05));'
                    'color:#fde68a;font-size:0.85rem;font-weight:600;">'
                    '🔤 <b>Character mode</b> — recognises: A–Z and 0–9</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="padding:8px 12px;border-radius:12px;'
                    'background:rgba(239,68,68,0.1);color:#fca5a5;font-size:0.85rem;">'
                    '⚠️ Character model not trained yet — run <code>preprocess_characters.py</code> '
                    '→ <code>train_character_model.py</code> → <code>convert_character_tflite.py</code></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    # Sync mode into _state every render so the callback always reads the right value
    with _lock:
        _state["mode"] = st.session_state["recognition_mode"]

    col_video, col_info = st.columns([3, 2], gap="large")

    # ── Left: Camera ──────────────────────────────────────────────────────────
    with col_video:
        st.markdown('<div class="clay-section-title">📹 Camera Feed</div>',
                    unsafe_allow_html=True)

        model_ready = (interp is not None) or (char_interp is not None)
        if model_ready:
            st.markdown('<div class="camera-wrapper">', unsafe_allow_html=True)
            webrtc_streamer(
                key="sign-lang",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480},
                    "audio": False,
                },
                async_processing=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='background:linear-gradient(145deg,#1a2240,#141a34);"
                "border:2px dashed rgba(99,102,241,0.3);border-radius:24px;"
                "height:360px;display:flex;align-items:center;justify-content:center;"
                "flex-direction:column;gap:12px;'>"
                "<span style='font-size:2.5rem;'>📷</span>"
                "<span style='color:#475569;font-weight:600;'>"
                "No model loaded — run training pipeline first</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='color:#334155;font-size:0.8rem;margin-top:10px;"
            "text-align:center;font-weight:500;'>"
            "💡 Good lighting and a plain background improve accuracy</div>",
            unsafe_allow_html=True,
        )

        # Subtitle strip
        if st.session_state["subtitles"]:
            with _lock:
                _mode_now_sub = _state["mode"]
                subtitle_text = (
                    "".join(_state["char_sentence"])
                    if _mode_now_sub == "Character"
                    else " ".join(_state["sentence"])
                )
            if subtitle_text:
                st.markdown(
                    f'<div class="subtitle-strip">📝 {subtitle_text}</div>',
                    unsafe_allow_html=True,
                )

    # ── Right: Prediction + Sentence ──────────────────────────────────────────
    with col_info:
        _mode_now = st.session_state["recognition_mode"]

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
            g_grad   = "linear-gradient(135deg,#34d399,#10b981)"
            fill_cls = "conf-fill-high"
        elif _c >= 0.5:
            g_grad   = "linear-gradient(135deg,#fbbf24,#f59e0b)"
            fill_cls = "conf-fill-medium"
        else:
            g_grad   = "linear-gradient(135deg,#818cf8,#6366f1)"
            fill_cls = "conf-fill-low"

        meaning = GESTURE_MEANINGS.get(_g, "") if _mode_now == "Word" else ""

        mode_badge = (
            '<span style="font-size:0.7rem;color:#fbbf24;font-weight:700;'
            'text-transform:uppercase;letter-spacing:1px;">🔤 Character Mode</span>'
            if _mode_now == "Character"
            else
            '<span style="font-size:0.7rem;color:#818cf8;font-weight:700;'
            'text-transform:uppercase;letter-spacing:1px;">🤙 Word Mode</span>'
        )

        # Prediction card
        st.markdown(f"""
        <div class="clay-card clay-card-violet" style="text-align:center;margin-bottom:14px;">
          <div style="margin-bottom:6px;">{mode_badge}</div>
          <div style="font-size:0.7rem;color:#64748b;font-weight:700;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
            🎯 Current Prediction
          </div>
          <div style="font-size:3rem;font-weight:800;
                      background:{g_grad};
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      filter:drop-shadow(0 0 14px rgba(139,92,246,0.4));
                      line-height:1.1;margin-bottom:4px;">{_g}</div>
          <div style="font-size:0.8rem;color:#475569;font-weight:500;margin-bottom:10px;
                      font-style:italic;">{meaning}</div>
          <div style="color:#64748b;font-size:0.85rem;font-weight:600;margin-bottom:14px;">
            Confidence: <span style="color:#c7d2fe;font-weight:800;">{conf_pct}%</span>
          </div>
          <div style="margin-bottom:10px;">
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
              <span style="font-size:0.75rem;color:#94a3b8;">{_sf}/{_stab_cfg}</span>
            </div>
            <div class="conf-track">
              <div style="width:{stable_pct}%;height:100%;
                          background:linear-gradient(90deg,#818cf8,#a78bfa);
                          border-radius:8px;transition:width 0.3s;"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(99,102,241,0.12);">', unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        # WORD MODE — Sentence builder
        # ═════════════════════════════════════════════════════════════════════
        if _mode_now == "Word":
            _words = list(_lock.__class__ and _state["sentence"]) if False else []
            with _lock:
                _words = list(_state["sentence"])

            sentence_str = (
                " ".join(_words) if _words
                else "<span style='color:#334155;font-style:italic;'>Waiting for gestures…</span>"
            )
            st.markdown(f"""
            <div style="margin-bottom:6px;">
              <div class="clay-section-title">📝 Sentence Builder</div>
            </div>
            <div class="sentence-box">{sentence_str}</div>
            """, unsafe_allow_html=True)

            # Translation
            if _words and st.session_state["language"] != DEFAULT_LANGUAGE:
                tl = translate_sentence(_words, st.session_state["language"])
                st.markdown(f"""
                <div class="trans-card">
                  <div class="trans-lang">{st.session_state["language"]}</div>
                  <div class="trans-text">{tl}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # Word controls: Speak | Undo | Clear
            speak_text = " ".join(_words) if _words else ""
            btn_speak, btn_undo, btn_clear = st.columns(3)

            with btn_speak:
                if speak_text:
                    st.markdown(_tts_js(speak_text), unsafe_allow_html=True)
                else:
                    st.markdown(_disabled_btn_html("🔊 Speak"), unsafe_allow_html=True)

            with btn_undo:
                if st.button("⬅️ Undo", use_container_width=True,
                             disabled=not _words, key="word_btn_undo"):
                    with _lock:
                        if _state["sentence"]:
                            _state["sentence"].pop()
                    st.rerun()

            with btn_clear:
                if st.button("🗑️ Clear", use_container_width=True,
                             disabled=not _words, key="word_btn_clear"):
                    with _lock:
                        _state["sentence"]     = []
                        _state["last_stable"]  = None
                        _state["stable_count"] = 0
                    smoother.reset()
                    st.session_state["_last_auto_spoken"] = ""
                    st.rerun()

            # Auto-speak: speak each new word (not individual characters)
            # Only fires in word mode.
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

            st.markdown('<hr style="border-color:rgba(99,102,241,0.12);">', unsafe_allow_html=True)

            # Word history chips
            st.markdown('<div class="clay-section-title">🕐 Word History (this session)</div>',
                        unsafe_allow_html=True)
            if _words:
                chips = "".join(
                    f'<span class="gesture-chip">{w}</span>'
                    for w in reversed(_words[-20:])
                )
                st.markdown(f'<div style="line-height:2.3;">{chips}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#334155;font-size:0.88rem;font-style:italic;">'
                    'No words recognised yet — start signing!</div>',
                    unsafe_allow_html=True,
                )

        # ═════════════════════════════════════════════════════════════════════
        # CHARACTER MODE — Sentence builder
        # H E L L O → HELLO
        # ═════════════════════════════════════════════════════════════════════
        else:
            with _lock:
                _chars = list(_state["char_sentence"])

            # Display: join chars directly (spaces stored as " " in list)
            char_display = "".join(_chars) if _chars else ""
            char_box_content = (
                f'<span style="letter-spacing:0.12em;font-size:1.1rem;">{char_display}</span>'
                if char_display
                else "<span style='color:#334155;font-style:italic;'>Waiting for characters…</span>"
            )
            st.markdown(f"""
            <div style="margin-bottom:6px;">
              <div class="clay-section-title">🔤 Character Builder</div>
            </div>
            <div class="sentence-box" style="min-height:52px;font-family:monospace;">
              {char_box_content}
            </div>
            """, unsafe_allow_html=True)

            # Translation of the joined character string
            if char_display.strip() and st.session_state["language"] != DEFAULT_LANGUAGE:
                tl = translate(char_display.strip(), st.session_state["language"], "en")
                st.markdown(f"""
                <div class="trans-card">
                  <div class="trans-lang">{st.session_state["language"]}</div>
                  <div class="trans-text">{tl}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # ── Row 1: Speak | Space | Backspace ─────────────────────────────
            speak_text_char = char_display.strip()
            c_speak, c_space, c_back = st.columns(3)

            with c_speak:
                if speak_text_char:
                    st.markdown(_tts_js(speak_text_char), unsafe_allow_html=True)
                else:
                    st.markdown(_disabled_btn_html("🔊 Speak"), unsafe_allow_html=True)

            with c_space:
                # Add a space character between words (stored as " " in list)
                if st.button("⎵ Space", use_container_width=True,
                             key="char_btn_space"):
                    with _lock:
                        # Only add space if last char is not already a space
                        if _state["char_sentence"] and _state["char_sentence"][-1] != " ":
                            _state["char_sentence"].append(" ")
                            _state["char_last_accepted"] = " "
                    st.rerun()

            with c_back:
                if st.button("⌫ Backspace", use_container_width=True,
                             disabled=not _chars,
                             key="char_btn_back"):
                    with _lock:
                        if _state["char_sentence"]:
                            _state["char_sentence"].pop()
                            # Reset last_accepted so same char can be re-signed
                            _state["char_last_accepted"] = (
                                _state["char_sentence"][-1]
                                if _state["char_sentence"] else None
                            )
                    st.rerun()

            # ── Row 2: Clear ──────────────────────────────────────────────────
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear All", use_container_width=True,
                         disabled=not _chars,
                         key="char_btn_clear"):
                with _lock:
                    _state["char_sentence"]       = []
                    _state["char_last_accepted"]  = None
                    _state["char_debounce_count"] = 0
                    _state["stable_count"]        = 0
                    _state["last_stable"]         = None
                char_smoother.reset()
                st.rerun()

            # Character mode: auto-speak is intentionally disabled per-character.
            # TTS speaks the full completed string only when the user clicks Speak.
            # This avoids the annoyance of the engine saying "H... E... L... L... O".

            st.markdown('<hr style="border-color:rgba(99,102,241,0.12);">', unsafe_allow_html=True)

            # Character history chips (last 30 chars)
            st.markdown('<div class="clay-section-title">🕐 Recent Characters</div>',
                        unsafe_allow_html=True)
            if _chars:
                chips = "".join(
                    f'<span class="gesture-chip">{c if c != " " else "⎵"}</span>'
                    for c in _chars[-30:]
                )
                st.markdown(f'<div style="line-height:2.3;">{chips}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#334155;font-size:0.88rem;font-style:italic;">'
                    'No characters yet — start signing!</div>',
                    unsafe_allow_html=True,
                )


# =============================================================================
# PAGE 3 — HISTORY
# =============================================================================
def page_history() -> None:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 8px;">
      <div class="clay-title" style="font-size:2rem;">📜 Gesture History</div>
      <div class="clay-subtitle">All recognised gestures from this session</div>
    </div>
    <hr style="border-color:rgba(99,102,241,0.15);">
    """, unsafe_allow_html=True)

    gh: GestureHistory = st.session_state["history"]

    col_dl, col_clr, col_info = st.columns([2, 2, 4], gap="medium")

    with col_dl:
        if not gh.is_empty():
            csv_bytes = gh.to_csv_bytes()
            st.download_button(
                label="⬇️ Export CSV",
                data=csv_bytes,
                file_name="gesture_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Export CSV", disabled=True, use_container_width=True)

    with col_clr:
        if st.button("🗑️ Clear History", use_container_width=True,
                     disabled=gh.is_empty()):
            gh.clear()
            st.success("History cleared.")
            st.rerun()

    with col_info:
        total = gh.count()
        avg_c = gh.average_confidence()
        top_g = gh.most_frequent_gesture() or "—"
        st.markdown(f"""
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
          <span class="badge badge-violet">📝 {total} entries</span>
          <span class="badge badge-blue">⭐ Avg conf: {avg_c:.0%}</span>
          <span class="badge badge-green">🏆 Top: {top_g}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if gh.is_empty():
        st.markdown("""
        <div class="clay-card" style="text-align:center;padding:48px 24px;">
          <div style="font-size:3rem;margin-bottom:14px;">📭</div>
          <div style="font-size:1.1rem;font-weight:700;color:#475569;">No history yet</div>
          <div style="font-size:0.88rem;color:#334155;margin-top:8px;">
            Head over to <b>Live Translator</b> and start signing.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    rows = gh.to_display_dicts()
    st.markdown('<div class="clay-section-title">📋 Gesture Log (newest first)</div>',
                unsafe_allow_html=True)

    header = "<tr>" + "".join(
        f"<th style='padding:10px 14px;font-size:0.78rem;color:#64748b;"
        f"text-transform:uppercase;letter-spacing:0.8px;font-weight:700;"
        f"border-bottom:1px solid rgba(99,102,241,0.15);text-align:left;'>{col}</th>"
        for col in ["Time", "Gesture", "Translation", "Language", "Confidence"]
    ) + "</tr>"

    body_rows = []
    for row in rows[:200]:
        conf_val = row["Confidence"].replace("%", "")
        try:
            cv = float(conf_val)
            conf_color = "#34d399" if cv >= 75 else ("#fbbf24" if cv >= 50 else "#f87171")
        except ValueError:
            conf_color = "#94a3b8"

        body_rows.append(
            "<tr style='border-bottom:1px solid rgba(99,102,241,0.08);'>"
            f"<td style='padding:10px 14px;font-size:0.82rem;color:#475569;'>{row['Time']}</td>"
            f"<td style='padding:10px 14px;font-size:0.88rem;font-weight:700;color:#c7d2fe;'>{row['Gesture']}</td>"
            f"<td style='padding:10px 14px;font-size:0.88rem;color:#e2e8f0;'>{row['Translation']}</td>"
            f"<td style='padding:10px 14px;'><span class='badge badge-blue'>{row['Language']}</span></td>"
            f"<td style='padding:10px 14px;font-size:0.88rem;font-weight:700;color:{conf_color};'>{row['Confidence']}</td>"
            "</tr>"
        )

    table_html = f"""
    <div class="clay-card" style="padding:0;overflow:hidden;">
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;">
          <thead style="background:linear-gradient(145deg,#1a2240,#141a34);">
            {header}
          </thead>
          <tbody>{"".join(body_rows)}</tbody>
        </table>
      </div>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    if len(rows) > 200:
        st.markdown(
            f'<div style="text-align:center;color:#475569;font-size:0.82rem;margin-top:8px;">'
            f'Showing latest 200 of {len(rows)} entries. Export CSV to see all.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="clay-section-title">📊 Gesture Frequency</div>',
                unsafe_allow_html=True)
    freq = gh.gesture_frequency()
    if freq:
        max_count = max(freq.values())
        freq_html = ""
        for gesture, count in freq.items():
            pct = int(count / max_count * 100)
            freq_html += f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:0.88rem;color:#c7d2fe;font-weight:600;">{gesture}</span>
                <span style="font-size:0.82rem;color:#818cf8;font-weight:700;">{count}×</span>
              </div>
              <div class="conf-track">
                <div class="conf-fill-low" style="width:{pct}%;
                  background:linear-gradient(90deg,#818cf8,#a78bfa);transition:width 0.4s;"></div>
              </div>
            </div>"""
        st.markdown(f'<div class="clay-card">{freq_html}</div>',
                    unsafe_allow_html=True)


# =============================================================================
# PAGE 4 — ANALYTICS
# =============================================================================
def page_analytics() -> None:
    st.markdown(f"""
    <div style="text-align:center;padding:18px 0 8px;">
      <div style="display:inline-block;margin-bottom:8px;">{svg_analytics()}</div>
      <div class="clay-title" style="font-size:2rem;">📊 Analytics</div>
      <div class="clay-subtitle">Session performance metrics</div>
    </div>
    <hr style="border-color:rgba(99,102,241,0.15);">
    """, unsafe_allow_html=True)

    analytics: SessionAnalytics = st.session_state["analytics"]
    snap = analytics.snapshot()
    gh: GestureHistory = st.session_state["history"]

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    kpis = [
        ("Total Gestures", str(snap.total_gestures), "#818cf8"),
        ("Avg Confidence", f"{snap.average_confidence:.0%}", "#34d399"),
        ("Accuracy", f"{snap.recognition_accuracy:.0%}", "#60a5fa"),
        ("Session Time", analytics.format_duration(), "#fbbf24"),
        ("Live FPS", f"{snap.current_fps:.0f}", "#a78bfa"),
    ]
    for col, (label, val, color) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(f"""
            <div class="clay-card" style="text-align:center;padding:18px 10px;">
              <div style="font-size:1.6rem;font-weight:800;color:{color};">{val}</div>
              <div style="font-size:0.72rem;color:#64748b;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.8px;margin-top:4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    row1_left, row1_right = st.columns(2, gap="large")

    with row1_left:
        st.markdown('<div class="clay-section-title">🎯 Confidence Breakdown</div>',
                    unsafe_allow_html=True)
        hi  = snap.high_confidence_count
        lo  = snap.low_confidence_count
        tot = snap.total_gestures or 1
        hi_pct = int(hi / tot * 100)
        lo_pct = int(lo / tot * 100)
        st.markdown(f"""
        <div class="clay-card">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;">
            <div style="text-align:center;padding:14px;
                        background:linear-gradient(145deg,#064e3b,#065f46);
                        border-radius:16px;border:1px solid rgba(52,211,153,0.2);">
              <div style="font-size:1.8rem;font-weight:800;color:#34d399;">{hi}</div>
              <div style="font-size:0.75rem;color:#6ee7b7;font-weight:600;">High Confidence</div>
              <div style="font-size:0.72rem;color:#34d399;margin-top:2px;">≥ threshold</div>
            </div>
            <div style="text-align:center;padding:14px;
                        background:linear-gradient(145deg,#450a0a,#3b0606);
                        border-radius:16px;border:1px solid rgba(239,68,68,0.2);">
              <div style="font-size:1.8rem;font-weight:800;color:#f87171;">{lo}</div>
              <div style="font-size:0.75rem;color:#fca5a5;font-weight:600;">Low Confidence</div>
              <div style="font-size:0.72rem;color:#f87171;margin-top:2px;">&lt; threshold</div>
            </div>
          </div>
          <div style="margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
              <span style="font-size:0.78rem;color:#64748b;font-weight:600;">High Conf Rate</span>
              <span style="font-size:0.78rem;color:#34d399;font-weight:700;">{hi_pct}%</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill-high" style="width:{hi_pct}%;transition:width 0.5s;"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with row1_right:
        st.markdown('<div class="clay-section-title">🤙 Gesture Frequency</div>',
                    unsafe_allow_html=True)
        freq = gh.gesture_frequency()
        if freq:
            max_count = max(freq.values()) or 1
            bars_html = ""
            color_cycle = ["#818cf8","#a78bfa","#60a5fa","#34d399",
                           "#fbbf24","#f87171","#06b6d4","#ec4899",
                           "#84cc16","#f97316"]
            for i, (g, cnt) in enumerate(list(freq.items())[:8]):
                pct = int(cnt / max_count * 100)
                col = color_cycle[i % len(color_cycle)]
                bars_html += f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:0.85rem;color:#c7d2fe;font-weight:600;">{g}</span>
                    <span style="font-size:0.82rem;color:{col};font-weight:700;">{cnt}×</span>
                  </div>
                  <div class="conf-track">
                    <div style="width:{pct}%;height:100%;border-radius:8px;
                                background:{col};transition:width 0.4s;"></div>
                  </div>
                </div>"""
            st.markdown(f'<div class="clay-card">{bars_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="clay-card" style="text-align:center;padding:40px;">
              <div style="color:#334155;font-style:italic;">No gestures recorded yet.</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="clay-section-title">⚡ FPS History</div>',
                unsafe_allow_html=True)
    fps_hist = snap.fps_history
    if fps_hist:
        import pandas as pd
        df_fps = pd.DataFrame({"FPS": fps_hist})
        st.line_chart(df_fps, use_container_width=True, height=180)
    else:
        st.markdown("""
        <div class="clay-card" style="text-align:center;padding:30px;">
          <div style="color:#334155;font-style:italic;">Start the camera to collect FPS data.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset Analytics", key="reset_analytics"):
        analytics.reset()
        st.success("Analytics reset.")
        st.rerun()


# =============================================================================
# PAGE 5 — SETTINGS
# =============================================================================
def page_settings() -> None:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 8px;">
      <div class="clay-title" style="font-size:2rem;">⚙️ Settings</div>
      <div class="clay-subtitle">Customise appearance, language, and inference</div>
    </div>
    <hr style="border-color:rgba(99,102,241,0.15);">
    """, unsafe_allow_html=True)

    changed = False
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        # Language
        st.markdown('<div class="clay-section-title">🌍 Translation Language</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="clay-card" style="padding:20px 22px;">', unsafe_allow_html=True)
        lang_options = list(SUPPORTED_LANGUAGES.keys())
        lang_idx = lang_options.index(st.session_state["language"]) \
                   if st.session_state["language"] in lang_options else 0
        new_lang = st.selectbox(
            "Target Language", options=lang_options, index=lang_idx,
            label_visibility="collapsed", key="sel_language",
        )
        if new_lang != st.session_state["language"]:
            st.session_state["language"] = new_lang
            changed = True
        st.markdown("</div>", unsafe_allow_html=True)

        # Appearance
        st.markdown('<div class="clay-section-title" style="margin-top:16px;">🎨 Appearance</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="clay-card" style="padding:20px 22px;">', unsafe_allow_html=True)
        new_font = st.radio(
            "Font Size", options=["normal", "large"],
            index=0 if st.session_state["font_size"] == "normal" else 1,
            horizontal=True, key="radio_font",
        )
        if new_font != st.session_state["font_size"]:
            st.session_state["font_size"] = new_font
            changed = True
        new_hc = st.toggle("High Contrast Mode",
                           value=st.session_state["high_contrast"], key="tog_hc")
        if new_hc != st.session_state["high_contrast"]:
            st.session_state["high_contrast"] = new_hc
            changed = True
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        # TTS & Subtitles
        st.markdown('<div class="clay-section-title">🔊 TTS & Subtitles</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="clay-card" style="padding:20px 22px;">', unsafe_allow_html=True)
        new_auto = st.toggle(
            "Auto-speak each word when recognised (Word mode only)",
            value=st.session_state["auto_speak"], key="tog_auto_speak",
        )
        if new_auto != st.session_state["auto_speak"]:
            st.session_state["auto_speak"] = new_auto
            changed = True
        new_sub = st.toggle(
            "Show live subtitle strip below camera",
            value=st.session_state["subtitles"], key="tog_subtitles",
        )
        if new_sub != st.session_state["subtitles"]:
            st.session_state["subtitles"] = new_sub
            changed = True
        st.markdown(
            '<div style="font-size:0.78rem;color:#475569;margin-top:8px;">'
            'Character mode never auto-speaks individual letters — use the Speak button.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Inference — Word model
        st.markdown('<div class="clay-section-title" style="margin-top:16px;">🧠 Word Model Inference</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="clay-card" style="padding:20px 22px;">', unsafe_allow_html=True)
        new_conf = st.slider(
            "Confidence Threshold", min_value=0.40, max_value=0.99,
            value=float(st.session_state["conf_threshold"]),
            step=0.05, format="%.2f", key="sl_conf",
        )
        if abs(new_conf - st.session_state["conf_threshold"]) > 0.001:
            st.session_state["conf_threshold"] = new_conf
            changed = True
        new_stab = st.slider(
            "Stability Frames (before word is added)",
            min_value=5, max_value=40,
            value=int(st.session_state["stable_frames_cfg"]),
            step=1, key="sl_stab",
        )
        if new_stab != st.session_state["stable_frames_cfg"]:
            st.session_state["stable_frames_cfg"] = new_stab
            changed = True
        st.markdown("</div>", unsafe_allow_html=True)

        # Inference — Character model
        st.markdown('<div class="clay-section-title" style="margin-top:16px;">🔤 Character Model Inference</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="clay-card" style="padding:20px 22px;">', unsafe_allow_html=True)
        new_char_conf = st.slider(
            "Character Confidence Threshold", min_value=0.40, max_value=0.99,
            value=float(st.session_state["char_conf_threshold"]),
            step=0.05, format="%.2f", key="sl_char_conf",
        )
        if abs(new_char_conf - st.session_state["char_conf_threshold"]) > 0.001:
            st.session_state["char_conf_threshold"] = new_char_conf
            changed = True
        new_char_stab = st.slider(
            "Character Stability Frames",
            min_value=5, max_value=50,
            value=int(st.session_state["char_stable_frames_cfg"]),
            step=1, key="sl_char_stab",
        )
        if new_char_stab != st.session_state["char_stable_frames_cfg"]:
            st.session_state["char_stable_frames_cfg"] = new_char_stab
            changed = True
        st.markdown(
            '<div style="font-size:0.78rem;color:#475569;margin-top:8px;">'
            'Higher stability = fewer false positives but slightly slower response.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    s1, s2 = st.columns([2, 2])
    with s1:
        if changed:
            st.success("✅ Settings saved automatically.")
            inject_css(
                theme=st.session_state["theme"],
                large_font=(st.session_state["font_size"] == "large"),
                high_contrast=st.session_state["high_contrast"],
            )
    with s2:
        if st.button("↩️ Reset to Defaults", use_container_width=True, key="reset_settings"):
            st.session_state["language"]              = DEFAULT_LANGUAGE
            st.session_state["font_size"]             = DEFAULT_FONT_SIZE
            st.session_state["high_contrast"]         = DEFAULT_HIGH_CONTRAST
            st.session_state["auto_speak"]            = AUTO_SPEAK_DEFAULT
            st.session_state["subtitles"]             = SUBTITLE_ENABLED_DEFAULT
            st.session_state["conf_threshold"]        = CONFIDENCE_THRESHOLD
            st.session_state["stable_frames_cfg"]     = STABLE_FRAME_COUNT
            st.session_state["char_conf_threshold"]   = CHARACTER_CONFIDENCE_THRESHOLD
            st.session_state["char_stable_frames_cfg"]= CHARACTER_STABLE_FRAME_COUNT
            st.success("Settings reset to defaults.")
            st.rerun()

    # Summary card
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="clay-section-title">📋 Current Settings Summary</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="clay-card" style="padding:18px 22px;">
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
        <div class="stat-card"><div class="stat-label">Language</div>
          <div class="stat-value" style="font-size:0.95rem;">{st.session_state["language"]}</div></div>
        <div class="stat-card"><div class="stat-label">Word Conf.</div>
          <div class="stat-value">{st.session_state["conf_threshold"]:.0%}</div></div>
        <div class="stat-card"><div class="stat-label">Word Stability</div>
          <div class="stat-value">{st.session_state["stable_frames_cfg"]}</div></div>
        <div class="stat-card"><div class="stat-label">Char Conf.</div>
          <div class="stat-value">{st.session_state["char_conf_threshold"]:.0%}</div></div>
        <div class="stat-card"><div class="stat-label">Char Stability</div>
          <div class="stat-value">{st.session_state["char_stable_frames_cfg"]}</div></div>
        <div class="stat-card"><div class="stat-label">Auto Speak</div>
          <div class="stat-value">{"On" if st.session_state["auto_speak"] else "Off"}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 6 — ABOUT
# =============================================================================
def page_about() -> None:
    st.markdown(f"""
    <div style="text-align:center;padding:28px 0 16px;">
      <div style="display:flex;align-items:center;justify-content:center;
                  gap:20px;flex-wrap:wrap;margin-bottom:16px;">
        <div>{svg_hand_3d()}</div>
        <div>
          <div class="clay-title">About This Project</div>
          <div class="clay-subtitle" style="margin-top:6px;">Final-year Computer Vision project</div>
        </div>
        <div>{svg_ai_robot()}</div>
      </div>
    </div>
    <hr style="border-color:rgba(99,102,241,0.15);">
    """, unsafe_allow_html=True)

    st.markdown('<div class="clay-section-title">📖 Project Overview</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="clay-card" style="line-height:1.8;">
      <p style="color:#cbd5e1;font-size:0.95rem;">
        The <b style="color:#c7d2fe;">Real-Time Sign Language Translator</b> captures live webcam footage,
        detects hand landmarks using <b>Google MediaPipe</b>, classifies the gesture with a
        <b>TensorFlow Lite</b> dense neural network, and reads the result aloud using the
        <b>Web Speech API</b> — all in the browser.
      </p>
      <p style="color:#94a3b8;font-size:0.88rem;margin-top:12px;">
        Two independent models share the same 42-feature MediaPipe pipeline:<br>
        <b>Model 1 (Word)</b> — 6 ISLRTC word gestures. Uses <code>models/gesture_model.tflite</code>
        + <code>models/scaler.pkl</code> + <code>models/label_map.json</code>.<br>
        <b>Model 2 (Character)</b> — 36 ISLRTC alphabet/digit signs (A–Z, 0–9).
        Uses <code>models/character_model.tflite</code> + <code>models/character_scaler.pkl</code>
        + <code>models/character_label_map.json</code>.<br>
        Scalers and label maps are <b>never</b> mixed between models.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="clay-section-title">🏗️ Architecture Pipeline</div>',
                unsafe_allow_html=True)
    arch_steps = [
        ("📷", "Webcam Frame",      "WebRTC via streamlit-webrtc"),
        ("🔍", "MediaPipe Hands",   "21 landmarks (x, y) → 42 features"),
        ("📐", "Normalisation",     "Wrist-origin, scale to [−1, 1]"),
        ("⚖️", "StandardScaler",   "Word scaler OR character scaler"),
        ("🧠", "TFLite Dense NN",   "42 → 256 → 128 → 64 → N classes"),
        ("🗳️", "PredictionSmoother","Rolling majority vote over frames"),
        ("📝", "Sentence Builder",  "Word list OR character list + debounce"),
        ("🔊", "Web Speech API",    "Browser TTS — speaks full string"),
    ]
    arch_cols = st.columns(4, gap="small")
    for i, (icon, title, desc) in enumerate(arch_steps):
        with arch_cols[i % 4]:
            st.markdown(f"""
            <div class="clay-card" style="text-align:center;padding:16px 10px;margin-bottom:8px;">
              <div style="font-size:1.8rem;margin-bottom:6px;">{icon}</div>
              <div style="font-size:0.88rem;font-weight:700;color:#c7d2fe;margin-bottom:4px;">{title}</div>
              <div style="font-size:0.75rem;color:#475569;line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="clay-section-title">🛠️ Tech Stack</div>', unsafe_allow_html=True)
    stack = [
        ("Python 3.10+",         "Language",         "#818cf8"),
        ("OpenCV 4.9",           "Computer Vision",  "#60a5fa"),
        ("MediaPipe 0.10",       "Hand Detection",   "#34d399"),
        ("TFLite / Keras 2.16",  "Deep Learning",    "#fbbf24"),
        ("Streamlit 1.35",       "Web UI",           "#f87171"),
        ("streamlit-webrtc",     "Browser Camera",   "#06b6d4"),
        ("Web Speech API",       "TTS",              "#a78bfa"),
        ("NumPy / Pandas",       "Data Processing",  "#ec4899"),
        ("deep-translator",      "Translation",      "#84cc16"),
        ("Docker / GHCR",        "Deployment",       "#f97316"),
    ]
    stack_cols = st.columns(5, gap="small")
    for i, (name, category, color) in enumerate(stack):
        with stack_cols[i % 5]:
            st.markdown(f"""
            <div class="clay-card" style="text-align:center;padding:14px 10px;margin-bottom:8px;">
              <div style="font-size:0.95rem;font-weight:700;color:{color};">{name}</div>
              <div style="font-size:0.72rem;color:#475569;margin-top:4px;">{category}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="clay-section-title">📈 Performance Metrics</div>',
                unsafe_allow_html=True)
    metrics = [
        ("Inference Latency",   "~8 ms",    "#34d399"),
        ("Landmark Extraction", "~15 ms",   "#60a5fa"),
        ("End-to-End FPS",      "25–35",    "#fbbf24"),
        ("Word Model (TFLite)", "~0.9 MB",  "#818cf8"),
        ("Char Model (TFLite)", "~0.9 MB",  "#a78bfa"),
        ("Test Accuracy",       ">95%",     "#f87171"),
    ]
    m_cols = st.columns(3, gap="medium")
    for i, (label, val, color) in enumerate(metrics):
        with m_cols[i % 3]:
            st.markdown(f"""
            <div class="clay-card" style="display:flex;align-items:center;
                        gap:14px;padding:14px 18px;margin-bottom:8px;">
              <div style="font-size:1.5rem;font-weight:800;color:{color};
                          min-width:70px;text-align:center;">{val}</div>
              <div style="font-size:0.85rem;color:#94a3b8;font-weight:600;">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MAIN ROUTER
# =============================================================================
render_sidebar()

page = st.session_state["page"]

if page == "🏠 Home":
    page_home()
elif page == "📷 Live Translator":
    page_live_translator()
elif page == "📜 History":
    page_history()
elif page == "📊 Analytics":
    page_analytics()
elif page == "⚙️ Settings":
    page_settings()
elif page == "ℹ️ About":
    page_about()
else:
    page_home()
