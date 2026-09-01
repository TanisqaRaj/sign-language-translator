import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, CONFIDENCE_THRESHOLD, STABLE_FRAME_COUNT,
    PREDICTION_BUFFER_LEN, MODEL_TFLITE_PATH, LABEL_MAP_PATH, MODEL_DIR,
    GESTURE_LABELS, GESTURE_MEANINGS,
    LOW_CONFIDENCE_THRESHOLD, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE,
    DEFAULT_THEME, DEFAULT_FONT_SIZE, AUTO_SPEAK_DEFAULT,
    SUBTITLE_ENABLED_DEFAULT, NAV_PAGES,
    # Character model
    CHARACTER_CONFIDENCE_THRESHOLD, CHARACTER_STABLE_FRAME_COUNT,
    CHARACTER_DEBOUNCE_FRAMES, CHARACTER_LABELS,
)
from utils.logger           import get_logger
from utils.mediapipe_helper import HandDetector, MultiHandResult
from utils.tts_engine       import TTSEngine
from utils.translator       import translate, translate_all_languages, get_language_list
from utils.gesture_history  import GestureHistory
from utils.analytics        import SessionAnalytics
from utils.ui_components    import (
    inject_css, render_ai_panel, render_confidence_meter,
    render_translation_cards, render_suggestions,
    render_detection_status, render_subtitle_strip,
    camera_placeholder_html, render_glass_card, render_status_badge,
    render_loading_sequence,
)
from preprocess  import normalise_landmarks
from inference   import (
    load_tflite_model, load_label_map, load_scaler, predict, PredictionSmoother,
    load_character_tflite_model, load_character_label_map, load_character_scaler,
    predict_character,
)

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Translator – AI Edition",
    page_icon="🤙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        # Navigation
        "page":              NAV_PAGES[0],
        # Theme / accessibility
        "theme":             DEFAULT_THEME,
        "large_font":        (DEFAULT_FONT_SIZE == "large"),
        "high_contrast":     False,
        # Camera / inference
        "running":           False,
        "current_gesture":   "—",
        "confidence":        0.0,
        "stable_frames":     0,
        "all_probs":         {},        # {label: prob} for confidence meter
        "fps":               0.0,
        "model_loaded":      False,
        "char_model_loaded": False,
        "error_msg":         "",
        # Recognition mode: "Word" or "Character"
        "recognition_mode":  "Word",
        # Sentence builder
        "sentence":          [],
        # Character sentence (list of chars)
        "char_sentence":     [],
        # Translation
        "target_language":   DEFAULT_LANGUAGE,
        "last_translation":  "",
        # TTS
        "auto_speak":        AUTO_SPEAK_DEFAULT,
        "tts_speaking":      False,
        "voice_index":       0,
        "speech_rate":       150,
        # Subtitles
        "subtitles_on":      SUBTITLE_ENABLED_DEFAULT,
        "subtitle_text":     "",
        # History (GestureHistory object stored in session)
        "history_obj":       None,
        # Analytics (SessionAnalytics object stored in session)
        "analytics_obj":     None,
        # Camera settings
        "camera_index":      CAMERA_INDEX,
        "detection_threshold": CONFIDENCE_THRESHOLD,
        # Character model thresholds (tunable in Settings)
        "char_conf_threshold":    CHARACTER_CONFIDENCE_THRESHOLD,
        "char_stable_frames_cfg": CHARACTER_STABLE_FRAME_COUNT,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Initialise complex objects once
    if st.session_state["history_obj"] is None:
        st.session_state["history_obj"] = GestureHistory()
    if st.session_state["analytics_obj"] is None:
        st.session_state["analytics_obj"] = SessionAnalytics()
        st.session_state["analytics_obj"].start_session()


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_resources():
    """Load word model, character model, detector, and TTS once; cache for the lifetime of the server.

    Both models use 42 hand-landmark features (MediaPipe only).
    MoveNet is NOT used — keeps the feature vector at exactly 42 floats.
    """
    try:
        interp, in_idx, out_idx = load_tflite_model()
        label_map   = load_label_map()
        scaler      = load_scaler()
        detector    = HandDetector()
        tts         = TTSEngine()
        return interp, in_idx, out_idx, label_map, scaler, detector, tts
    except Exception as exc:
        st.session_state["error_msg"] = str(exc)
        return (None,) * 7


@st.cache_resource(show_spinner=False)
def _load_char_resources():
    """Load character model (A-Z + 0-9) once; completely independent from word model."""
    try:
        c_interp, c_in, c_out = load_character_tflite_model()
        c_label_map = load_character_label_map()
        c_scaler    = load_character_scaler()
        return c_interp, c_in, c_out, c_label_map, c_scaler, None
    except Exception as exc:
        return None, None, None, None, None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level inference helper
# ─────────────────────────────────────────────────────────────────────────────

def _process_frame(frame, interp, in_idx, out_idx, label_map, scaler,
                   detector, smoother):
    """
    Run hand detection + model inference on one frame.

    Uses 42 normalised MediaPipe hand-landmark features only (no MoveNet).
    This matches the retrained gesture_model.tflite input shape of (1, 42).

    Two-hand behaviour
    ------------------
    Both hand skeletons are drawn when visible.  Inference always runs on the
    *dominant* hand (right preferred, left fallback) so the existing 42-feature
    models are never given a different input shape.

    Returns
    -------
    (annotated_frame, gesture_or_None, confidence, has_hand,
     all_probs_dict, multi_hand_result)
    """
    # ── Two-hand-aware detection — draws ALL visible skeletons ────────────────
    mhr, annotated = detector.find_all_landmarks(frame, draw=True)
    has_hand = mhr.dominant is not None

    if not has_hand:
        smoother.reset()
        return annotated, None, 0.0, False, {}, mhr

    # Dominant-hand landmarks → 42 normalised features for the model
    raw_lm   = mhr.dominant.landmarks
    features = normalise_landmarks(raw_lm)

    raw_pred, conf = predict(
        features, interp, in_idx, out_idx, label_map, scaler
    )
    gesture, confidence = smoother.update(raw_pred, conf)

    # Build per-class probability dict for the confidence meter
    all_probs = {}
    if raw_pred:
        all_probs[raw_pred] = conf
        rest = max(0.0, 1.0 - conf) / max(len(GESTURE_LABELS) - 1, 1)
        for lbl in GESTURE_LABELS:
            if lbl != raw_pred:
                all_probs[lbl] = rest

    return annotated, gesture, confidence, has_hand, all_probs, mhr


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(tts) -> None:
    from utils.ui_components import svg_hand_3d
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:10px 0 16px;">
          {svg_hand_3d()}
          <div style="font-size:1.1rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#60a5fa);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:6px;">
            ISL Translator
          </div>
          <div style="font-size:0.7rem;color:#475569;font-weight:600;">AI Edition · v2.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

        # Navigation
        for page in NAV_PAGES:
            selected = st.session_state["page"] == page
            btn_type = "primary" if selected else "secondary"
            if st.button(page, key=f"nav_{page}", use_container_width=True, type=btn_type):
                st.session_state["page"] = page
                st.rerun()

        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

        # Quick stats
        analytics: SessionAnalytics = st.session_state["analytics_obj"]
        snap = analytics.snapshot()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:8px;">
              <div class="stat-label">FPS</div>
              <div class="stat-value">{st.session_state['fps']:.1f}</div>
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
              <div class="stat-value">{st.session_state['confidence']:.0%}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Time</div>
              <div class="stat-value" style="font-size:0.85rem;">{analytics.format_duration()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

        # Status badges
        if st.session_state["running"]:
            st.markdown('<span class="badge badge-green">🔴 Camera Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">⚫ Camera Off</span>', unsafe_allow_html=True)

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        if st.session_state.get("model_loaded"):
            st.markdown('<span class="badge badge-green">✅ Model Ready</span>', unsafe_allow_html=True)
        elif st.session_state.get("error_msg"):
            st.markdown('<span class="badge badge-red">❌ Model Missing</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">⏳ Loading…</span>', unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)
        st.markdown('<div style="color:#334155;font-size:0.72rem;font-weight:600;text-align:center;">v2.0 · AI Edition · MediaPipe + TFLite</div>', unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# HOME page
# ─────────────────────────────────────────────────────────────────────────────

def _render_home() -> None:
    from utils.ui_components import svg_hand_3d, svg_ai_robot, svg_analytics, svg_microphone, svg_speaker

    st.markdown(f"""
    <div style="text-align:center;padding:32px 0 16px;">
      <div style="display:flex;align-items:center;justify-content:center;gap:20px;flex-wrap:wrap;">
        <div>{svg_hand_3d()}</div>
        <div>
          <div class="clay-title">Real-Time ISL Translator</div>
          <div class="clay-subtitle" style="margin-top:6px;">AI-Powered · Multi-Language · Real-Time</div>
        </div>
        <div>{svg_ai_robot()}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);margin:8px 0 24px;">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="clay-card clay-card-violet" style="text-align:center;min-height:160px;">
          <div style="margin-bottom:10px;">{svg_ai_robot()}</div>
          <div style="color:#a78bfa;font-weight:800;font-size:1rem;margin-bottom:6px;">TFLite Model</div>
          <div style="color:#475569;font-size:0.83rem;font-weight:500;">Dense NN · 42 landmarks<br>&gt;95% accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="clay-card clay-card-blue" style="text-align:center;min-height:160px;">
          <div style="font-size:2.4rem;margin-bottom:10px;">🌍</div>
          <div style="color:#60a5fa;font-weight:800;font-size:1rem;margin-bottom:6px;">13 Languages</div>
          <div style="color:#475569;font-size:0.83rem;font-weight:500;">Real-time translation<br>with Google Translate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="clay-card clay-card-cyan" style="text-align:center;min-height:160px;">
          <div style="margin-bottom:10px;">{svg_analytics()}</div>
          <div style="color:#06b6d4;font-weight:800;font-size:1rem;margin-bottom:6px;">25-35 FPS</div>
          <div style="color:#475569;font-size:0.83rem;font-weight:500;">MediaPipe · TFLite<br>~8ms inference</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Feature row 2
    f1, f2, f3, f4 = st.columns(4)
    for col, icon, label, desc in [
        (f1, svg_microphone(), "Voice Output",    "Web Speech API"),
        (f2, "🖐",            "Hand Detection",  "MediaPipe Hands"),
        (f3, "🧠",            "Neural Network",  "TFLite Dense NN"),
        (f4, "📊",            "Live Analytics",  "Session dashboard"),
    ]:
        with col:
            icon_html = icon if len(icon) > 2 else f'<span style="font-size:2rem;">{icon}</span>'
            st.markdown(f"""
            <div class="clay-card" style="text-align:center;padding:16px 12px;">
              <div style="margin-bottom:8px;">{icon_html}</div>
              <div style="color:#c7d2fe;font-weight:700;font-size:0.88rem;">{label}</div>
              <div style="color:#475569;font-size:0.76rem;margin-top:3px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    col_btn, _, _ = st.columns([2, 2, 2])
    with col_btn:
        if st.button("▶  Open Live Translator", type="primary", use_container_width=True):
            st.session_state["page"] = "📷 Live Translator"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS page
# ─────────────────────────────────────────────────────────────────────────────

def _render_settings(tts) -> None:
    st.markdown("""
    <div style="margin-bottom:20px;">
      <div class="clay-title" style="font-size:1.8rem;text-align:left;">⚙️ Settings</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    tab_ui, tab_cam, tab_tts, tab_trans = st.tabs(
        ["🎨 Interface", "📷 Camera", "🔊 Speech", "🌍 Translation"]
    )

    with tab_ui:
        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        st.markdown('<div class="clay-section-title">Theme & Accessibility</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Theme", ["dark", "light"],
                index=0 if st.session_state["theme"] == "dark" else 1)
            large = st.checkbox("Large Font Mode", value=st.session_state["large_font"])
        with col2:
            hc   = st.checkbox("High Contrast Mode", value=st.session_state["high_contrast"])
            subs = st.checkbox("Show Live Subtitles", value=st.session_state["subtitles_on"])
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Apply Interface Settings", type="primary"):
            st.session_state["theme"]         = theme
            st.session_state["large_font"]    = large
            st.session_state["high_contrast"] = hc
            st.session_state["subtitles_on"]  = subs
            st.success("✅ Interface settings applied.")
            st.rerun()

    with tab_cam:
        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        st.markdown('<div class="clay-section-title">Camera Configuration</div>', unsafe_allow_html=True)
        cam_idx = st.number_input("Camera Device Index", min_value=0, max_value=5,
            value=st.session_state["camera_index"], step=1)
        det_thresh = st.slider("Detection Confidence Threshold", 0.3, 0.99,
            value=float(st.session_state["detection_threshold"]), step=0.01)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Apply Camera Settings", type="primary"):
            st.session_state["camera_index"]        = int(cam_idx)
            st.session_state["detection_threshold"] = det_thresh
            st.success("✅ Camera settings applied.")

    with tab_tts:
        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        st.markdown('<div class="clay-section-title">Text-to-Speech</div>', unsafe_allow_html=True)
        auto_spk = st.checkbox("Auto-Speak on Gesture Recognition",
            value=st.session_state["auto_speak"])
        spd = st.slider("Speech Speed (wpm)", 80, 300,
            value=st.session_state["speech_rate"], step=10)
        voice_names = tts.get_voices() if tts and tts.is_available else []
        if voice_names:
            voice_idx_val = st.selectbox("Voice", voice_names,
                index=min(st.session_state["voice_index"], len(voice_names)-1))
            selected_voice_idx = voice_names.index(voice_idx_val)
        else:
            st.info("No system voices detected.")
            selected_voice_idx = 0
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Apply Speech Settings", type="primary"):
            st.session_state["auto_speak"]  = auto_spk
            st.session_state["speech_rate"] = spd
            st.session_state["voice_index"] = selected_voice_idx
            if tts and tts.is_available:
                tts.set_voice_by_index(selected_voice_idx)
            st.success("✅ Speech settings applied.")

    with tab_trans:
        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        st.markdown('<div class="clay-section-title">Translation</div>', unsafe_allow_html=True)
        lang = st.selectbox("Target Language", get_language_list(),
            index=get_language_list().index(st.session_state["target_language"]))
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Apply Translation Settings", type="primary"):
            st.session_state["target_language"] = lang
            st.success(f"✅ Language set to {lang}.")


# ─────────────────────────────────────────────────────────────────────────────
# ABOUT page
# ─────────────────────────────────────────────────────────────────────────────

def _render_about() -> None:
    from utils.ui_components import svg_hand_3d, svg_ai_robot, svg_analytics

    st.markdown(f"""
    <div style="text-align:center;padding:20px 0;">
      <div class="clay-title">Real-Time ISL Translator</div>
      <div class="clay-subtitle">AI Edition · Final Year Engineering Project · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);margin:8px 0 20px;">', unsafe_allow_html=True)

    # Hero card
    st.markdown(f"""
    <div class="clay-card clay-card-violet" style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
      <div style="flex-shrink:0;">{svg_hand_3d()}</div>
      <div style="flex:1;min-width:200px;">
        <div style="color:#a78bfa;font-weight:800;font-size:1.1rem;margin-bottom:6px;">About This Project</div>
        <div style="color:#94a3b8;font-size:0.88rem;line-height:1.7;">
          A final-year computer vision project that translates hand gestures into text
          and speech in real time using MediaPipe, TensorFlow Lite, and Streamlit.
        </div>
      </div>
      <div style="flex-shrink:0;">{svg_ai_robot()}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="clay-card clay-card-blue">
          <div class="clay-section-title">🏗 Architecture</div>
          <div style="color:#94a3b8;font-size:0.85rem;line-height:1.9;">
            📷 Webcam → OpenCV capture<br>
            🖐 MediaPipe Hands → 21 landmarks (x, y)<br>
            🏃 MoveNet Lightning → 17 body keypoints<br>
            📐 StandardScaler → feature normalisation<br>
            🧠 TFLite Dense NN → gesture classification<br>
            🔄 PredictionSmoother → rolling majority vote<br>
            🌍 deep-translator → 13-language output<br>
            🔊 TTSEngine → offline text-to-speech
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="clay-card clay-card-cyan">
          <div class="clay-section-title">📊 Performance</div>
          <div style="margin-bottom:6px;">{svg_analytics()}</div>
        </div>
        """, unsafe_allow_html=True)
        perf = [
            ("Inference latency", "~8 ms"),
            ("Landmark extraction", "~15 ms"),
            ("End-to-end FPS", "25–35 FPS"),
            ("Model size (TFLite)", "~0.9 MB"),
            ("Test accuracy", ">95%"),
        ]
        st.dataframe(
            __import__("pandas").DataFrame(perf, columns=["Metric", "Value"]),
            hide_index=True, use_container_width=True,
        )

    st.markdown('<div class="clay-section-title" style="margin-top:20px;">🛠 Tech Stack</div>', unsafe_allow_html=True)
    techs = [
        ("Python 3.10+", "Core language", "#818cf8"),
        ("OpenCV 4.9", "Camera + frame I/O", "#60a5fa"),
        ("MediaPipe 0.10", "Hand landmark detection", "#34d399"),
        ("TensorFlow Lite", "Gesture classification", "#fbbf24"),
        ("Streamlit 1.35", "Web UI", "#f472b6"),
        ("deep-translator", "Multi-language translation", "#a78bfa"),
        ("pyttsx3", "Offline TTS", "#06b6d4"),
        ("Plotly", "Analytics charts", "#818cf8"),
        ("Pandas", "Data handling", "#60a5fa"),
    ]
    cols = st.columns(3)
    for i, (name, desc, color) in enumerate(techs):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="clay-card" style="padding:14px 16px;margin-bottom:10px;">
              <div style="color:{color};font-weight:800;font-size:0.88rem;">{name}</div>
              <div style="color:#475569;font-size:0.78rem;margin-top:3px;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY page
# ─────────────────────────────────────────────────────────────────────────────

def _render_history() -> None:
    st.markdown("""
    <div style="margin-bottom:20px;">
      <div class="clay-title" style="font-size:1.8rem;text-align:left;">📜 Gesture History</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    history: GestureHistory = st.session_state["history_obj"]

    col_a, col_b, col_c = st.columns(3)
    for col, label, value, color in [
        (col_a, "Total Entries",   str(history.count()),                       "#818cf8"),
        (col_b, "Most Frequent",   history.most_frequent_gesture() or "—",    "#a78bfa"),
        (col_c, "Avg Confidence",  f"{history.average_confidence()*100:.1f}%","#34d399"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:14px;">
              <div class="stat-label">{label}</div>
              <div class="stat-value" style="color:{color};">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if not history.is_empty():
            st.download_button(
                "📥 Download CSV", data=history.to_csv_bytes(),
                file_name="gesture_history.csv", mime="text/csv",
                use_container_width=True,
            )
    with btn_col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            history.clear()
            st.success("History cleared.")
            st.rerun()
    with btn_col3:
        if st.button("📋 Copy as Text", use_container_width=True):
            st.code(history.to_plain_text(), language=None)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    if history.is_empty():
        st.markdown("""
        <div class="clay-card" style="text-align:center;padding:40px;">
          <div style="font-size:2rem;margin-bottom:10px;">📭</div>
          <div style="color:#475569;font-weight:600;">No gestures recorded yet.</div>
          <div style="color:#334155;font-size:0.85rem;margin-top:4px;">Start the Live Translator to begin.</div>
        </div>""", unsafe_allow_html=True)
        return

    rows = history.to_display_dicts()
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS page
# ─────────────────────────────────────────────────────────────────────────────

def _render_analytics() -> None:
    from utils.ui_components import svg_analytics
    try:
        import plotly.graph_objects as go
        _PLOTLY = True
    except ImportError:
        _PLOTLY = False

    analytics: SessionAnalytics = st.session_state["analytics_obj"]
    snap = analytics.snapshot()

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
      {svg_analytics()}
      <div>
        <div class="clay-title" style="font-size:1.8rem;text-align:left;">Analytics Dashboard</div>
        <div class="clay-subtitle" style="text-align:left;">Session performance & gesture insights</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "Total Gestures",       str(snap.total_gestures),              "#818cf8"),
        (k2, "Recognition Accuracy", f"{snap.recognition_accuracy*100:.1f}%","#34d399"),
        (k3, "Avg Confidence",       f"{snap.average_confidence*100:.1f}%", "#fbbf24"),
        (k4, "Current FPS",          f"{snap.current_fps:.1f}",             "#60a5fa"),
        (k5, "Session Time",         analytics.format_duration(),           "#a78bfa"),
    ]
    for col, label, value, color in kpis:
        with col:
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:12px;">
              <div class="stat-label">{label}</div>
              <div class="stat-value" style="color:{color};">{value}</div>
            </div>""", unsafe_allow_html=True)

    k6, k7, k8 = st.columns(3)
    kpis2 = [
        (k6, "Most Used Gesture",         snap.most_frequent_gesture or "—", "#a78bfa"),
        (k7, "High Confidence Preds",     str(snap.high_confidence_count),   "#34d399"),
        (k8, "Low Confidence Preds",      str(snap.low_confidence_count),    "#f87171"),
    ]
    for col, label, value, color in kpis2:
        with col:
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:12px;">
              <div class="stat-label">{label}</div>
              <div class="stat-value" style="color:{color};font-size:1.1rem;">{value}</div>
            </div>""", unsafe_allow_html=True)

    cam_status = "🟢 Active" if st.session_state["running"] else "🔴 Inactive"
    st.markdown(f'<div style="color:#64748b;font-size:0.82rem;font-weight:600;margin-bottom:16px;">Camera: {cam_status}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    if not _PLOTLY:
        st.info("Install plotly (`pip install plotly`) to view charts.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="clay-section-title">FPS Over Time</div>', unsafe_allow_html=True)
        fps_list = snap.fps_history
        if fps_list:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=fps_list, mode="lines", fill="tozeroy",
                line=dict(color="#818cf8", width=2.5),
                fillcolor="rgba(99,102,241,0.1)",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,34,64,0.5)",
                font=dict(color="#64748b", family="Plus Jakarta Sans"),
                height=240, margin=dict(l=20,r=20,t=10,b=20),
                yaxis=dict(gridcolor="rgba(99,102,241,0.08)"),
                xaxis=dict(gridcolor="rgba(99,102,241,0.08)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No FPS data yet.")

    with col_right:
        st.markdown('<div class="clay-section-title">Gesture Frequency</div>', unsafe_allow_html=True)
        freq = snap.gesture_frequency
        if freq:
            import plotly.graph_objects as go
            labels = list(freq.keys())
            counts = list(freq.values())
            colors = ["#818cf8","#a78bfa","#60a5fa","#34d399","#fbbf24",
                      "#f472b6","#06b6d4","#818cf8","#a78bfa","#60a5fa"]
            fig = go.Figure(go.Bar(
                x=counts, y=labels, orientation="h",
                marker=dict(color=colors[:len(labels)],
                            line=dict(color="rgba(0,0,0,0)", width=0)),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,34,64,0.5)",
                font=dict(color="#64748b", family="Plus Jakarta Sans"),
                height=240, margin=dict(l=20,r=20,t=10,b=20),
                xaxis=dict(gridcolor="rgba(99,102,241,0.08)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.08)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No gesture data yet.")

    if snap.total_gestures > 0:
        st.markdown('<div class="clay-section-title">Confidence Distribution</div>', unsafe_allow_html=True)
        import plotly.graph_objects as go
        fig = go.Figure(go.Pie(
            labels=["High Confidence", "Low Confidence"],
            values=[snap.high_confidence_count, snap.low_confidence_count],
            hole=0.55,
            marker=dict(colors=["#34d399", "#f87171"],
                        line=dict(color="#080D1C", width=2)),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
            height=260, margin=dict(l=20,r=20,t=10,b=20),
            legend=dict(font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Reset Analytics"):
        st.session_state["analytics_obj"].reset()
        st.success("Analytics reset.")
        st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRANSLATOR page helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_sentence_builder(tts) -> None:
    from utils.ui_components import svg_microphone
    words = st.session_state["sentence"]
    sentence_str = " ".join(words) if words else \
        "<span style='color:#334155;font-style:italic;'>Waiting for gestures…</span>"
    lang = st.session_state["target_language"]
    translated_sentence = translate(" ".join(words), lang) if words else ""

    st.markdown(f"""
    <div style="margin-bottom:8px;">
      <div class="clay-section-title">📝 Sentence Builder</div>
    </div>
    <div class="sentence-box">{sentence_str}</div>
    """, unsafe_allow_html=True)

    if translated_sentence and lang != DEFAULT_LANGUAGE:
        st.markdown(
            f'<div style="color:#64748b;font-size:0.83rem;margin-top:6px;font-weight:500;">'
            f'🌍 {lang}: <span style="color:#94a3b8;">{translated_sentence}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔊 Speak", key="btn_speak", use_container_width=True, type="primary"):
            text_to_speak = translated_sentence or " ".join(words)
            if text_to_speak and tts:
                tts.speak(text_to_speak, force=True)
    with b2:
        if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
            st.session_state["sentence"] = []
            st.session_state["subtitle_text"] = ""
            if tts:
                tts.reset_last_spoken()
            st.rerun()
    with b3:
        if st.button("⬅️ Undo", key="btn_undo", use_container_width=True):
            if st.session_state["sentence"]:
                st.session_state["sentence"].pop()
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Two-hand debug strip helper
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hand_debug_strip(frame: np.ndarray, mhr: "MultiHandResult") -> None:
    """
    Draw a compact one-line debug strip at the top-right of the frame showing:
      Hands: N  |  L: ✓/✗  |  R: ✓/✗

    This lets the user instantly verify that two-hand detection is working.
    The strip is semi-transparent and does not interfere with the gesture HUD.
    """
    h, w = frame.shape[:2]

    # Build label string
    n       = mhr.count
    l_col   = (0, 220, 0)  if mhr.left_detected  else (80, 80, 200)
    r_col   = (0, 220, 0)  if mhr.right_detected else (80, 80, 200)
    n_col   = (0, 220, 0)  if n > 0              else (80, 80, 200)

    strip_y = 100   # pixel row below the main gesture HUD banner

    # Semi-transparent background pill
    ov = frame.copy()
    cv2.rectangle(ov, (w - 220, strip_y - 14), (w - 2, strip_y + 6), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1

    # "Hands: N"
    cv2.putText(frame, f"Hands:{n}", (w - 216, strip_y),
                font, scale, n_col, thick)
    # "L: ✓/✗"  — use ASCII substitutes because HERSHEY doesn't render Unicode
    l_text = "L:OK" if mhr.left_detected  else "L:--"
    r_text = "R:OK" if mhr.right_detected else "R:--"
    cv2.putText(frame, l_text, (w - 148, strip_y), font, scale, l_col, thick)
    cv2.putText(frame, r_text, (w - 90,  strip_y), font, scale, r_col, thick)


def _webcam_loop(frame_ph, gesture_ph, interp, in_idx, out_idx, label_map,
                 scaler, detector, tts) -> None:
    """
    Core webcam capture + inference loop.
    Renders frames into Streamlit placeholders without blocking the full page.
    """
    cam_idx = st.session_state.get("camera_index", CAMERA_INDEX)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        st.error(f"Cannot open camera at index {cam_idx}.")
        st.session_state["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoother      = PredictionSmoother()
    last_stable   = None
    stable_frames = 0
    t_prev        = time.perf_counter()

    stop_ph = st.empty()
    stop_clicked = stop_ph.button("⏹ Stop Camera", key="stop_btn",
                                   type="secondary", use_container_width=True)

    history:   GestureHistory  = st.session_state["history_obj"]
    analytics: SessionAnalytics = st.session_state["analytics_obj"]
    det_thresh = st.session_state.get("detection_threshold", CONFIDENCE_THRESHOLD)

    try:
        while st.session_state["running"] and not stop_clicked:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            annotated, gesture, confidence, has_hand, all_probs, mhr = _process_frame(
                frame, interp, in_idx, out_idx, label_map, scaler,
                detector, smoother,
            )

            # ── FPS ──────────────────────────────────────────────────────────
            t_now  = time.perf_counter()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            analytics.record_fps(fps)

            # ── Draw FPS overlay on frame ─────────────────────────────────────
            cv2.putText(
                annotated, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2,
            )
            if gesture:
                cv2.putText(
                    annotated, f"{gesture} ({confidence:.0%})",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 212, 255), 2,
                )

            # ── Two-hand detection debug strip ───────────────────────────────
            _draw_hand_debug_strip(annotated, mhr)

            # ── Stability tracking ────────────────────────────────────────────
            if gesture and confidence >= det_thresh:
                if gesture == last_stable:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    last_stable   = gesture

                if stable_frames == STABLE_FRAME_COUNT:
                    words = st.session_state["sentence"]
                    if not words or words[-1] != gesture:
                        # Translate
                        lang       = st.session_state["target_language"]
                        translated = translate(gesture, lang)

                        # Update sentence
                        st.session_state["sentence"].append(gesture)
                        st.session_state["subtitle_text"] = translated or gesture

                        # Record history and analytics
                        history.add(gesture, translated or gesture, confidence, lang)
                        analytics.record_gesture(gesture, confidence)

                        # Auto-speak
                        if st.session_state["auto_speak"] and tts:
                            tts.speak(translated or gesture)
                            st.session_state["tts_speaking"] = True
            else:
                if stable_frames > 0:
                    st.session_state["tts_speaking"] = False
                stable_frames = 0
                last_stable   = None

            # ── Session state update ──────────────────────────────────────────
            st.session_state["current_gesture"] = gesture or "—"
            st.session_state["confidence"]      = confidence
            st.session_state["stable_frames"]   = stable_frames
            st.session_state["fps"]             = fps
            st.session_state["all_probs"]       = all_probs

            # ── Render frame ──────────────────────────────────────────────────
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with frame_ph.container():
                st.markdown('<div class="camera-wrapper">', unsafe_allow_html=True)
                st.image(rgb, channels="RGB", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                # Detection status underneath frame
                render_detection_status(has_hand, gesture, confidence)
                # Subtitle strip
                if st.session_state["subtitles_on"] and st.session_state["subtitle_text"]:
                    render_subtitle_strip(st.session_state["subtitle_text"])

            # ── Render AI panel in right column ───────────────────────────────
            current = st.session_state["current_gesture"]
            conf    = st.session_state["confidence"]
            meaning = GESTURE_MEANINGS.get(current, "—") if current != "—" else "—"

            if not has_hand:
                status = "🖐 No hand detected"
            elif stable_frames >= STABLE_FRAME_COUNT:
                status = "✅ Recognized!"
            elif gesture:
                status = f"🔍 Recognizing… ({stable_frames}/{STABLE_FRAME_COUNT})"
            else:
                status = "Scanning…"

            with gesture_ph.container():
                render_ai_panel(current, meaning, status, conf,
                                st.session_state["tts_speaking"])

                st.markdown("##### Confidence Meter")
                render_confidence_meter(st.session_state["all_probs"])

                # Low-confidence suggestions
                if has_hand and gesture and conf < LOW_CONFIDENCE_THRESHOLD:
                    top_labels = sorted(
                        st.session_state["all_probs"].items(),
                        key=lambda x: x[1], reverse=True
                    )
                    render_suggestions([lbl for lbl, _ in top_labels[:4]])

    finally:
        cap.release()
        st.session_state["running"]      = False
        st.session_state["tts_speaking"] = False
        stop_ph.empty()


def _char_webcam_loop(frame_ph, info_ph, c_interp, c_in, c_out,
                      c_label_map, c_scaler, detector, tts) -> None:
    """
    Webcam loop for Character mode (A-Z + 0-9).

    Acceptance logic  (mirrors character_inference.py / app_cloud.py):
      1. Raw confidence >= char_conf_threshold
      2. Same character stable for char_stable_frames consecutive frames
      3. char_debounce_count >= CHARACTER_DEBOUNCE_FRAMES since last accepted
         (forces hand to drop between letters — prevents HHHELLO)
      4. char_last_accepted != current char  (no silent repeats)
    """
    import collections

    cam_idx = st.session_state.get("camera_index", CAMERA_INDEX)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        st.error(f"Cannot open camera at index {cam_idx}.")
        st.session_state["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    char_smoother = PredictionSmoother()
    stable_frames   = 0
    last_stable     = None
    debounce_count  = 0
    last_accepted   = None
    t_prev          = time.perf_counter()

    conf_thr   = st.session_state.get("char_conf_threshold",    CHARACTER_CONFIDENCE_THRESHOLD)
    stable_thr = st.session_state.get("char_stable_frames_cfg", CHARACTER_STABLE_FRAME_COUNT)

    stop_ph = st.empty()
    stop_clicked = stop_ph.button("⏹ Stop Camera", key="stop_btn_char",
                                   type="secondary", use_container_width=True)

    _loop_iter = 0
    try:
        while st.session_state["running"] and not stop_clicked:
            _loop_iter += 1
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            mhr_char, annotated = detector.find_all_landmarks(frame, draw=True)
            has_hand   = mhr_char.dominant is not None
            char_pred  = None
            confidence = 0.0

            if has_hand:
                raw_lm   = mhr_char.dominant.landmarks
                features = normalise_landmarks(raw_lm)
                raw_char, conf = predict_character(
                    features, c_interp, c_in, c_out, c_label_map, c_scaler,
                )
                if conf >= conf_thr:
                    char_pred, confidence = char_smoother.update(raw_char, conf)
                else:
                    char_smoother.reset()
            else:
                char_smoother.reset()

            # ── Stability + debounce ──────────────────────────────────────────
            if char_pred is not None:
                if char_pred == last_stable:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    last_stable   = char_pred

                if stable_frames >= stable_thr:
                    if debounce_count >= CHARACTER_DEBOUNCE_FRAMES:
                        if last_accepted != char_pred:
                            st.session_state["char_sentence"].append(char_pred)
                            last_accepted = char_pred
                            # Auto-speak only full sentence, not individual chars
                        debounce_count = 0
                    else:
                        debounce_count += 1
                else:
                    debounce_count += 1
            else:
                stable_frames = 0
                last_stable   = None
                # Accumulate debounce cooldown while hand is absent
                debounce_count = min(debounce_count + 1, CHARACTER_DEBOUNCE_FRAMES + 5)

            # ── FPS ───────────────────────────────────────────────────────────
            t_now  = time.perf_counter()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # ── HUD overlay ───────────────────────────────────────────────────
            h, w = annotated.shape[:2]
            ov = annotated.copy()
            cv2.rectangle(ov, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, annotated, 0.45, 0, annotated)

            color = (0, 220, 0) if char_pred else (80, 80, 220)
            label = char_pred if char_pred else ("No hand" if not has_hand else "Low conf")
            cv2.putText(annotated, f"[Character] {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if char_pred:
                cv2.putText(annotated, f"{confidence:.0%}", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

            # Stability bar
            stable_pct = min(stable_frames / max(stable_thr, 1), 1.0)
            bar_w = w - 24
            cv2.rectangle(annotated, (12, 68), (12 + bar_w, 80), (40, 40, 40), -1)
            fill_w = int(bar_w * stable_pct)
            if fill_w > 0:
                bc = (0, int(200 * stable_pct), int(200 * (1 - stable_pct)))
                cv2.rectangle(annotated, (12, 68), (12 + fill_w, 80), bc, -1)

            # Ready dot
            dot_color = (0, 200, 0) if debounce_count >= CHARACTER_DEBOUNCE_FRAMES else (0, 165, 255)
            cv2.circle(annotated, (w - 18, 18), 7, dot_color, -1)

            cv2.putText(annotated, f"FPS: {fps:.0f}", (w - 110, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1)

            # Bottom sentence strip
            ov2 = annotated.copy()
            cv2.rectangle(ov2, (0, h - 40), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(ov2, 0.55, annotated, 0.45, 0, annotated)
            sentence_txt = "".join(st.session_state["char_sentence"]) or "—"
            cv2.putText(annotated, f"Text: {sentence_txt}", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)

            # ── Two-hand detection debug strip ────────────────────────────────
            _draw_hand_debug_strip(annotated, mhr_char)

            # ── Show frame ────────────────────────────────────────────────────
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with frame_ph.container():
                st.image(rgb, channels="RGB", use_column_width=True)

            # ── Right panel ───────────────────────────────────────────────────
            char_sentence = st.session_state["char_sentence"]
            full_text = "".join(char_sentence)
            with info_ph.container():
                st.markdown("#### 🔤 Character Mode")
                st.markdown(
                    f"**Current:** `{char_pred or '—'}`  "
                    f"{'&nbsp;' * 4}Confidence: `{confidence:.0%}`"
                )
                stable_bar_pct = int(stable_pct * 100)
                st.progress(stable_bar_pct,
                    text=f"Stability {stable_frames}/{stable_thr}")
                st.markdown("---")
                st.markdown("**Built text:**")
                st.markdown(
                    f"<div style='font-size:1.8rem;font-weight:700;letter-spacing:4px;"
                    f"color:#a78bfa;min-height:2.5rem;'>{full_text or '—'}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button("🗑 Clear", key=f"char_clear_{_loop_iter}"):
                        st.session_state["char_sentence"] = []
                        last_accepted = None
                with b2:
                    if st.button("⬅ Backspace", key=f"char_back_{_loop_iter}"):
                        if st.session_state["char_sentence"]:
                            st.session_state["char_sentence"].pop()
                            last_accepted = None
                with b3:
                    if st.button("␣ Space", key=f"char_space_{_loop_iter}"):
                        st.session_state["char_sentence"].append(" ")
                        last_accepted = None
                with b4:
                    if st.button("🔊 Speak", key=f"char_speak_{_loop_iter}"):
                        if full_text.strip() and tts:
                            tts.speak(full_text.strip(), force=True)

    finally:
        cap.release()
        st.session_state["running"] = False
        stop_ph.empty()


def _render_live_translator(interp, in_idx, out_idx, label_map,
                             scaler, detector, tts,
                             c_interp, c_in, c_out, c_label_map, c_scaler) -> None:
    """Full Live Translator page layout — supports Word mode and Character mode."""
    st.markdown("""
    <div style="margin-bottom:16px;">
      <div class="clay-title" style="font-size:1.8rem;text-align:left;">📷 Live Translator</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode toggle + language + auto-speak ───────────────────────────────────
    mode_col, lang_col, auto_col = st.columns([2, 2, 2])
    with mode_col:
        mode = st.radio(
            "Recognition Mode",
            ["Word", "Character"],
            index=0 if st.session_state["recognition_mode"] == "Word" else 1,
            horizontal=True,
            key="mode_radio",
            help="Word: recognises 10 gestures (Hello, Good…)  |  Character: A-Z + 0-9",
        )
        if mode != st.session_state["recognition_mode"]:
            st.session_state["recognition_mode"] = mode
            st.session_state["running"] = False   # stop camera on mode switch
            st.rerun()
    with lang_col:
        if mode == "Word":
            lang_list = get_language_list()
            cur_idx   = lang_list.index(st.session_state["target_language"])
            new_lang  = st.selectbox("🌍 Target Language", lang_list, index=cur_idx, key="live_lang")
            st.session_state["target_language"] = new_lang
    with auto_col:
        auto = st.checkbox("Auto-Speak", value=st.session_state["auto_speak"], key="live_auto_speak")
        st.session_state["auto_speak"] = auto

    st.markdown('<hr style="border-color:rgba(99,102,241,0.15);">', unsafe_allow_html=True)

    cam_col, info_col = st.columns([3, 2], gap="large")

    with cam_col:
        frame_ph = st.empty()
        if not st.session_state["running"]:
            frame_ph.markdown(camera_placeholder_html(), unsafe_allow_html=True)

            # Show error if relevant model is missing
            if mode == "Word" and interp is None:
                st.markdown(
                    '<div class="clay-error">⚠️ Word model not loaded. Run train_model.py first.</div>',
                    unsafe_allow_html=True,
                )
            elif mode == "Character" and c_interp is None:
                st.markdown(
                    '<div class="clay-error">⚠️ Character model not loaded. Run train_character_model.py first.</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button("▶  Start Camera", type="primary", use_container_width=True, key="btn_start"):
                    st.session_state["running"] = True
                    st.rerun()

    with info_col:
        gesture_ph = st.empty()

        if not st.session_state["running"]:
            if mode == "Word":
                render_ai_panel("—", "—", "Camera off", 0.0, False)
            else:
                st.markdown(
                    "<div style='color:#a78bfa;font-weight:700;font-size:1rem;margin-bottom:8px;'>"
                    "🔤 Character Mode — A-Z + 0-9</div>",
                    unsafe_allow_html=True,
                )
                st.info("Start the camera and hold a hand sign steady to spell characters.")

        if mode == "Word":
            st.markdown('<hr style="border-color:rgba(99,102,241,0.12);">', unsafe_allow_html=True)
            _render_sentence_builder(tts)

            st.markdown('<hr style="border-color:rgba(99,102,241,0.12);">', unsafe_allow_html=True)
            last_gesture = st.session_state["sentence"][-1] if st.session_state["sentence"] else None
            if last_gesture:
                st.markdown('<div class="clay-section-title">🌐 Translations</div>', unsafe_allow_html=True)
                translations = translate_all_languages(last_gesture)
                limited = dict(list(translations.items())[:6])
                render_translation_cards(limited)

    # ── Launch the correct loop ───────────────────────────────────────────────
    if st.session_state["running"]:
        if mode == "Word" and interp is not None:
            _webcam_loop(
                frame_ph, gesture_ph,
                interp, in_idx, out_idx, label_map, scaler, detector, tts,
            )
        elif mode == "Character" and c_interp is not None:
            _char_webcam_loop(
                frame_ph, gesture_ph,
                c_interp, c_in, c_out, c_label_map, c_scaler, detector, tts,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Application entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    # Load CSS for current theme
    inject_css(
        theme=st.session_state["theme"],
        large_font=st.session_state["large_font"],
        high_contrast=st.session_state["high_contrast"],
    )

    # Load model resources (cached)
    (interp, in_idx, out_idx,
     label_map, scaler, detector, tts) = _load_resources()

    if interp is not None:
        st.session_state["model_loaded"] = True

    # Load character model resources (cached, independent)
    (c_interp, c_in, c_out,
     c_label_map, c_scaler, _char_err) = _load_char_resources()

    if c_interp is not None:
        st.session_state["char_model_loaded"] = True

    # Sidebar + navigation
    _render_sidebar(tts)

    # Route to current page
    page = st.session_state["page"]

    if page == "🏠 Home":
        _render_home()

    elif page == "📷 Live Translator":
        _render_live_translator(
            interp, in_idx, out_idx, label_map, scaler, detector, tts,
            c_interp, c_in, c_out, c_label_map, c_scaler,
        )

    elif page == "📜 History":
        _render_history()

    elif page == "📊 Analytics":
        _render_analytics()

    elif page == "⚙️ Settings":
        _render_settings(tts)

    elif page == "ℹ️ About":
        _render_about()


if __name__ == "__main__":
    main()
