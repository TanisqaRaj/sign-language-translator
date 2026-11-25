# =============================================================================
# app.py  –  Local Streamlit entry point
# Sign Language Translator – Professional Edition
# =============================================================================
# Runs with:  streamlit run app.py
#
# Uses local cv2.VideoCapture for webcam access.
# All page logic lives in app_pages/; ML in inference.py / preprocess.py.
# =============================================================================

import os
import sys

# Suppress TensorFlow / MediaPipe / absl noise before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"   # hides TF deprecated warnings
os.environ["GLOG_minloglevel"]        = "2"   # hides MediaPipe W/E/F logs
os.environ["MEDIAPIPE_DISABLE_GPU"]   = "1"

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Config & constants ────────────────────────────────────────────────────────
from config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    STABLE_FRAME_COUNT,
    PREDICTION_BUFFER_LEN,
    MODEL_TFLITE_PATH,
    LABEL_MAP_PATH,
    MODEL_DIR,
    GESTURE_LABELS,
    GESTURE_MEANINGS,
    LOW_CONFIDENCE_THRESHOLD,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_THEME,
    DEFAULT_FONT_SIZE,
    AUTO_SPEAK_DEFAULT,
    SUBTITLE_ENABLED_DEFAULT,
    CHARACTER_CONFIDENCE_THRESHOLD,
    CHARACTER_STABLE_FRAME_COUNT,
    CHARACTER_DEBOUNCE_FRAMES,
    CHARACTER_LABELS,
)

# ── Shared utilities ──────────────────────────────────────────────────────────
from utils.logger           import get_logger
from utils.mediapipe_helper import HandDetector
from utils.tts_engine       import TTSEngine
from utils.translator       import translate, translate_all_languages, get_language_list
from utils.gesture_history  import GestureHistory
from utils.analytics        import SessionAnalytics
from utils.ui_components    import (
    inject_css,
    svg_hand_3d,
    render_ai_panel,
)

# ── ML loading ────────────────────────────────────────────────────────────────
from inference import (
    load_tflite_model,
    load_label_map,
    load_scaler,
    predict,
    PredictionSmoother,
    load_character_tflite_model,
    load_character_label_map,
    load_character_scaler,
    predict_character,
)
from preprocess import normalise_landmarks

# ── Page modules ──────────────────────────────────────────────────────────────
from app_pages.home            import show_home
from app_pages.live_translator import show_live_translator
from app_pages.history         import show_history
from app_pages.analytics       import show_analytics
from app_pages.settings        import show_settings_local
from app_pages.about           import show_about

logger = get_logger(__name__)

# Navigation pages shown in the sidebar (matches app_cloud.py exactly)
_NAV = [
    "🏠 Home",
    "📷 Live Translator",
    "📜 History",
    "📊 Analytics",
    "⚙️ Settings",
    "ℹ️ About",
]

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Translator",
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
        "page":              _NAV[0],
        # Theme / accessibility
        "theme":             DEFAULT_THEME,
        "large_font":        (DEFAULT_FONT_SIZE == "large"),
        "high_contrast":     False,
        # Camera / inference
        "running":           False,
        "current_gesture":   "—",
        "confidence":        0.0,
        "stable_frames":     0,
        "all_probs":         {},
        "fps":               0.0,
        "model_loaded":      False,
        "char_model_loaded": False,
        "error_msg":         "",
        # Recognition mode: "Word" or "Character"
        "recognition_mode":  "Word",
        # Sentence builder
        "sentence":          [],
        # Character sentence
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
        # History & analytics objects
        "history_obj":       None,
        "analytics_obj":     None,
        # Camera settings
        "camera_index":      CAMERA_INDEX,
        "detection_threshold": CONFIDENCE_THRESHOLD,
        # Character model thresholds
        "char_conf_threshold":    CHARACTER_CONFIDENCE_THRESHOLD,
        "char_stable_frames_cfg": CHARACTER_STABLE_FRAME_COUNT,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state["history_obj"] is None:
        st.session_state["history_obj"] = GestureHistory()
    if st.session_state["analytics_obj"] is None:
        obj = SessionAnalytics()
        obj.start_session()
        st.session_state["analytics_obj"] = obj


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_resources():
    try:
        interp, in_idx, out_idx = load_tflite_model()
        label_map = load_label_map()
        scaler    = load_scaler()
        detector  = HandDetector()
        tts       = TTSEngine()
        return interp, in_idx, out_idx, label_map, scaler, detector, tts
    except Exception as exc:
        st.session_state["error_msg"] = str(exc)
        return (None,) * 7


@st.cache_resource(show_spinner=False)
def _load_char_resources():
    try:
        c_interp, c_in, c_out = load_character_tflite_model()
        c_label_map = load_character_label_map()
        c_scaler    = load_character_scaler()
        return c_interp, c_in, c_out, c_label_map, c_scaler, None
    except Exception as exc:
        return None, None, None, None, None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(tts) -> None:
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
        analytics: SessionAnalytics = st.session_state["analytics_obj"]
        snap = analytics.snapshot()

        st.markdown('<div class="clay-section-title">Live Stats</div>', unsafe_allow_html=True)
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
              <div class="stat-value" style="font-size:0.82rem;">{analytics.format_duration()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr style="border-top:1px solid rgba(131,166,206,0.2);">', unsafe_allow_html=True)

        # ── Status indicators ──────────────────────────────────────────────────
        if st.session_state["running"]:
            st.markdown('<span class="badge badge-green">● Camera Active</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">○ Camera Off</span>',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        if st.session_state.get("model_loaded"):
            st.markdown('<span class="badge badge-green">✓ Model Ready</span>',
                        unsafe_allow_html=True)
        elif st.session_state.get("error_msg"):
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
# Application entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    inject_css(
        theme=st.session_state["theme"],
        large_font=st.session_state["large_font"],
        high_contrast=st.session_state["high_contrast"],
    )

    # Load models (cached)
    (interp, in_idx, out_idx,
     label_map, scaler, detector, tts) = _load_resources()

    if interp is not None:
        st.session_state["model_loaded"] = True

    (c_interp, c_in, c_out,
     c_label_map, c_scaler, _char_err) = _load_char_resources()

    if c_interp is not None:
        st.session_state["char_model_loaded"] = True

    _render_sidebar(tts)

    page = st.session_state["page"]

    if page == "🏠 Home":
        show_home()

    elif page == "📷 Live Translator":
        show_live_translator(
            interp, in_idx, out_idx, label_map, scaler, detector, tts,
            c_interp, c_in, c_out, c_label_map, c_scaler,
        )

    elif page == "📜 History":
        show_history(history_key="history_obj")

    elif page == "ℹ️ About":
        show_about()

    # Fallback – keep analytics/settings accessible if navigated to directly
    elif page == "📊 Analytics":
        show_analytics(
            analytics_key="analytics_obj",
            history_key="history_obj",
            use_plotly=True,
        )

    elif page == "⚙️ Settings":
        show_settings_local(tts)

    else:
        show_home()


if __name__ == "__main__":
    main()
