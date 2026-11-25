# =============================================================================
# pages/settings.py  –  Settings page
#
# Two distinct functions:
#   show_settings_local(tts)  →  used by app.py
#   show_settings_cloud()     →  used by app_cloud.py
#
# They differ in layout (tabs vs 2-column), session-state keys, and TTS
# handling.  No logic is shared beyond the page heading.
# =============================================================================

from __future__ import annotations

import streamlit as st

from config import (
    CONFIDENCE_THRESHOLD,
    STABLE_FRAME_COUNT,
    CAMERA_INDEX,
    DEFAULT_LANGUAGE,
    DEFAULT_THEME,
    DEFAULT_FONT_SIZE,
    DEFAULT_HIGH_CONTRAST,
    AUTO_SPEAK_DEFAULT,
    SUBTITLE_ENABLED_DEFAULT,
    SUPPORTED_LANGUAGES,
    CHARACTER_CONFIDENCE_THRESHOLD,
    CHARACTER_STABLE_FRAME_COUNT,
)
from utils.translator import get_language_list


# ---------------------------------------------------------------------------
# Premium CSS for both settings views
# ---------------------------------------------------------------------------
_SETTINGS_CSS = """
<style>
/* ── Hero banner ──────────────────────────────────────────────────── */
.settings-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0d1b2a 0%, #0f172a 45%, #1a103a 100%);
    border: 1px solid rgba(196,140,179,0.18);
    border-radius: 20px;
    padding: 30px 32px 26px;
    margin-bottom: 28px;
    animation: settings-fade-up 0.5s ease both;
}
.settings-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 55% 80% at 85% 50%,
            rgba(196,140,179,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 5% 80%,
            rgba(131,166,206,0.09) 0%, transparent 70%);
    pointer-events: none;
}
.settings-hero-inner {
    position: relative;
    z-index: 1;
}
.settings-hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0 0%, #C48CB3 50%, #83A6CE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 6px;
}
.settings-hero-sub {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 500;
}
.settings-hero-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.settings-hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(196,140,179,0.10);
    border: 1px solid rgba(196,140,179,0.22);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.73rem;
    color: #C48CB3;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.settings-hero-pill.blue {
    background: rgba(131,166,206,0.10);
    border-color: rgba(131,166,206,0.22);
    color: #83A6CE;
}

/* ── Premium tab cards ────────────────────────────────────────────── */
.settings-card {
    background: linear-gradient(145deg, rgba(8,13,28,0.9) 0%, rgba(15,23,42,0.95) 100%);
    border: 1px solid rgba(131,166,206,0.14);
    border-radius: 18px;
    padding: 22px 22px 18px;
    margin-bottom: 16px;
    animation: settings-fade-up 0.45s ease both;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.settings-card-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #83A6CE;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.settings-card-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(131,166,206,0.25), transparent);
    border-radius: 1px;
}

/* ── Theme preview swatches ───────────────────────────────────────── */
.theme-preview-row {
    display: flex;
    gap: 12px;
    margin: 12px 0 4px;
}
.theme-swatch {
    flex: 1;
    border-radius: 12px;
    padding: 14px 12px;
    border: 2px solid transparent;
    text-align: center;
    font-size: 0.75rem;
    font-weight: 600;
    transition: border-color 0.2s, box-shadow 0.2s;
    cursor: default;
}
.theme-swatch.dark-swatch {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    border-color: rgba(131,166,206,0.4);
}
.theme-swatch.light-swatch {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    color: #1e293b;
    border-color: rgba(196,140,179,0.3);
}
.theme-swatch.active-swatch {
    box-shadow: 0 0 16px rgba(131,166,206,0.25);
    border-color: #83A6CE;
}

/* ── Camera diagram ───────────────────────────────────────────────── */
.camera-diagram {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 16px;
    background: rgba(131,166,206,0.04);
    border: 1px dashed rgba(131,166,206,0.2);
    border-radius: 14px;
    margin-bottom: 14px;
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
}
.camera-diagram .cam-icon {
    font-size: 2.2rem;
    opacity: 0.75;
}
.camera-diagram-arrow {
    color: #83A6CE;
    font-size: 1.2rem;
    opacity: 0.6;
}

/* ── Speech speed visual ──────────────────────────────────────────── */
.speed-bar-row {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 32px;
    margin: 8px 0 4px;
}
.speed-bar {
    flex: 1;
    border-radius: 4px 4px 0 0;
    background: linear-gradient(180deg, #83A6CE, rgba(131,166,206,0.3));
    transition: height 0.3s ease;
}

/* ── Section titles (cloud view) ──────────────────────────────────── */
.cloud-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #83A6CE;
    margin: 20px 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.cloud-section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(131,166,206,0.2), transparent);
    border-radius: 1px;
}

/* ── Summary stat cards ───────────────────────────────────────────── */
/* inherits .stat-card from CLAY_CSS — no override needed */

/* ── Glow on active toggles ───────────────────────────────────────── */
div[data-testid="stCheckbox"] label:has(input:checked) span:first-child,
div[data-testid="stToggle"]:has(input:checked) div[data-testid="stToggleSwitch"] {
    box-shadow: 0 0 10px rgba(131,166,206,0.4);
}

/* ── Apply / Reset button styling ────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #83A6CE, #C48CB3) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(131,166,206,0.25) !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 22px rgba(131,166,206,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Animations ───────────────────────────────────────────────────── */
@keyframes settings-fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""

# Language → emoji flag mapping (best-effort)
_LANG_FLAGS: dict[str, str] = {
    "English":    "🇬🇧",
    "Spanish":    "🇪🇸",
    "French":     "🇫🇷",
    "German":     "🇩🇪",
    "Hindi":      "🇮🇳",
    "Arabic":     "🇸🇦",
    "Portuguese": "🇧🇷",
    "Japanese":   "🇯🇵",
    "Chinese":    "🇨🇳",
    "Korean":     "🇰🇷",
    "Italian":    "🇮🇹",
    "Russian":    "🇷🇺",
    "Dutch":      "🇳🇱",
    "Turkish":    "🇹🇷",
    "Polish":     "🇵🇱",
    "Swedish":    "🇸🇪",
}


def _lang_flag(lang: str) -> str:
    """Return flag emoji for a language name, or 🌐 as fallback."""
    return _LANG_FLAGS.get(lang, "🌐")


def _speed_bars_html(wpm: int, lo: int = 80, hi: int = 300) -> str:
    """Return animated bar-visualiser HTML for speech speed."""
    n_bars   = 12
    fraction = max(0.0, min(1.0, (wpm - lo) / (hi - lo)))
    active   = max(1, round(fraction * n_bars))
    bars     = ""
    for i in range(n_bars):
        height = 40 + 55 * (i / (n_bars - 1))   # 40 % … 95 %
        if i < active:
            bg = f"linear-gradient(180deg,#83A6CE,rgba(131,166,206,0.4))"
            op = "1"
        else:
            bg = "rgba(131,166,206,0.12)"
            op = "0.35"
        bars += (
            f'<div class="speed-bar" style="height:{height:.0f}%;'
            f'background:{bg};opacity:{op};"></div>'
        )
    return (
        f'<div style="margin-bottom:6px;">'
        f'  <div style="font-size:0.72rem;color:#64748b;font-weight:600;'
        f'              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
        f'    Speed visualiser</div>'
        f'  <div class="speed-bar-row">{bars}</div>'
        f'  <div style="display:flex;justify-content:space-between;'
        f'              font-size:0.68rem;color:#475569;margin-top:4px;">'
        f'    <span>Slow (80 wpm)</span><span style="color:#83A6CE;font-weight:700;">'
        f'{wpm} wpm</span><span>Fast (300 wpm)</span>'
        f'  </div>'
        f'</div>'
    )


# =============================================================================
# LOCAL  (app.py)  —  4-tab layout, uses TTSEngine, 'target_language' key
# =============================================================================

def show_settings_local(tts) -> None:
    """
    Render Settings page for the local app (app.py).

    Uses session-state keys: theme, large_font, high_contrast, subtitles_on,
    camera_index, detection_threshold, auto_speak, speech_rate, voice_index,
    target_language, char_conf_threshold, char_stable_frames_cfg.
    """
    # Inject premium CSS
    st.markdown(_SETTINGS_CSS, unsafe_allow_html=True)

    # ── Top Bar with Return to Home ───────────────────────────────────────────
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Return to Home", key="settings_local_back_home", type="secondary"):
            st.session_state["page"] = "🏠 Home"
            st.rerun()

    # ── Premium hero header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="settings-hero">
      <div class="settings-hero-inner">
        <div class="settings-hero-title">⚙️ Settings</div>
        <div class="settings-hero-sub">Customise the app to your preferences</div>
        <div class="settings-hero-pills">
          <span class="settings-hero-pill">🎨 Interface</span>
          <span class="settings-hero-pill blue">📷 Camera</span>
          <span class="settings-hero-pill">🔊 Speech</span>
          <span class="settings-hero-pill blue">🌍 Translation</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_ui, tab_cam, tab_tts, tab_trans = st.tabs(
        ["🎨 Interface", "📷 Camera", "🔊 Speech", "🌍 Translation"]
    )

    # ── Interface tab ──────────────────────────────────────────────────────────
    with tab_ui:
        current_theme = st.session_state["theme"]
        dark_active   = "active-swatch" if current_theme == "dark"  else ""
        light_active  = "active-swatch" if current_theme == "light" else ""

        st.markdown(f"""
        <div class="settings-card">
          <div class="settings-card-title">🎨 Theme &amp; Accessibility</div>
          <div class="theme-preview-row">
            <div class="theme-swatch dark-swatch {dark_active}">
              <div style="font-size:1.5rem;margin-bottom:4px;">🌙</div>
              Dark Mode
            </div>
            <div class="theme-swatch light-swatch {light_active}">
              <div style="font-size:1.5rem;margin-bottom:4px;">☀️</div>
              Light Mode
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox(
                "Theme", ["dark", "light"],
                index=0 if st.session_state["theme"] == "dark" else 1,
            )
            large = st.checkbox("Large Font Mode", value=st.session_state["large_font"])
        with col2:
            hc   = st.checkbox("High Contrast Mode", value=st.session_state["high_contrast"])
            subs = st.checkbox("Show Live Subtitles", value=st.session_state["subtitles_on"])
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Interface Settings", type="primary", key="apply_ui"):
            st.session_state["theme"]         = theme
            st.session_state["large_font"]    = large
            st.session_state["high_contrast"] = hc
            st.session_state["subtitles_on"]  = subs
            st.success("✅ Interface settings applied.")
            st.rerun()

    # ── Camera tab ─────────────────────────────────────────────────────────────
    with tab_cam:
        cam_idx   = st.session_state["camera_index"]
        det_thresh = float(st.session_state["detection_threshold"])

        st.markdown(f"""
        <div class="settings-card">
          <div class="settings-card-title">📷 Camera Configuration</div>
          <div class="camera-diagram">
            <span class="cam-icon">📷</span>
            <div class="camera-diagram-arrow">→</div>
            <div style="text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:4px;">✋</div>
              <div>Hand Detection</div>
            </div>
            <div class="camera-diagram-arrow">→</div>
            <div style="text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:4px;">🧠</div>
              <div>AI Classifier</div>
            </div>
            <div class="camera-diagram-arrow">→</div>
            <div style="text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:4px;">💬</div>
              <div>Translation</div>
            </div>
          </div>
          <div style="display:flex;gap:10px;margin-top:4px;">
            <div class="stat-card" style="flex:1;text-align:center;">
              <div class="stat-label">Device Index</div>
              <div class="stat-value" style="color:#83A6CE;">{cam_idx}</div>
            </div>
            <div class="stat-card" style="flex:1;text-align:center;">
              <div class="stat-label">Conf. Threshold</div>
              <div class="stat-value" style="color:#C48CB3;">{det_thresh:.0%}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        cam_idx_new = st.number_input(
            "Camera Device Index", min_value=0, max_value=5,
            value=st.session_state["camera_index"], step=1,
        )
        det_thresh_new = st.slider(
            "Detection Confidence Threshold", 0.3, 0.99,
            value=float(st.session_state["detection_threshold"]), step=0.01,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Camera Settings", type="primary", key="apply_cam"):
            st.session_state["camera_index"]        = int(cam_idx_new)
            st.session_state["detection_threshold"] = det_thresh_new
            st.success("✅ Camera settings applied.")

    # ── Speech tab ─────────────────────────────────────────────────────────────
    with tab_tts:
        current_rate = st.session_state["speech_rate"]

        st.markdown(
            f'<div class="settings-card">'
            f'<div class="settings-card-title">🔊 Text-to-Speech</div>'
            f'{_speed_bars_html(current_rate)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        auto_spk = st.checkbox(
            "Auto-Speak on Gesture Recognition",
            value=st.session_state["auto_speak"],
        )
        spd = st.slider(
            "Speech Speed (wpm)", 80, 300,
            value=st.session_state["speech_rate"], step=10,
        )
        voice_names = tts.get_voices() if tts and tts.is_available else []
        if voice_names:
            voice_idx_val = st.selectbox(
                "Voice", voice_names,
                index=min(st.session_state["voice_index"], len(voice_names) - 1),
            )
            selected_voice_idx = voice_names.index(voice_idx_val)
        else:
            st.info("No system voices detected.")
            selected_voice_idx = 0
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Speech Settings", type="primary", key="apply_tts"):
            st.session_state["auto_speak"]  = auto_spk
            st.session_state["speech_rate"] = spd
            st.session_state["voice_index"] = selected_voice_idx
            if tts and tts.is_available:
                tts.set_voice_by_index(selected_voice_idx)
            st.success("✅ Speech settings applied.")

    # ── Translation tab ────────────────────────────────────────────────────────
    with tab_trans:
        cur_lang = st.session_state["target_language"]
        flag     = _lang_flag(cur_lang)

        st.markdown(f"""
        <div class="settings-card">
          <div class="settings-card-title">🌍 Translation</div>
          <div style="display:flex;align-items:center;gap:14px;
                      background:rgba(131,166,206,0.06);
                      border:1px solid rgba(131,166,206,0.15);
                      border-radius:12px;padding:14px 16px;margin-bottom:12px;">
            <span style="font-size:2.4rem;">{flag}</span>
            <div>
              <div style="font-size:0.72rem;color:#64748b;font-weight:700;
                           text-transform:uppercase;letter-spacing:0.06em;">
                Current Language</div>
              <div style="font-size:1.25rem;font-weight:800;color:#83A6CE;margin-top:2px;">
                {cur_lang}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        lang_list = get_language_list()
        cur_idx   = lang_list.index(st.session_state["target_language"]) \
                    if st.session_state["target_language"] in lang_list else 0
        lang = st.selectbox("Target Language", lang_list, index=cur_idx)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Translation Settings", type="primary", key="apply_trans"):
            st.session_state["target_language"] = lang
            st.success(f"✅ Language set to {lang}.")


# =============================================================================
# CLOUD  (app_cloud.py)  —  4-tab layout identical to show_settings_local,
#                           but uses cloud session-state keys and Web Speech API
# =============================================================================

def show_settings_cloud() -> None:
    """
    Render Settings page for the cloud app (app_cloud.py).
    Same 4-tab layout as show_settings_local — Interface / Camera / Speech / Translation.
    Uses session-state keys: language, font_size, high_contrast, auto_speak,
    subtitles, conf_threshold, stable_frames_cfg, char_conf_threshold,
    char_stable_frames_cfg, theme.
    """
    from utils.ui_components import inject_css

    st.markdown(_SETTINGS_CSS, unsafe_allow_html=True)

    # ── Top Bar ───────────────────────────────────────────────────────────────
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Return to Home", key="settings_cloud_back_home", type="secondary"):
            st.session_state["page"] = "🏠 Home"
            st.rerun()

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="settings-hero">
      <div class="settings-hero-inner">
        <div class="settings-hero-title">⚙️ Settings</div>
        <div class="settings-hero-sub">Customise the app to your preferences</div>
        <div class="settings-hero-pills">
          <span class="settings-hero-pill">🎨 Interface</span>
          <span class="settings-hero-pill blue">📷 Camera</span>
          <span class="settings-hero-pill">🔊 Speech</span>
          <span class="settings-hero-pill blue">🌍 Translation</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_ui, tab_cam, tab_tts, tab_trans = st.tabs(
        ["🎨 Interface", "📷 Camera", "🔊 Speech", "🌍 Translation"]
    )

    # ── Interface tab ─────────────────────────────────────────────────────────
    with tab_ui:
        current_theme = st.session_state["theme"]
        dark_active   = "active-swatch" if current_theme == "dark"  else ""
        light_active  = "active-swatch" if current_theme == "light" else ""

        st.markdown(f"""
        <div class="settings-card">
          <div class="settings-card-title">🎨 Theme &amp; Accessibility</div>
          <div class="theme-preview-row">
            <div class="theme-swatch dark-swatch {dark_active}">
              <div style="font-size:1.5rem;margin-bottom:4px;">🌙</div>
              Dark Mode
            </div>
            <div class="theme-swatch light-swatch {light_active}">
              <div style="font-size:1.5rem;margin-bottom:4px;">☀️</div>
              Light Mode
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox(
                "Theme", ["dark", "light"],
                index=0 if st.session_state["theme"] == "dark" else 1,
                key="cloud_theme_sel",
            )
            large = st.checkbox(
                "Large Font Mode",
                value=(st.session_state["font_size"] == "large"),
                key="cloud_large_font",
            )
        with col2:
            hc   = st.checkbox("High Contrast Mode",
                               value=st.session_state["high_contrast"], key="cloud_hc")
            subs = st.checkbox("Show Live Subtitles",
                               value=st.session_state["subtitles"], key="cloud_subs")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Interface Settings", type="primary", key="cloud_apply_ui"):
            st.session_state["theme"]        = theme
            st.session_state["font_size"]    = "large" if large else "normal"
            st.session_state["high_contrast"] = hc
            st.session_state["subtitles"]    = subs
            st.session_state["subtitles_on"] = subs
            inject_css(
                theme=theme,
                large_font=large,
                high_contrast=hc,
            )
            st.success("✅ Interface settings applied.")
            st.rerun()

    # ── Camera tab ────────────────────────────────────────────────────────────
    with tab_cam:
        det_thresh = float(st.session_state["conf_threshold"])

        st.markdown(f"""
        <div class="settings-card">
          <div class="settings-card-title">📷 Camera Configuration</div>
          <div class="camera-diagram">
            <span class="cam-icon">📷</span>
            <div class="camera-diagram-arrow">→</div>
            <div style="text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:4px;">✋</div>
              <div>Hand Detection</div>
            </div>
            <div class="camera-diagram-arrow">→</div>
            <div style="text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:4px;">🧠</div>
              <div>AI Classifier</div>
            </div>
            <div class="camera-diagram-arrow">→</div>
            <div style="text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:4px;">💬</div>
              <div>Translation</div>
            </div>
          </div>
          <div style="display:flex;gap:10px;margin-top:4px;">
            <div class="stat-card" style="flex:1;text-align:center;">
              <div class="stat-label">WebRTC</div>
              <div class="stat-value" style="color:#83A6CE;">Browser</div>
            </div>
            <div class="stat-card" style="flex:1;text-align:center;">
              <div class="stat-label">Conf. Threshold</div>
              <div class="stat-value" style="color:#C48CB3;">{det_thresh:.0%}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        new_conf = st.slider(
            "Detection Confidence Threshold", 0.30, 0.99,
            value=float(st.session_state["conf_threshold"]), step=0.01,
            key="cloud_det_conf",
        )
        new_stab = st.slider(
            "Stability Frames (before word is added)", 5, 40,
            value=int(st.session_state["stable_frames_cfg"]), step=1,
            key="cloud_stab",
        )
        new_char_conf = st.slider(
            "Character Confidence Threshold", 0.30, 0.99,
            value=float(st.session_state["char_conf_threshold"]), step=0.01,
            key="cloud_char_conf",
        )
        new_char_stab = st.slider(
            "Character Stability Frames", 5, 50,
            value=int(st.session_state["char_stable_frames_cfg"]), step=1,
            key="cloud_char_stab",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Camera Settings", type="primary", key="cloud_apply_cam"):
            st.session_state["conf_threshold"]         = new_conf
            st.session_state["stable_frames_cfg"]      = new_stab
            st.session_state["char_conf_threshold"]    = new_char_conf
            st.session_state["char_stable_frames_cfg"] = new_char_stab
            st.success("✅ Camera settings applied.")

    # ── Speech tab ────────────────────────────────────────────────────────────
    with tab_tts:
        current_rate = st.session_state.get("speech_rate", 150)

        st.markdown(
            f'<div class="settings-card">'
            f'<div class="settings-card-title">🔊 Text-to-Speech (Web Speech API)</div>'
            f'{_speed_bars_html(current_rate)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        auto_spk = st.checkbox(
            "Auto-Speak on Gesture Recognition",
            value=st.session_state["auto_speak"],
            key="cloud_auto_spk",
        )
        spd = st.slider(
            "Speech Speed (wpm)", 80, 300,
            value=current_rate, step=10,
            key="cloud_spd",
        )
        st.info("Voice selection uses your browser's built-in Web Speech API voices.", icon="ℹ️")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Speech Settings", type="primary", key="cloud_apply_tts"):
            st.session_state["auto_speak"]  = auto_spk
            st.session_state["speech_rate"] = spd
            st.success("✅ Speech settings applied.")

    # ── Translation tab ───────────────────────────────────────────────────────
    with tab_trans:
        cur_lang = st.session_state["language"]
        flag     = _lang_flag(cur_lang)

        st.markdown(f"""
        <div class="settings-card">
          <div class="settings-card-title">🌍 Translation</div>
          <div style="display:flex;align-items:center;gap:14px;
                      background:rgba(131,166,206,0.06);
                      border:1px solid rgba(131,166,206,0.15);
                      border-radius:12px;padding:14px 16px;margin-bottom:12px;">
            <span style="font-size:2.4rem;">{flag}</span>
            <div>
              <div style="font-size:0.72rem;color:#64748b;font-weight:700;
                           text-transform:uppercase;letter-spacing:0.06em;">Current Language</div>
              <div style="font-size:1.25rem;font-weight:800;color:#83A6CE;margin-top:2px;">
                {cur_lang}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        lang_options = list(SUPPORTED_LANGUAGES.keys())
        lang_idx     = lang_options.index(cur_lang) if cur_lang in lang_options else 0
        lang = st.selectbox("Target Language", lang_options, index=lang_idx,
                            key="cloud_lang_sel")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Apply Translation Settings", type="primary", key="cloud_apply_trans"):
            st.session_state["language"]         = lang
            st.session_state["target_language"]  = lang
            st.success(f"✅ Language set to {lang}.")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        if st.button("↩️ Reset to Defaults", use_container_width=True, key="cloud_reset"):
            st.session_state["language"]               = DEFAULT_LANGUAGE
            st.session_state["target_language"]        = DEFAULT_LANGUAGE
            st.session_state["font_size"]              = DEFAULT_FONT_SIZE
            st.session_state["high_contrast"]          = DEFAULT_HIGH_CONTRAST
            st.session_state["auto_speak"]             = AUTO_SPEAK_DEFAULT
            st.session_state["subtitles"]              = SUBTITLE_ENABLED_DEFAULT
            st.session_state["subtitles_on"]           = SUBTITLE_ENABLED_DEFAULT
            st.session_state["conf_threshold"]         = CONFIDENCE_THRESHOLD
            st.session_state["stable_frames_cfg"]      = STABLE_FRAME_COUNT
            st.session_state["char_conf_threshold"]    = CHARACTER_CONFIDENCE_THRESHOLD
            st.session_state["char_stable_frames_cfg"] = CHARACTER_STABLE_FRAME_COUNT
            st.success("Settings reset to defaults.")
            st.rerun()
