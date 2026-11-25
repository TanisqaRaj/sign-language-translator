# =============================================================================
# app_pages/about.py  –  About Page
# Shared by both app.py and app_cloud.py.
# =============================================================================

import streamlit as st
import pandas as pd

from utils.ui_components import svg_hand_3d, svg_ai_robot, svg_analytics


def show_about() -> None:
    """Render the premium SignFlow About page."""

    # ── Return to Home ─────────────────────────────────────────────────────────
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Return to Home", key="about_back_home", type="secondary"):
            st.session_state["page"] = "🏠 Home"
            st.rerun()

    # ── Page Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="pro-page-header" style="margin-bottom:32px;">
      <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
        <div style="flex:1;min-width:260px;">
          <div style="font-size:0.72rem;font-weight:700;letter-spacing:2.5px;
                      text-transform:uppercase;color:rgba(131,166,206,0.85);
                      margin-bottom:10px;">About SignFlow</div>
          <h1 style="font-size:2.2rem;font-weight:900;color:#FFFFFF;
                     margin:0 0 12px;line-height:1.15;letter-spacing:-0.3px;">
            Bridging Communication<br>
            <span style="background:linear-gradient(135deg,#83A6CE,#C48CB3);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              Through AI
            </span>
          </h1>
          <p style="color:rgba(255,255,255,0.65);font-size:0.95rem;
                    line-height:1.65;margin:0;max-width:480px;font-weight:400;">
            A final-year computer vision project that translates hand gestures
            into text and speech in real time using MediaPipe, TensorFlow Lite,
            and Streamlit.
          </p>
        </div>
        <div style="display:flex;gap:12px;align-items:center;opacity:0.85;">
          {svg_hand_3d()}
          {svg_ai_robot()}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Overview card ──────────────────────────────────────────────────
    st.markdown("""
    <div class="clay-card">
      <div class="section-title-dark" style="margin-bottom:14px;">Project Overview</div>
      <p style="color:#284155;font-size:0.92rem;line-height:1.75;margin:0;">
        The <strong style="color:#284155;">Real-Time Sign Language Translator</strong>
        captures live webcam footage, detects hand landmarks using
        <strong>Google MediaPipe</strong>, classifies the gesture with a
        <strong>TensorFlow Lite</strong> dense neural network, and reads the
        result aloud — all in real time, directly in the browser.
      </p>
      <p style="color:#6b8aaa;font-size:0.9rem;line-height:1.7;margin:14px 0 0;">
        Two independent models share the same 42-feature MediaPipe pipeline —
        <strong style="color:#83A6CE;">Model 1 (Word)</strong> for common ISL gestures and
        <strong style="color:#C48CB3;">Model 2 (Character)</strong> for the full A–Z, 0–9 alphabet.
        Scalers and label maps are never mixed between models.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture Pipeline ──────────────────────────────────────────────────
    st.markdown("""
    <div class="section-title-dark" style="margin:24px 0 14px;">Architecture Pipeline</div>
    """, unsafe_allow_html=True)

    arch_steps = [
        ("📷", "Capture",        "Webcam / WebRTC"),
        ("🔍", "Detect",         "MediaPipe Hands"),
        ("📐", "Normalise",      "Wrist-origin · ±1"),
        ("⚖️", "Scale",         "StandardScaler"),
        ("🧠", "Classify",       "TFLite Dense NN"),
        ("🗳️", "Smooth",        "Rolling vote"),
        ("📝", "Build",          "Sentence builder"),
        ("🔊", "Speak",          "TTS output"),
    ]
    arch_cols = st.columns(4, gap="small")
    for i, (icon, title, desc) in enumerate(arch_steps):
        with arch_cols[i % 4]:
            accent = "#83A6CE" if i % 2 == 0 else "#C48CB3"
            st.markdown(f"""
            <div class="clay-card" style="text-align:center;padding:18px 12px;margin-bottom:10px;">
              <div style="font-size:1.7rem;margin-bottom:8px;">{icon}</div>
              <div style="font-size:0.88rem;font-weight:700;color:{accent};
                          margin-bottom:4px;">{title}</div>
              <div style="font-size:0.74rem;color:#6b8aaa;line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Performance + Tech Stack (2 columns) ───────────────────────────────────
    col_perf, col_tech = st.columns(2, gap="large")

    with col_perf:
        st.markdown("""
        <div class="section-title-dark" style="margin-bottom:14px;">Performance</div>
        """, unsafe_allow_html=True)
        perf = [
            ("Inference latency",    "~8 ms"),
            ("Landmark extraction",  "~15 ms"),
            ("End-to-end FPS",       "25–35 FPS"),
            ("Word model (TFLite)",  "~0.9 MB"),
            ("Char model (TFLite)",  "~0.9 MB"),
            ("Test accuracy",        "> 95%"),
        ]
        for metric, value in perf:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 16px;margin-bottom:6px;
                        background:#FFFFFF;border-radius:12px;
                        border:1px solid rgba(131,166,206,0.18);
                        box-shadow:0 1px 4px rgba(40,65,85,0.05);">
              <span style="font-size:0.86rem;color:#284155;font-weight:500;">{metric}</span>
              <span style="font-size:0.86rem;color:#83A6CE;font-weight:800;
                           background:rgba(131,166,206,0.1);padding:2px 10px;
                           border-radius:20px;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_tech:
        st.markdown("""
        <div class="section-title-dark" style="margin-bottom:14px;">Technology Stack</div>
        """, unsafe_allow_html=True)
        techs = [
            ("Python 3.10+",          "Core language",             "#284155"),
            ("OpenCV 4.9",            "Camera + frame I/O",        "#83A6CE"),
            ("MediaPipe 0.10",        "Hand landmark detection",   "#C48CB3"),
            ("TensorFlow Lite",       "Gesture classification",    "#83A6CE"),
            ("Streamlit 1.35",        "Web UI",                    "#C48CB3"),
            ("streamlit-webrtc",      "Browser camera access",     "#83A6CE"),
            ("deep-translator",       "13-language translation",   "#C48CB3"),
            ("pyttsx3 / Web Speech",  "Text-to-speech output",     "#83A6CE"),
        ]
        for name, desc, color in techs:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        padding:9px 16px;margin-bottom:6px;
                        background:#FFFFFF;border-radius:12px;
                        border:1px solid rgba(131,166,206,0.18);
                        box-shadow:0 1px 4px rgba(40,65,85,0.05);">
              <div style="width:8px;height:8px;border-radius:50%;
                          background:{color};flex-shrink:0;"></div>
              <div>
                <div style="font-size:0.84rem;font-weight:700;color:#284155;">{name}</div>
                <div style="font-size:0.74rem;color:#6b8aaa;margin-top:1px;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
