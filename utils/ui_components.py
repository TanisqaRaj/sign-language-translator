# =============================================================================
# utils/ui_components.py  –  Claymorphism Design System
# =============================================================================

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Claymorphism CSS
# ─────────────────────────────────────────────────────────────────────────────

CLAY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a1040 0%, #080D1C 45%, #0a0f24 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1329 0%, #111827 100%);
    border-right: 2px solid rgba(99,102,241,0.18);
    box-shadow: 4px 0 32px rgba(0,0,0,0.4);
}

/* ── Clay card – the core building block ── */
.clay-card {
    background: linear-gradient(145deg, #1e2a4a, #182040);
    border-radius: 24px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow:
        8px 8px 20px rgba(0,0,0,0.55),
        -3px -3px 10px rgba(99,102,241,0.08),
        inset 0 1px 0 rgba(255,255,255,0.07),
        inset 0 -1px 0 rgba(0,0,0,0.3);
    border: 1px solid rgba(99,102,241,0.15);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.clay-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
}
.clay-card:hover {
    transform: translateY(-2px);
    box-shadow:
        10px 12px 24px rgba(0,0,0,0.6),
        -3px -3px 10px rgba(99,102,241,0.1),
        inset 0 1px 0 rgba(255,255,255,0.09),
        inset 0 -1px 0 rgba(0,0,0,0.3);
}

/* ── Clay card variants ── */
.clay-card-blue {
    background: linear-gradient(145deg, #1a2c5a, #142042);
    border-color: rgba(59,130,246,0.25);
    box-shadow:
        8px 8px 20px rgba(0,0,0,0.55),
        -3px -3px 10px rgba(59,130,246,0.08),
        inset 0 1px 0 rgba(255,255,255,0.07),
        inset 0 -1px 0 rgba(0,0,0,0.3);
}
.clay-card-violet {
    background: linear-gradient(145deg, #221a4a, #1a1238);
    border-color: rgba(139,92,246,0.25);
    box-shadow:
        8px 8px 20px rgba(0,0,0,0.55),
        -3px -3px 10px rgba(139,92,246,0.1),
        inset 0 1px 0 rgba(255,255,255,0.07),
        inset 0 -1px 0 rgba(0,0,0,0.3);
}
.clay-card-cyan {
    background: linear-gradient(145deg, #0d2a38, #0a1f2c);
    border-color: rgba(6,182,212,0.25);
    box-shadow:
        8px 8px 20px rgba(0,0,0,0.55),
        -3px -3px 10px rgba(6,182,212,0.08),
        inset 0 1px 0 rgba(255,255,255,0.07),
        inset 0 -1px 0 rgba(0,0,0,0.3);
}

/* ── AI panel ── */
.ai-panel {
    background: linear-gradient(145deg, #1e2a4a, #182040);
    border-radius: 28px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow:
        10px 10px 24px rgba(0,0,0,0.6),
        -4px -4px 12px rgba(139,92,246,0.1),
        inset 0 2px 0 rgba(255,255,255,0.08),
        inset 0 -2px 0 rgba(0,0,0,0.35);
    border: 1px solid rgba(139,92,246,0.2);
    position: relative;
    overflow: hidden;
}
.ai-panel::after {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.ai-panel h3 { color: #a78bfa; margin: 0 0 14px 0; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }

/* ── Gesture display ── */
.gesture-big {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 1px;
    line-height: 1.1;
    filter: drop-shadow(0 0 12px rgba(139,92,246,0.35));
}
.gesture-meaning {
    font-size: 0.88rem;
    color: #94a3b8;
    text-align: center;
    margin-top: 6px;
    font-weight: 500;
}

/* ── Clay buttons ── */
.stButton > button {
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.18s ease !important;
    box-shadow:
        0 6px 14px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.12),
        inset 0 -2px 0 rgba(0,0,0,0.25) !important;
    border: none !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow:
        0 10px 22px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.15),
        inset 0 -2px 0 rgba(0,0,0,0.25) !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow:
        0 2px 6px rgba(0,0,0,0.4),
        inset 0 2px 4px rgba(0,0,0,0.3) !important;
}

/* ── Status badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 13px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
}
.badge-green  { background: linear-gradient(135deg,#064e3b,#065f46); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.badge-blue   { background: linear-gradient(135deg,#1e3a5f,#1e40af); color: #93c5fd; border: 1px solid rgba(147,197,253,0.25); }
.badge-yellow { background: linear-gradient(135deg,#451a03,#78350f); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.badge-red    { background: linear-gradient(135deg,#450a0a,#7f1d1d); color: #fca5a5; border: 1px solid rgba(252,165,165,0.25); }
.badge-violet { background: linear-gradient(135deg,#2e1065,#4c1d95); color: #c4b5fd; border: 1px solid rgba(196,181,253,0.25); }

/* ── Camera wrapper ── */
.camera-wrapper {
    border-radius: 24px;
    overflow: hidden;
    box-shadow:
        0 0 0 2px rgba(99,102,241,0.3),
        0 12px 40px rgba(0,0,0,0.6),
        inset 0 0 0 1px rgba(255,255,255,0.05);
    position: relative;
}

/* ── Translation cards ── */
.trans-card {
    background: linear-gradient(145deg, #1a2240, #141a34);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 16px;
    padding: 13px 17px;
    margin: 7px 0;
    box-shadow:
        4px 4px 12px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.06);
    transition: all 0.2s ease;
}
.trans-card:hover {
    border-color: rgba(139,92,246,0.35);
    transform: translateX(3px);
    box-shadow: 6px 4px 16px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
}
.trans-lang  { font-size: 0.7rem; color: #818cf8; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; }
.trans-text  { font-size: 1.15rem; color: #e2e8f0; font-weight: 600; margin-top: 5px; }

/* ── Sentence box ── */
.sentence-box {
    background: linear-gradient(145deg, #1a2240, #141a34);
    border: 2px solid rgba(99,102,241,0.22);
    border-radius: 20px;
    padding: 18px 22px;
    font-size: 1.2rem;
    color: #e2e8f0;
    min-height: 62px;
    word-break: break-word;
    line-height: 1.65;
    font-weight: 500;
    box-shadow:
        inset 4px 4px 12px rgba(0,0,0,0.35),
        inset -2px -2px 8px rgba(99,102,241,0.06),
        0 2px 8px rgba(0,0,0,0.3);
    letter-spacing: 0.2px;
}

/* ── Subtitle strip ── */
.subtitle-strip {
    background: linear-gradient(90deg, rgba(99,102,241,0.18), rgba(139,92,246,0.18));
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 0 0 20px 20px;
    padding: 11px 22px;
    font-size: 1.05rem;
    color: #e2e8f0;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

/* ── Confidence bar track ── */
.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
}
.conf-fill-high   { background: linear-gradient(90deg, #34d399, #10b981); border-radius: 8px; height: 100%; }
.conf-fill-medium { background: linear-gradient(90deg, #fbbf24, #f59e0b); border-radius: 8px; height: 100%; }
.conf-fill-low    { background: linear-gradient(90deg, #f87171, #ef4444); border-radius: 8px; height: 100%; }

/* ── Stat grid ── */
.stat-card {
    background: linear-gradient(145deg, #1a2240, #141a34);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 18px;
    padding: 14px 16px;
    text-align: center;
    box-shadow:
        5px 5px 14px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.07);
}
.stat-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.stat-value { font-size: 1.3rem; font-weight: 800; color: #818cf8; margin-top: 4px; }

/* ── Gesture badge chips ── */
.gesture-chip {
    display: inline-block;
    background: linear-gradient(135deg, #1e2a4a, #182040);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 50px;
    padding: 5px 13px;
    margin: 4px;
    font-size: 0.82rem;
    color: #c7d2fe;
    font-weight: 600;
    box-shadow: 2px 3px 8px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.07);
    transition: all 0.18s;
    cursor: default;
}
.gesture-chip:hover {
    background: linear-gradient(135deg, #2d3a6a, #232855);
    border-color: rgba(139,92,246,0.4);
    transform: translateY(-1px);
    box-shadow: 3px 5px 12px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.1);
}

/* ── How-to steps ── */
.step-item {
    background: linear-gradient(145deg, #1a2240, #141a34);
    border-left: 3px solid #6366f1;
    border-radius: 0 14px 14px 0;
    padding: 10px 16px;
    margin-bottom: 9px;
    font-size: 0.88rem;
    color: #cbd5e1;
    box-shadow: 3px 4px 10px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
    font-weight: 500;
}
.step-num {
    color: #818cf8;
    font-weight: 800;
    margin-right: 8px;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1a2240, #141a34) !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 18px !important;
    padding: 14px !important;
    box-shadow: 5px 5px 14px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.07) !important;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #818cf8 !important;
    font-weight: 800 !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid rgba(99,102,241,0.12) !important; margin: 16px 0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(145deg, #1a2240, #141a34);
    border-radius: 16px;
    padding: 4px;
    gap: 4px;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.4);
    border: 1px solid rgba(99,102,241,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    font-weight: 700;
    color: #64748b;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #312e81, #1e1b4b) !important;
    color: #a5b4fc !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}

/* ── Selectbox / Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: linear-gradient(145deg, #1a2240, #141a34) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    box-shadow: inset 2px 2px 6px rgba(0,0,0,0.35) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, #818cf8, #6366f1) !important;
    box-shadow: 0 3px 10px rgba(99,102,241,0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    border-radius: 50% !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 10px rgba(99,102,241,0.4) !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080D1C; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#6366f1,#8b5cf6); border-radius: 3px; }

/* ── Nav buttons in sidebar ── */
.sidebar-nav-btn button {
    text-align: left !important;
    justify-content: flex-start !important;
}

/* ── Page title gradient ── */
.clay-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    line-height: 1.15;
    filter: drop-shadow(0 2px 8px rgba(139,92,246,0.25));
}
.clay-subtitle {
    text-align: center;
    color: #475569;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: -4px;
    letter-spacing: 0.3px;
}

/* ── Section headings ── */
.clay-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

/* ── Loading screen ── */
.clay-loading {
    text-align: center;
    padding: 32px 20px;
    font-size: 1.1rem;
    color: #818cf8;
    font-weight: 600;
}

/* ── Error banner ── */
.clay-error {
    background: linear-gradient(145deg, #450a0a, #3b0606);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px;
    padding: 16px 20px;
    color: #fca5a5;
    font-weight: 600;
    box-shadow: 4px 4px 12px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
}
</style>
"""

HIGH_CONTRAST_CSS = """
<style>
.stApp { background: #000000 !important; }
.clay-card, .ai-panel, .trans-card, .stat-card { background: #111 !important; border: 2px solid #ffffff !important; }
.gesture-big { font-size: 3rem !important; background: none !important; -webkit-text-fill-color: #ffff00 !important; }
.sentence-box { background: #111 !important; color: #ffffff !important; border: 2px solid #ffffff !important; font-size: 1.4rem !important; }
.trans-text { color: #ffffff !important; font-size: 1.4rem !important; }
</style>
"""

LARGE_FONT_CSS = """
<style>
html, body, [class*="css"] { font-size: 18px !important; }
.gesture-big { font-size: 3.4rem !important; }
.sentence-box { font-size: 1.5rem !important; }
.trans-text   { font-size: 1.5rem !important; }
p, li, .stMarkdown { font-size: 1.05rem !important; }
</style>
"""



# ─────────────────────────────────────────────────────────────────────────────
# 3D Clay SVG Illustrations
# ─────────────────────────────────────────────────────────────────────────────

def svg_hand_3d() -> str:
    """3D clay-style hand / ILY gesture SVG."""
    return """
<svg width="110" height="120" viewBox="0 0 110 120" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="palmGrad" cx="45%" cy="55%" r="55%">
      <stop offset="0%" stop-color="#818cf8"/>
      <stop offset="60%" stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#3730a3"/>
    </radialGradient>
    <radialGradient id="fingerGrad" cx="40%" cy="30%" r="60%">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="#5b21b6"/>
    </radialGradient>
    <radialGradient id="thumbGrad" cx="35%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#c4b5fd"/>
      <stop offset="100%" stop-color="#6d28d9"/>
    </radialGradient>
    <filter id="handShadow" x="-20%" y="-10%" width="150%" height="150%">
      <feDropShadow dx="4" dy="6" stdDeviation="6" flood-color="#1e1b4b" flood-opacity="0.7"/>
    </filter>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- Shadow -->
  <ellipse cx="55" cy="112" rx="32" ry="7" fill="#1e1b4b" opacity="0.5"/>
  <!-- Palm -->
  <rect x="22" y="54" width="58" height="52" rx="14" fill="url(#palmGrad)" filter="url(#handShadow)"/>
  <!-- Palm highlight -->
  <ellipse cx="42" cy="62" rx="12" ry="6" fill="rgba(255,255,255,0.12)" transform="rotate(-10 42 62)"/>
  <!-- Index finger -->
  <rect x="28" y="18" width="14" height="42" rx="7" fill="url(#fingerGrad)" filter="url(#handShadow)"/>
  <ellipse cx="35" cy="22" rx="5" ry="3" fill="rgba(255,255,255,0.18)"/>
  <!-- Middle finger -->
  <rect x="46" y="12" width="14" height="46" rx="7" fill="url(#fingerGrad)" filter="url(#handShadow)"/>
  <ellipse cx="53" cy="16" rx="5" ry="3" fill="rgba(255,255,255,0.18)"/>
  <!-- Ring finger (folded, stub) -->
  <rect x="64" y="40" width="13" height="22" rx="6" fill="url(#fingerGrad)" opacity="0.85"/>
  <!-- Pinky -->
  <rect x="72" y="28" width="11" height="34" rx="5" fill="url(#fingerGrad)" filter="url(#handShadow)"/>
  <ellipse cx="77" cy="32" rx="4" ry="2.5" fill="rgba(255,255,255,0.18)"/>
  <!-- Thumb -->
  <rect x="6" y="52" width="22" height="12" rx="6" fill="url(#thumbGrad)" transform="rotate(-30 6 52)" filter="url(#handShadow)"/>
  <ellipse cx="12" cy="52" rx="4" ry="2.5" fill="rgba(255,255,255,0.2)" transform="rotate(-30 12 52)"/>
  <!-- Knuckle indents -->
  <circle cx="35" cy="58" r="3" fill="rgba(0,0,0,0.15)"/>
  <circle cx="53" cy="56" r="3" fill="rgba(0,0,0,0.15)"/>
  <circle cx="70" cy="58" r="2.5" fill="rgba(0,0,0,0.12)"/>
  <!-- Glow ring -->
  <circle cx="55" cy="72" r="28" stroke="rgba(139,92,246,0.2)" stroke-width="1.5" fill="none" filter="url(#glow)"/>
</svg>"""


def svg_ai_robot() -> str:
    """3D clay-style AI assistant robot SVG."""
    return """
<svg width="90" height="100" viewBox="0 0 90 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="headGrad" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#4c1d95"/>
    </radialGradient>
    <radialGradient id="bodyGrad" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#4f46e5"/>
      <stop offset="100%" stop-color="#312e81"/>
    </radialGradient>
    <radialGradient id="screenGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </radialGradient>
    <filter id="botShadow">
      <feDropShadow dx="3" dy="5" stdDeviation="5" flood-color="#1e1b4b" flood-opacity="0.7"/>
    </filter>
  </defs>
  <!-- Shadow -->
  <ellipse cx="45" cy="97" rx="26" ry="5" fill="#1e1b4b" opacity="0.45"/>
  <!-- Body -->
  <rect x="16" y="52" width="58" height="40" rx="14" fill="url(#bodyGrad)" filter="url(#botShadow)"/>
  <!-- Body highlight -->
  <ellipse cx="35" cy="60" rx="14" ry="6" fill="rgba(255,255,255,0.1)" transform="rotate(-8 35 60)"/>
  <!-- Screen on body -->
  <rect x="24" y="60" width="42" height="24" rx="8" fill="url(#screenGrad)" opacity="0.9"/>
  <rect x="24" y="60" width="42" height="24" rx="8" fill="url(#screenGrad)"/>
  <!-- Screen shine -->
  <rect x="26" y="62" width="16" height="5" rx="3" fill="rgba(255,255,255,0.25)"/>
  <!-- Screen content dots -->
  <circle cx="38" cy="74" r="3" fill="#7fffd4"/>
  <circle cx="48" cy="74" r="3" fill="#fde68a"/>
  <circle cx="58" cy="74" r="3" fill="#c4b5fd"/>
  <!-- Head -->
  <rect x="18" y="10" width="54" height="46" rx="18" fill="url(#headGrad)" filter="url(#botShadow)"/>
  <!-- Head highlight -->
  <ellipse cx="36" cy="18" rx="14" ry="7" fill="rgba(255,255,255,0.13)" transform="rotate(-10 36 18)"/>
  <!-- Left eye -->
  <circle cx="32" cy="34" r="9" fill="#1e1b4b"/>
  <circle cx="32" cy="34" r="6" fill="#06b6d4"/>
  <circle cx="32" cy="34" r="3.5" fill="#e0f2fe"/>
  <circle cx="30" cy="32" r="1.2" fill="rgba(255,255,255,0.7)"/>
  <!-- Right eye -->
  <circle cx="58" cy="34" r="9" fill="#1e1b4b"/>
  <circle cx="58" cy="34" r="6" fill="#06b6d4"/>
  <circle cx="58" cy="34" r="3.5" fill="#e0f2fe"/>
  <circle cx="56" cy="32" r="1.2" fill="rgba(255,255,255,0.7)"/>
  <!-- Antenna -->
  <rect x="42" y="2" width="6" height="12" rx="3" fill="#a78bfa"/>
  <circle cx="45" cy="2" r="5" fill="#c4b5fd"/>
  <circle cx="45" cy="2" r="3" fill="#ddd6fe"/>
  <!-- Ears -->
  <rect x="4" y="22" width="14" height="22" rx="7" fill="url(#headGrad)" opacity="0.85"/>
  <rect x="72" y="22" width="14" height="22" rx="7" fill="url(#headGrad)" opacity="0.85"/>
  <!-- Arms -->
  <rect x="2" y="56" width="14" height="28" rx="7" fill="url(#bodyGrad)" opacity="0.8"/>
  <rect x="74" y="56" width="14" height="28" rx="7" fill="url(#bodyGrad)" opacity="0.8"/>
</svg>"""


def svg_microphone() -> str:
    """3D clay-style microphone SVG."""
    return """
<svg width="64" height="80" viewBox="0 0 64 80" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="micGrad" cx="40%" cy="30%" r="60%">
      <stop offset="0%" stop-color="#ec4899"/>
      <stop offset="100%" stop-color="#9d174d"/>
    </radialGradient>
    <radialGradient id="baseGrad" cx="40%" cy="35%" r="55%">
      <stop offset="0%" stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#312e81"/>
    </radialGradient>
    <filter id="micShadow">
      <feDropShadow dx="3" dy="4" stdDeviation="4" flood-color="#1e1b4b" flood-opacity="0.65"/>
    </filter>
  </defs>
  <ellipse cx="32" cy="77" rx="18" ry="4" fill="#1e1b4b" opacity="0.4"/>
  <!-- Stand base -->
  <rect x="14" y="68" width="36" height="9" rx="5" fill="url(#baseGrad)" filter="url(#micShadow)"/>
  <ellipse cx="25" cy="70" rx="8" ry="3" fill="rgba(255,255,255,0.1)"/>
  <!-- Stand pole -->
  <rect x="28" y="46" width="8" height="24" rx="4" fill="url(#baseGrad)" opacity="0.9"/>
  <!-- Mic body -->
  <rect x="14" y="6" width="36" height="46" rx="18" fill="url(#micGrad)" filter="url(#micShadow)"/>
  <!-- Mic highlight -->
  <ellipse cx="26" cy="14" rx="8" ry="12" fill="rgba(255,255,255,0.15)" transform="rotate(-10 26 14)"/>
  <!-- Grille lines -->
  <line x1="20" y1="28" x2="44" y2="28" stroke="rgba(0,0,0,0.2)" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="18" y1="34" x2="46" y2="34" stroke="rgba(0,0,0,0.2)" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="20" y1="40" x2="44" y2="40" stroke="rgba(0,0,0,0.2)" stroke-width="1.5" stroke-linecap="round"/>
  <!-- Top shine -->
  <ellipse cx="32" cy="11" rx="8" ry="5" fill="rgba(255,255,255,0.18)"/>
</svg>"""


def svg_speaker() -> str:
    """3D clay-style speaker SVG."""
    return """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="spkGrad" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#0e7490"/>
    </radialGradient>
    <filter id="spkShadow">
      <feDropShadow dx="3" dy="4" stdDeviation="4" flood-color="#1e1b4b" flood-opacity="0.6"/>
    </filter>
  </defs>
  <ellipse cx="32" cy="61" rx="22" ry="5" fill="#1e1b4b" opacity="0.4"/>
  <!-- Speaker body -->
  <rect x="6" y="8" width="52" height="50" rx="16" fill="url(#spkGrad)" filter="url(#spkShadow)"/>
  <!-- Highlight -->
  <ellipse cx="22" cy="16" rx="12" ry="7" fill="rgba(255,255,255,0.14)" transform="rotate(-10 22 16)"/>
  <!-- Speaker cone outer -->
  <circle cx="32" cy="34" r="16" fill="#0c4a6e"/>
  <!-- Speaker cone mid -->
  <circle cx="32" cy="34" r="10" fill="#0369a1"/>
  <!-- Speaker cone inner -->
  <circle cx="32" cy="34" r="5" fill="#06b6d4"/>
  <!-- Dust cap -->
  <circle cx="32" cy="34" r="2.5" fill="#67e8f9"/>
  <!-- Cone highlight -->
  <circle cx="27" cy="29" r="3" fill="rgba(255,255,255,0.12)"/>
  <!-- Corner screws -->
  <circle cx="16" cy="17" r="3" fill="rgba(0,0,0,0.25)"/>
  <circle cx="48" cy="17" r="3" fill="rgba(0,0,0,0.25)"/>
  <circle cx="16" cy="51" r="3" fill="rgba(0,0,0,0.25)"/>
  <circle cx="48" cy="51" r="3" fill="rgba(0,0,0,0.25)"/>
</svg>"""


def svg_analytics() -> str:
    """3D clay-style analytics / chart SVG."""
    return """
<svg width="80" height="70" viewBox="0 0 80 70" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bar1" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#818cf8"/>
      <stop offset="100%" stop-color="#4f46e5"/>
    </linearGradient>
    <linearGradient id="bar2" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="#6d28d9"/>
    </linearGradient>
    <linearGradient id="bar3" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#60a5fa"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="bar4" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>
    <filter id="barShadow">
      <feDropShadow dx="2" dy="3" stdDeviation="3" flood-color="#1e1b4b" flood-opacity="0.6"/>
    </filter>
  </defs>
  <!-- Base platform -->
  <rect x="4" y="54" width="72" height="10" rx="5" fill="#1e2040" opacity="0.8"/>
  <!-- Bar 1 -->
  <rect x="10" y="26" width="12" height="30" rx="6" fill="url(#bar1)" filter="url(#barShadow)"/>
  <ellipse cx="16" cy="29" rx="4" ry="2" fill="rgba(255,255,255,0.2)"/>
  <!-- Bar 2 -->
  <rect x="26" y="14" width="12" height="42" rx="6" fill="url(#bar2)" filter="url(#barShadow)"/>
  <ellipse cx="32" cy="17" rx="4" ry="2" fill="rgba(255,255,255,0.2)"/>
  <!-- Bar 3 -->
  <rect x="42" y="34" width="12" height="22" rx="6" fill="url(#bar3)" filter="url(#barShadow)"/>
  <ellipse cx="48" cy="37" rx="4" ry="2" fill="rgba(255,255,255,0.2)"/>
  <!-- Bar 4 -->
  <rect x="58" y="20" width="12" height="36" rx="6" fill="url(#bar4)" filter="url(#barShadow)"/>
  <ellipse cx="64" cy="23" rx="4" ry="2" fill="rgba(255,255,255,0.2)"/>
  <!-- Floating sphere decoration -->
  <circle cx="70" cy="10" r="8" fill="url(#bar2)" opacity="0.7" filter="url(#barShadow)"/>
  <circle cx="67" cy="7" r="2.5" fill="rgba(255,255,255,0.25)"/>
</svg>"""



# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_css(theme: str = "dark", large_font: bool = False, high_contrast: bool = False) -> None:
    st.markdown(CLAY_CSS, unsafe_allow_html=True)
    if high_contrast:
        st.markdown(HIGH_CONTRAST_CSS, unsafe_allow_html=True)
    if large_font:
        st.markdown(LARGE_FONT_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Component helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_glass_card(content_html: str, variant: str = "") -> None:
    """Render content inside a clay card. variant: '' | 'blue' | 'violet' | 'cyan'"""
    cls = f"clay-card clay-card-{variant}" if variant else "clay-card"
    st.markdown(f'<div class="{cls}">{content_html}</div>', unsafe_allow_html=True)


def render_ai_panel(gesture: str, meaning: str, status: str, confidence: float, speaking: bool) -> None:
    """Render the AI assistant panel with claymorphism styling."""
    speak_badge = '<span class="badge badge-blue">🔊 Speaking…</span>' if speaking else ""
    conf_pct = f"{confidence * 100:.0f}%"
    if confidence >= 0.75:
        conf_color = "#34d399"
        fill_class = "conf-fill-high"
    elif confidence >= 0.5:
        conf_color = "#fbbf24"
        fill_class = "conf-fill-medium"
    else:
        conf_color = "#f87171"
        fill_class = "conf-fill-low"

    bar_w = int(confidence * 100)
    robot_svg = svg_ai_robot()

    html = f"""
    <div class="ai-panel">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
        <div style="flex-shrink:0;">{robot_svg}</div>
        <div style="flex:1;min-width:0;">
          <div style="font-size:0.7rem;color:#64748b;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:6px;">Detected Gesture</div>
          <div class="gesture-big">{gesture}</div>
          <div class="gesture-meaning">{meaning}</div>
        </div>
      </div>
      <div style="background:rgba(0,0,0,0.2);border-radius:12px;padding:12px 14px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
          <span style="color:#64748b;font-size:0.82rem;font-weight:600;">Status</span>
          <span style="color:#c7d2fe;font-size:0.82rem;font-weight:700;">{status}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="color:#64748b;font-size:0.82rem;font-weight:600;">Confidence</span>
          <span style="color:{conf_color};font-size:0.9rem;font-weight:800;">{conf_pct}</span>
        </div>
        <div class="conf-track">
          <div class="{fill_class}" style="width:{bar_w}%;transition:width 0.4s ease;"></div>
        </div>
      </div>
      <div style="margin-top:10px;">{speak_badge}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_confidence_meter(all_probs: dict) -> None:
    """Render per-gesture confidence bars."""
    if not all_probs:
        return
    rows = []
    for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = int(prob * 100)
        if pct >= 75:
            fill_cls = "conf-fill-high"
            val_color = "#34d399"
        elif pct >= 50:
            fill_cls = "conf-fill-medium"
            val_color = "#fbbf24"
        else:
            fill_cls = "conf-fill-low"
            val_color = "#f87171"
        rows.append(f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.83rem;color:#94a3b8;font-weight:600;">{label}</span>
            <span style="font-size:0.83rem;font-weight:800;color:{val_color};">{pct}%</span>
          </div>
          <div class="conf-track">
            <div class="{fill_cls}" style="width:{pct}%;transition:width 0.35s ease;"></div>
          </div>
        </div>""")
    st.markdown(
        '<div class="clay-card"><div class="clay-section-title" style="margin-bottom:14px;">📊 Confidence Meter</div>'
        + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def render_translation_cards(translations: dict) -> None:
    """Render translation result cards in a 2-column grid."""
    if not translations:
        return
    items = list(translations.items())
    html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'
    for lang, text in items:
        html += f"""
        <div class="trans-card">
          <div class="trans-lang">{lang}</div>
          <div class="trans-text">{text}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(text: str, level: str = "green") -> None:
    st.markdown(f'<span class="badge badge-{level}">{text}</span>', unsafe_allow_html=True)


def render_detection_status(has_hand: bool, gesture, confidence: float) -> None:
    if not has_hand:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-yellow">🖐 Scanning for hand…</span></div>'
    elif gesture and confidence >= 0.75:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-green">✅ Gesture Recognised</span></div>'
    elif gesture:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-yellow">🔍 Recognising…</span></div>'
    else:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-blue">🖐 Hand Detected</span></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_suggestions(suggestions: list) -> None:
    if not suggestions:
        return
    chips = " ".join(
        f'<span class="gesture-chip">○ {s}</span>'
        for s in suggestions[1:]
    )
    first = suggestions[0] if suggestions else ""
    st.markdown(
        f'<div class="clay-card"><div style="font-size:0.78rem;color:#64748b;font-weight:600;margin-bottom:10px;">💡 Did you mean?</div>'
        f'<span class="gesture-chip" style="border-color:rgba(52,211,153,0.4);color:#34d399;">✓ {first}</span>{chips}</div>',
        unsafe_allow_html=True,
    )


def render_subtitle_strip(text: str) -> None:
    if text:
        st.markdown(f'<div class="subtitle-strip">📝 {text}</div>', unsafe_allow_html=True)


def render_loading_sequence(stage: str) -> None:
    stages = {
        "model":      "🧠 Loading Model…",
        "camera":     "📷 Initialising Camera…",
        "translator": "🌍 Loading Translator…",
        "ready":      "✅ Ready",
    }
    msg = stages.get(stage, stage)
    color = "#34d399" if stage == "ready" else "#818cf8"
    st.markdown(
        f'<div class="clay-loading" style="color:{color};">{msg}</div>',
        unsafe_allow_html=True,
    )


def camera_placeholder_html() -> str:
    return (
        "<div style='background:linear-gradient(145deg,#1a2240,#141a34);"
        "border:2px dashed rgba(99,102,241,0.3);border-radius:24px;"
        "height:360px;display:flex;align-items:center;justify-content:center;"
        "flex-direction:column;gap:12px;"
        "box-shadow:8px 8px 20px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.05);'>"
        "<span style='font-size:2.5rem;'>📷</span>"
        "<span style='color:#475569;font-weight:600;font-size:1rem;'>Camera feed will appear here</span>"
        "<span style='color:#334155;font-size:0.82rem;'>Click ▶ Start Camera to begin</span>"
        "</div>"
    )
