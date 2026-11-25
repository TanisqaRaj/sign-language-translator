# =============================================================================
# utils/ui_components.py  –  SignFlow Premium Design System  (v2 – 3D Animated)
# =============================================================================
# Color Palette:
#   Deep Navy    #284155  — primary dark surface, text
#   Powder Blue  #83A6CE  — primary interactive, links, highlights
#   Dusty Mauve  #C48CB3  — accents, active states
#   Pale Pink    #E5C9D7  — soft backgrounds, highlights
#   White        #FFFFFF  — main surfaces
#
# Gradients:
#   Primary:  #83A6CE → #C48CB3
#   Hero bg:  #1a2f3f → #284155 → #1e3347
# =============================================================================

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# HIDE_SIDEBAR_CSS  — completely hides the Streamlit sidebar
# ─────────────────────────────────────────────────────────────────────────────

HIDE_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarNav"]      { display: none !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HOME_PAGE_CSS  — hides sidebar when `.hide-sidebar` class is active,
#                  plus home-page-specific layout helpers
# ─────────────────────────────────────────────────────────────────────────────

HOME_PAGE_CSS = """
<style>
/* ── SignFlow 3D AI Design System ── */
:root {
    --mauve: #C48CB3;
    --blush: #E5C9D7;
    --powder-blue: #83A6CE;
    --deep-navy: #284155;
    --dark-slate: #1a2f3f;
    --light-surface: #F9FAFC;
    --white: #FFFFFF;
}

/* Hide Streamlit sidebar completely on home page */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarNav"]      { display: none !important; }

/* Remove default Streamlit top/side margins for full immersion */
.main .block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #284155;
    background: #FFFFFF;
}

/* ── Navbar inner elements ── */
.nav-logo {
    font-size: 1.05rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.2px;
    white-space: nowrap;
    padding-left: 8px;
    flex-shrink: 0;
}
.nav-links {
    display: flex;
    align-items: center;
    gap: 4px;
    flex: 1;
    justify-content: center;
}
.nav-link {
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 40px;
    cursor: pointer;
    transition: all 0.25s ease;
    white-space: nowrap;
    border: 1px solid transparent;
}
.nav-link:hover {
    color: #FFFFFF;
    background: rgba(196,140,179,0.2);
    border-color: rgba(196,140,179,0.35);
}
.nav-link.active {
    color: #FFFFFF;
    background: linear-gradient(135deg,rgba(131,166,206,0.32),rgba(196,140,179,0.32));
    border-color: rgba(229,201,215,0.45);
    font-weight: 700;
}
.nav-cta {
    background: linear-gradient(135deg,#83A6CE,#C48CB3);
    color: #FFFFFF;
    font-size: 0.86rem;
    font-weight: 700;
    padding: 7px 18px;
    border-radius: 50px;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    box-shadow: 0 4px 18px rgba(131,166,206,0.45);
    transition: all 0.25s ease;
    letter-spacing: 0.2px;
}
.nav-cta:hover {
    background: linear-gradient(135deg,#95b6dc,#cf9bc0);
    box-shadow: 0 8px 24px rgba(196,140,179,0.55);
    transform: translateY(-2px);
}

/* ── Interactive Floating Glass Navbar ── */
.glass-navbar,
div[data-testid="stVerticalBlock"]:has(> div > div > .nav-glass-anchor) {
    position: fixed !important;
    top: 14px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(1200px, 94vw) !important;
    z-index: 99999 !important;
    background: rgba(26, 47, 63, 0.78) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid rgba(229, 201, 215, 0.22) !important;
    border-radius: 50px !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35), 0 0 24px rgba(131, 166, 206, 0.18) !important;
    padding: 6px 14px !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.glass-navbar.scrolled {
    background: rgba(20, 36, 49, 0.92) !important;
    border-color: rgba(131, 166, 206, 0.35) !important;
    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.45), 0 0 30px rgba(196, 140, 179, 0.2) !important;
}

/* Nav Brand Logo */
.nav-brand-logo {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.05rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.2px;
    white-space: nowrap;
    user-select: none;
    padding-left: 8px;
    height: 100%;
}
.nav-brand-logo .brand-title {
    background: linear-gradient(135deg, #FFFFFF 0%, #E5C9D7 45%, #83A6CE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Stylized Streamlit buttons inside Navbar */
div[data-testid="stVerticalBlock"]:has(> div > div > .nav-glass-anchor) button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: rgba(255, 255, 255, 0.85) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border-radius: 40px !important;
    padding: 6px 14px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: none !important;
}

div[data-testid="stVerticalBlock"]:has(> div > div > .nav-glass-anchor) button:hover {
    color: #FFFFFF !important;
    background: rgba(196, 140, 179, 0.2) !important;
    border-color: rgba(196, 140, 179, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Active Nav Button (Home) */
div[data-testid="stVerticalBlock"]:has(> div > div > .nav-glass-anchor) button[kind="primary"]:not(:has(span:contains("Translating"))) {
    background: linear-gradient(135deg, rgba(131, 166, 206, 0.32), rgba(196, 140, 179, 0.32)) !important;
    border: 1px solid rgba(229, 201, 215, 0.45) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(196, 140, 179, 0.25) !important;
}

/* Navbar CTA button */
div[data-testid="stVerticalBlock"]:has(> div > div > .nav-glass-anchor) div[data-testid="column"]:last-child button {
    background: linear-gradient(135deg, #83A6CE, #C48CB3) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 7px 18px !important;
    box-shadow: 0 4px 18px rgba(131, 166, 206, 0.45) !important;
    letter-spacing: 0.2px !important;
}
div[data-testid="stVerticalBlock"]:has(> div > div > .nav-glass-anchor) div[data-testid="column"]:last-child button:hover {
    background: linear-gradient(135deg, #95b6dc, #cf9bc0) !important;
    box-shadow: 0 8px 24px rgba(196, 140, 179, 0.55) !important;
    transform: translateY(-2px) !important;
}

/* ── Hero Section ── */
.home-hero-section {
    background: radial-gradient(circle at 50% 25%, #233b50 0%, #1a2f3f 45%, #142431 100%);
    min-height: 94vh;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 100px 32px 50px;
    border-bottom: 1px solid rgba(131, 166, 206, 0.15);
}

/* Ambient glow blobs */
.hero-bg {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}
.hero-blob-1 {
    position: absolute;
    top: -15%; right: -8%;
    width: 650px; height: 650px;
    background: radial-gradient(circle, rgba(196, 140, 179, 0.22) 0%, rgba(229, 201, 215, 0.08) 45%, transparent 70%);
    border-radius: 50%;
    filter: blur(40px);
    animation: float 9s ease-in-out infinite;
}
.hero-blob-2 {
    position: absolute;
    bottom: -15%; left: -8%;
    width: 550px; height: 550px;
    background: radial-gradient(circle, rgba(131, 166, 206, 0.22) 0%, rgba(40, 65, 85, 0.1) 50%, transparent 70%);
    border-radius: 50%;
    filter: blur(45px);
    animation: float 11s ease-in-out infinite reverse;
}
.hero-blob-3 {
    position: absolute;
    top: 40%; left: 35%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(229, 201, 215, 0.12) 0%, transparent 65%);
    border-radius: 50%;
    filter: blur(35px);
    animation: float 13s ease-in-out infinite 2s;
}

/* Keyframe animations */
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-16px) rotate(2deg); }
}
@keyframes particle-drift {
    0%, 100% { transform: translate(0, 0); opacity: 0.3; }
    50% { transform: translate(12px, -24px); opacity: 0.8; }
}
@keyframes pulse-ring {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.05); opacity: 0.4; }
    100% { transform: scale(0.95); opacity: 0.8; }
}
@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fade-in-up {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* 3D Step Cards */
.step-card-3d {
    background: #FFFFFF;
    border-radius: 24px;
    padding: 34px 24px 28px;
    text-align: center;
    border: 1px solid rgba(131, 166, 206, 0.22);
    box-shadow: 0 10px 30px rgba(40, 65, 85, 0.06), 0 1px 2px rgba(255, 255, 255, 0.9) inset;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
    height: 100%;
}
.step-card-3d::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #83A6CE, #C48CB3, #E5C9D7);
}
.step-card-3d:hover {
    transform: translateY(-8px) scale(1.015);
    box-shadow: 0 24px 48px rgba(40, 65, 85, 0.12), 0 0 20px rgba(131, 166, 206, 0.15);
    border-color: rgba(196, 140, 179, 0.4);
}
.step-number {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #83A6CE, #C48CB3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 12px;
}
.step-icon-wrap {
    width: 64px;
    height: 64px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(131, 166, 206, 0.15), rgba(196, 140, 179, 0.15));
    border: 1px solid rgba(131, 166, 206, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    margin: 0 auto 16px;
    box-shadow: 0 8px 20px rgba(131, 166, 206, 0.15);
}
.step-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #284155;
    margin-bottom: 10px;
}
.step-desc {
    font-size: 0.9rem;
    color: #5a7690;
    line-height: 1.6;
}

/* Premium 3D Feature Cards */
.feature-card-premium {
    background: #FFFFFF;
    border-radius: 22px;
    padding: 28px 22px;
    border: 1px solid rgba(131, 166, 206, 0.2);
    box-shadow: 0 8px 24px rgba(40, 65, 85, 0.06);
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: default;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.feature-card-premium::after {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: radial-gradient(circle at 80% 20%, rgba(229, 201, 215, 0.15), transparent 70%);
    pointer-events: none;
}
.feature-card-premium:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 45px rgba(40, 65, 85, 0.12), 0 0 25px rgba(196, 140, 179, 0.15);
    border-color: rgba(196, 140, 179, 0.45);
}
.feature-icon-wrap {
    width: 58px; height: 58px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 16px;
    background: linear-gradient(135deg, rgba(131, 166, 206, 0.14), rgba(196, 140, 179, 0.14));
    border: 1px solid rgba(131, 166, 206, 0.22);
    box-shadow: 0 6px 16px rgba(131, 166, 206, 0.15);
}

/* Floating Audio Control */
.audio-btn {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 10000;
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: rgba(30, 48, 64, 0.85);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1.5px solid rgba(229, 201, 215, 0.35);
    color: #FFFFFF;
    font-size: 1.45rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35), 0 0 18px rgba(131, 166, 206, 0.25);
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.audio-btn:hover {
    background: rgba(45, 72, 96, 0.95);
    border-color: rgba(196, 140, 179, 0.7);
    transform: scale(1.1) translateY(-2px);
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45), 0 0 24px rgba(196, 140, 179, 0.4);
}

/* Responsive Mobile Layouts */
@media (max-width: 768px) {
    .glass-navbar,
    div[data-testid="stVerticalBlock"]:has(> div > div > .nav-glass-anchor) {
        width: 96vw !important;
        padding: 4px 8px !important;
        border-radius: 24px !important;
    }
    .home-hero-section {
        padding: 90px 16px 40px !important;
        min-height: auto !important;
    }
    .hero-title {
        font-size: 2.2rem !important;
    }
    .step-card-3d, .feature-card-premium {
        margin-bottom: 16px;
    }
    .audio-btn {
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        font-size: 1.25rem;
    }
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Main Design System CSS  (CLAY_CSS)
# ─────────────────────────────────────────────────────────────────────────────

CLAY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════════════════════════════
   PREMIUM KEYFRAME ANIMATIONS
══════════════════════════════════════════════════════════════════ */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-12px); }
}
@keyframes glow-pulse {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
}
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes particle-drift {
    0%   { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { transform: translateY(-100vh) translateX(50px) rotate(360deg); opacity: 0; }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes wave {
    0%, 100% { transform: scaleY(0.3); }
    50%       { transform: scaleY(1); }
}
@keyframes slide-in-right {
    from { opacity: 0; transform: translateX(40px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes signflow-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(131,166,206,0.5); }
    70%  { box-shadow: 0 0 0 12px rgba(131,166,206,0); }
    100% { box-shadow: 0 0 0 0 rgba(131,166,206,0); }
}
@keyframes signflow-shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes orbit {
    from { transform: rotate(0deg) translateX(28px) rotate(0deg); }
    to   { transform: rotate(360deg) translateX(28px) rotate(-360deg); }
}

/* ══════════════════════════════════════════════════════════════════
   RESET & BASE
══════════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #284155;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Hide default Streamlit chrome ── */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* ── App background — soft gradient ── */
.stApp {
    background: linear-gradient(150deg, #f0f4f8 0%, #e8eef5 40%, #f5eef8 100%);
    min-height: 100vh;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1140px !important;
}

/* ══════════════════════════════════════════════════════════════════
   SIDEBAR  — Deep Navy dark panel
══════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2f3f 0%, #284155 55%, #1e3347 100%);
    border-right: 1px solid rgba(131,166,206,0.15);
    box-shadow: 4px 0 32px rgba(40,65,85,0.35);
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.9) !important;
}
section[data-testid="stSidebar"] hr {
    border-top: 1px solid rgba(131,166,206,0.2) !important;
}

/* ══════════════════════════════════════════════════════════════════
   GLASSMORPHISM CARDS
══════════════════════════════════════════════════════════════════ */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px;
    padding: 28px;
    box-shadow:
        0 8px 32px rgba(40,65,85,0.25),
        inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.3s ease;
}
.glass-card:hover {
    background: rgba(255,255,255,0.10);
    border-color: rgba(255,255,255,0.25);
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(40,65,85,0.3);
}

/* ══════════════════════════════════════════════════════════════════
   PREMIUM CLAY CARDS
══════════════════════════════════════════════════════════════════ */
.pro-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow:
        0 1px 3px rgba(40,65,85,0.06),
        0 4px 16px rgba(40,65,85,0.08),
        0 1px 1px rgba(255,255,255,0.8) inset;
    border: 1px solid rgba(131,166,206,0.18);
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.pro-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #83A6CE, #C48CB3, #83A6CE);
    background-size: 200% 100%;
    animation: gradient-shift 4s ease infinite;
    opacity: 0.85;
}
.pro-card:hover {
    transform: translateY(-4px);
    box-shadow:
        0 2px 6px rgba(40,65,85,0.08),
        0 16px 40px rgba(40,65,85,0.14);
}

/* clay-card alias */
.clay-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow:
        0 1px 3px rgba(40,65,85,0.06),
        0 4px 16px rgba(40,65,85,0.08);
    border: 1px solid rgba(131,166,206,0.18);
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.clay-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #83A6CE, #C48CB3);
    opacity: 0.7;
}
.clay-card:hover {
    transform: translateY(-3px);
    box-shadow:
        0 2px 8px rgba(40,65,85,0.1),
        0 12px 32px rgba(40,65,85,0.12);
}

.clay-card-blue   { border-color: rgba(131,166,206,0.3); }
.clay-card-violet { border-color: rgba(196,140,179,0.3); }
.clay-card-cyan   { border-color: rgba(131,166,206,0.25); }

/* Feature card */
.feature-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 22px 20px;
    margin-bottom: 14px;
    text-align: center;
    border: 1px solid rgba(131,166,206,0.2);
    box-shadow: 0 2px 12px rgba(40,65,85,0.07);
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1);
    position: relative;
    overflow: hidden;
}
.feature-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #83A6CE, #C48CB3);
    opacity: 0;
    transition: opacity 0.25s;
}
.feature-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 12px 32px rgba(40,65,85,0.14);
    border-color: rgba(131,166,206,0.4);
}
.feature-card:hover::after { opacity: 1; }

/* 3D Step Cards */
.step-card-3d {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    border: 1px solid rgba(131,166,206,0.2);
    box-shadow: 0 4px 20px rgba(40,65,85,0.08), 0 1px 1px rgba(255,255,255,0.8) inset;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.step-card-3d::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #83A6CE, #C48CB3);
}
.step-card-3d:hover {
    transform: translateY(-8px) rotateX(5deg);
    box-shadow: 0 20px 40px rgba(40,65,85,0.15);
}
.step-number {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #83A6CE, #C48CB3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 8px;
}

/* Premium Feature Cards (Section 4) */
.feature-card-premium {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 24px 20px;
    border: 1px solid rgba(131,166,206,0.18);
    box-shadow: 0 4px 20px rgba(40,65,85,0.07);
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    cursor: default;
    position: relative;
    overflow: hidden;
}
.feature-card-premium:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 16px 40px rgba(40,65,85,0.15);
    border-color: rgba(196,140,179,0.4);
}
.feature-icon-wrap {
    width: 60px; height: 60px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    margin: 0 auto 16px;
    background: linear-gradient(135deg, rgba(131,166,206,0.15), rgba(196,140,179,0.12));
    border: 1px solid rgba(131,166,206,0.2);
}

/* ══════════════════════════════════════════════════════════════════
   HERO / PAGE HEADER
══════════════════════════════════════════════════════════════════ */
.pro-page-header {
    background: linear-gradient(135deg, #284155 0%, #3a5a75 50%, #284155 100%);
    background-size: 200% 200%;
    animation: gradient-shift 8s ease infinite;
    border-radius: 24px;
    padding: 36px 40px;
    margin-bottom: 32px;
    color: #FFFFFF;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(40,65,85,0.3);
}
.pro-page-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(131,166,206,0.2) 0%, transparent 70%);
    pointer-events: none;
    animation: float 6s ease-in-out infinite;
}
.pro-page-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -30px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(196,140,179,0.15) 0%, transparent 70%);
    pointer-events: none;
    animation: float 8s ease-in-out infinite reverse;
}

.clay-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #284155 0%, #83A6CE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.clay-subtitle {
    text-align: center;
    color: #6b8aaa;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 4px;
    letter-spacing: 0.2px;
}

/* ══════════════════════════════════════════════════════════════════
   AUDIO BUTTON
══════════════════════════════════════════════════════════════════ */
.audio-btn {
    position: fixed;
    bottom: 32px; right: 32px;
    z-index: 10000;
    width: 56px; height: 56px;
    border-radius: 50%;
    background: rgba(40,65,85,0.85);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(131,166,206,0.4);
    color: white;
    font-size: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 8px 24px rgba(40,65,85,0.4);
    transition: all 0.3s ease;
}
.audio-btn:hover {
    background: rgba(131,166,206,0.3);
    border-color: rgba(131,166,206,0.7);
    transform: scale(1.1);
}

/* ══════════════════════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════════════════════ */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
    border: none !important;
    font-size: 0.9rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #83A6CE, #C48CB3) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(131,166,206,0.45) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6e93bd, #b47aa0) !important;
    box-shadow: 0 6px 24px rgba(131,166,206,0.55) !important;
    transform: translateY(-2px) !important;
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #284155 !important;
    border: 1.5px solid rgba(131,166,206,0.4) !important;
    box-shadow: 0 2px 8px rgba(40,65,85,0.07) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #E5C9D7 !important;
    border-color: #C48CB3 !important;
    color: #284155 !important;
    box-shadow: 0 4px 16px rgba(196,140,179,0.25) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(131,166,206,0.1) !important;
    color: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(131,166,206,0.2) !important;
    border-radius: 12px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding-left: 16px !important;
    box-shadow: none !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(131,166,206,0.22) !important;
    border-color: rgba(131,166,206,0.4) !important;
    color: #FFFFFF !important;
    transform: translateX(4px) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(131,166,206,0.35), rgba(196,140,179,0.3)) !important;
    border-color: rgba(196,140,179,0.5) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 12px rgba(196,140,179,0.2) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(131,166,206,0.45), rgba(196,140,179,0.4)) !important;
    transform: translateX(4px) !important;
}

/* ══════════════════════════════════════════════════════════════════
   STATUS BADGES
══════════════════════════════════════════════════════════════════ */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 50px;
    font-size: 0.77rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.badge-green  { background: rgba(131,166,206,0.12); color: #284155; border: 1px solid rgba(131,166,206,0.4); }
.badge-blue   { background: rgba(131,166,206,0.10); color: #3a5a75; border: 1px solid rgba(131,166,206,0.3); }
.badge-yellow { background: #fef9ee; color: #b45309; border: 1px solid rgba(180,83,9,0.2); }
.badge-red    { background: #fff1f2; color: #be123c; border: 1px solid rgba(190,18,60,0.2); }
.badge-violet { background: rgba(196,140,179,0.10); color: #8a4f7a; border: 1px solid rgba(196,140,179,0.3); }

section[data-testid="stSidebar"] .badge-green  { background: rgba(131,166,206,0.2) !important; color: #e0edf7 !important; border-color: rgba(131,166,206,0.4) !important; }
section[data-testid="stSidebar"] .badge-yellow { background: rgba(255,248,225,0.1) !important; color: #fde68a !important; border-color: rgba(253,230,138,0.3) !important; }
section[data-testid="stSidebar"] .badge-red    { background: rgba(255,241,242,0.1) !important; color: #fca5a5 !important; border-color: rgba(252,165,165,0.3) !important; }

/* ══════════════════════════════════════════════════════════════════
   CAMERA WRAPPER
══════════════════════════════════════════════════════════════════ */
.camera-wrapper {
    border-radius: 22px;
    overflow: hidden;
    box-shadow:
        0 0 0 3px rgba(131,166,206,0.3),
        0 0 0 6px rgba(196,140,179,0.12),
        0 12px 40px rgba(40,65,85,0.2);
    position: relative;
    background: #1a2f3f;
    transition: box-shadow 0.3s ease;
}
.camera-wrapper:hover {
    box-shadow:
        0 0 0 3px rgba(131,166,206,0.5),
        0 0 0 6px rgba(196,140,179,0.2),
        0 16px 48px rgba(40,65,85,0.25);
}
.camera-active { animation: signflow-pulse 1.8s ease-in-out infinite; }

/* ══════════════════════════════════════════════════════════════════
   TRANSLATION CARDS
══════════════════════════════════════════════════════════════════ */
.trans-card {
    background: linear-gradient(135deg, rgba(229,201,215,0.3), rgba(131,166,206,0.15));
    border: 1px solid rgba(196,140,179,0.25);
    border-radius: 16px;
    padding: 16px 20px;
    margin: 8px 0;
    box-shadow: 0 2px 10px rgba(40,65,85,0.06);
    transition: all 0.2s ease;
    animation: fade-in-up 0.4s ease both;
}
.trans-card:hover {
    border-color: rgba(196,140,179,0.5);
    transform: translateX(4px);
    box-shadow: 0 4px 18px rgba(196,140,179,0.2);
}
.trans-lang {
    font-size: 0.68rem;
    color: #C48CB3;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.trans-text {
    font-size: 1.1rem;
    color: #284155;
    font-weight: 600;
    margin-top: 5px;
}

/* ══════════════════════════════════════════════════════════════════
   SENTENCE BOX
══════════════════════════════════════════════════════════════════ */
.sentence-box {
    background: #FFFFFF;
    border: 2px solid rgba(131,166,206,0.3);
    border-radius: 18px;
    padding: 20px 24px;
    font-size: 1.25rem;
    color: #284155;
    min-height: 72px;
    word-break: break-word;
    line-height: 1.65;
    font-weight: 500;
    box-shadow:
        inset 0 2px 8px rgba(40,65,85,0.04),
        0 2px 12px rgba(40,65,85,0.06);
    letter-spacing: 0.2px;
    transition: border-color 0.2s;
}
.sentence-box:focus-within {
    border-color: #83A6CE;
    box-shadow: inset 0 2px 8px rgba(40,65,85,0.04), 0 0 0 3px rgba(131,166,206,0.15);
}

/* ══════════════════════════════════════════════════════════════════
   SUBTITLE STRIP
══════════════════════════════════════════════════════════════════ */
.subtitle-strip {
    background: linear-gradient(90deg, rgba(229,201,215,0.5), rgba(131,166,206,0.3));
    border: 1px solid rgba(196,140,179,0.3);
    border-radius: 0 0 18px 18px;
    padding: 12px 24px;
    font-size: 1rem;
    color: #284155;
    text-align: center;
    font-weight: 600;
    letter-spacing: 0.2px;
    animation: slide-in-right 0.4s ease both;
}

/* ══════════════════════════════════════════════════════════════════
   CONFIDENCE BARS
══════════════════════════════════════════════════════════════════ */
.conf-track {
    background: rgba(40,65,85,0.08);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.conf-fill-high   { background: linear-gradient(90deg, #83A6CE, #5b9abf); border-radius: 8px; height: 100%; }
.conf-fill-medium { background: linear-gradient(90deg, #f59e0b, #d97706); border-radius: 8px; height: 100%; }
.conf-fill-low    { background: linear-gradient(90deg, #C48CB3, #a86f99); border-radius: 8px; height: 100%; }

/* ══════════════════════════════════════════════════════════════════
   STAT CARDS (sidebar)
══════════════════════════════════════════════════════════════════ */
.stat-card {
    background: rgba(131,166,206,0.12);
    border: 1px solid rgba(131,166,206,0.2);
    border-radius: 14px;
    padding: 12px 14px;
    text-align: center;
    transition: background 0.2s;
}
.stat-card:hover { background: rgba(131,166,206,0.18); }
.stat-label {
    font-size: 0.63rem;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-top: 3px;
}

/* ══════════════════════════════════════════════════════════════════
   GESTURE CHIPS
══════════════════════════════════════════════════════════════════ */
.gesture-chip {
    display: inline-block;
    background: rgba(131,166,206,0.15);
    border: 1px solid rgba(131,166,206,0.3);
    border-radius: 50px;
    padding: 4px 14px;
    margin: 3px;
    font-size: 0.8rem;
    color: #284155;
    font-weight: 600;
    transition: all 0.2s cubic-bezier(0.34,1.56,0.64,1);
    cursor: default;
}
.gesture-chip:hover {
    background: rgba(196,140,179,0.2);
    border-color: rgba(196,140,179,0.5);
    transform: translateY(-3px);
}

/* ══════════════════════════════════════════════════════════════════
   STEP ITEMS (How-To sidebar)
══════════════════════════════════════════════════════════════════ */
.step-item {
    background: rgba(229,201,215,0.25);
    border-left: 3px solid #C48CB3;
    border-radius: 0 14px 14px 0;
    padding: 11px 18px;
    margin-bottom: 9px;
    font-size: 0.88rem;
    color: #284155;
    box-shadow: 0 2px 8px rgba(40,65,85,0.05);
    font-weight: 500;
    transition: border-color 0.2s, transform 0.2s;
}
.step-item:hover { border-color: #83A6CE; transform: translateX(3px); }
.step-num { color: #83A6CE; font-weight: 800; margin-right: 8px; }

/* ══════════════════════════════════════════════════════════════════
   STREAMLIT METRICS
══════════════════════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(131,166,206,0.2) !important;
    border-radius: 18px !important;
    padding: 16px !important;
    box-shadow: 0 2px 10px rgba(40,65,85,0.07) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(40,65,85,0.12) !important;
}
[data-testid="metric-container"] label {
    color: #6b8aaa !important;
    font-weight: 600 !important;
    font-size: 0.73rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #284155 !important;
    font-weight: 800 !important;
}

/* ══════════════════════════════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid rgba(131,166,206,0.2) !important;
    margin: 16px 0 !important;
}

/* ══════════════════════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 4px;
    gap: 4px;
    box-shadow: 0 2px 10px rgba(40,65,85,0.07);
    border: 1px solid rgba(131,166,206,0.2);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 600;
    color: #6b8aaa;
    transition: all 0.18s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #83A6CE, #C48CB3) !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 12px rgba(131,166,206,0.35) !important;
}

/* ══════════════════════════════════════════════════════════════════
   INPUTS / SELECTS / SLIDERS
══════════════════════════════════════════════════════════════════ */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(131,166,206,0.3) !important;
    border-radius: 12px !important;
    color: #284155 !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: #83A6CE !important;
    box-shadow: 0 0 0 3px rgba(131,166,206,0.15) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #83A6CE !important;
    box-shadow: 0 2px 8px rgba(131,166,206,0.4) !important;
    border-radius: 50% !important;
}

/* ══════════════════════════════════════════════════════════════════
   PROGRESS BARS
══════════════════════════════════════════════════════════════════ */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #83A6CE, #C48CB3) !important;
    border-radius: 8px !important;
}
.stProgress > div > div {
    background: rgba(40,65,85,0.08) !important;
    border-radius: 8px !important;
}

/* ══════════════════════════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f4f8; border-radius: 3px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #83A6CE, #C48CB3);
    border-radius: 3px;
}

/* ══════════════════════════════════════════════════════════════════
   SECTION HEADINGS
══════════════════════════════════════════════════════════════════ */
.clay-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 10px;
}
.section-title-dark {
    font-size: 0.72rem;
    font-weight: 700;
    color: #6b8aaa;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 10px;
}

/* ══════════════════════════════════════════════════════════════════
   LOADING & ERROR
══════════════════════════════════════════════════════════════════ */
.clay-loading {
    text-align: center;
    padding: 32px 20px;
    font-size: 1.1rem;
    color: #83A6CE;
    font-weight: 600;
    animation: glow-pulse 1.5s ease-in-out infinite;
}
.clay-error {
    background: #fff1f2;
    border: 1px solid rgba(190,18,60,0.2);
    border-radius: 14px;
    padding: 14px 18px;
    color: #be123c;
    font-weight: 600;
    font-size: 0.9rem;
}

/* ══════════════════════════════════════════════════════════════════
   PREDICTION CARD
══════════════════════════════════════════════════════════════════ */
.prediction-card {
    background: linear-gradient(145deg, #284155 0%, #3a5a75 60%, #2d4860 100%);
    border-radius: 22px;
    padding: 30px 26px;
    text-align: center;
    box-shadow:
        0 8px 32px rgba(40,65,85,0.35),
        0 2px 8px rgba(40,65,85,0.2),
        inset 0 1px 0 rgba(131,166,206,0.15);
    color: #FFFFFF;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.prediction-card::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(196,140,179,0.15) 0%, transparent 70%);
    pointer-events: none;
    animation: float 6s ease-in-out infinite;
}
.prediction-gesture {
    font-size: 3rem;
    font-weight: 900;
    color: #FFFFFF;
    line-height: 1.1;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.prediction-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: rgba(255,255,255,0.45);
    font-weight: 700;
    margin-bottom: 10px;
}

/* ══════════════════════════════════════════════════════════════════
   MODE BADGES
══════════════════════════════════════════════════════════════════ */
.mode-badge-word {
    display: inline-block;
    background: rgba(131,166,206,0.2);
    border: 1px solid rgba(131,166,206,0.45);
    border-radius: 50px;
    padding: 3px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    color: #c8dff0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.mode-badge-char {
    display: inline-block;
    background: rgba(196,140,179,0.22);
    border: 1px solid rgba(196,140,179,0.45);
    border-radius: 50px;
    padding: 3px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    color: #e8c8e0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

/* ══════════════════════════════════════════════════════════════════
   AI PANEL
══════════════════════════════════════════════════════════════════ */
.ai-panel {
    background: #FFFFFF;
    border-radius: 22px;
    padding: 26px;
    margin-bottom: 18px;
    box-shadow:
        0 4px 20px rgba(40,65,85,0.1),
        0 1px 4px rgba(40,65,85,0.06);
    border: 1px solid rgba(131,166,206,0.2);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.ai-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #83A6CE, #C48CB3, #83A6CE);
    background-size: 200% 100%;
    animation: gradient-shift 4s ease infinite;
}
.ai-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(40,65,85,0.15), 0 1px 4px rgba(40,65,85,0.06);
}
.ai-panel h3 {
    color: #284155;
    margin: 0 0 14px 0;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.gesture-big {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #284155, #83A6CE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -0.5px;
    line-height: 1.1;
    animation: fade-in-up 0.35s ease both;
}
.gesture-meaning {
    font-size: 0.88rem;
    color: #6b8aaa;
    text-align: center;
    margin-top: 6px;
    font-weight: 500;
}
.gesture-recognized {
    background: linear-gradient(
        90deg,
        rgba(131,166,206,0.15) 25%,
        rgba(196,140,179,0.25) 50%,
        rgba(131,166,206,0.15) 75%
    );
    background-size: 200% 100%;
    animation: signflow-shimmer 1.5s ease infinite;
}

/* ══════════════════════════════════════════════════════════════════
   CHECKBOX / RADIO
══════════════════════════════════════════════════════════════════ */
.stCheckbox span, .stRadio span { color: #284155 !important; }
section[data-testid="stSidebar"] .stCheckbox span,
section[data-testid="stSidebar"] .stRadio span { color: rgba(255,255,255,0.85) !important; }

/* ══════════════════════════════════════════════════════════════════
   DATAFRAME / TABLE
══════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden;
    border: 1px solid rgba(131,166,206,0.2) !important;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HIGH_CONTRAST_CSS
# ─────────────────────────────────────────────────────────────────────────────

HIGH_CONTRAST_CSS = """
<style>
.stApp { background: #000000 !important; }
.clay-card, .pro-card, .ai-panel, .trans-card, .stat-card {
    background: #111 !important;
    border: 2px solid #ffffff !important;
}
.gesture-big, .prediction-gesture {
    font-size: 3rem !important;
    background: none !important;
    -webkit-text-fill-color: #ffff00 !important;
    color: #ffff00 !important;
}
.sentence-box {
    background: #111 !important;
    color: #ffffff !important;
    border: 2px solid #ffffff !important;
    font-size: 1.4rem !important;
}
.trans-text { color: #ffffff !important; font-size: 1.4rem !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# LARGE_FONT_CSS
# ─────────────────────────────────────────────────────────────────────────────

LARGE_FONT_CSS = """
<style>
html, body, [class*="css"] { font-size: 18px !important; }
.gesture-big, .prediction-gesture { font-size: 3.4rem !important; }
.sentence-box { font-size: 1.5rem !important; }
.trans-text   { font-size: 1.5rem !important; }
p, li, .stMarkdown { font-size: 1.05rem !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SVG Illustrations  — SignFlow Premium Palette
# ─────────────────────────────────────────────────────────────────────────────

def svg_hand_3d() -> str:
    """3D clay-style hand / ILY gesture SVG — SignFlow palette."""
    return """
<svg width="110" height="120" viewBox="0 0 110 120" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="palmGrad" cx="45%" cy="55%" r="55%">
      <stop offset="0%" stop-color="#E5C9D7"/>
      <stop offset="55%" stop-color="#83A6CE"/>
      <stop offset="100%" stop-color="#284155"/>
    </radialGradient>
    <radialGradient id="fingerGrad" cx="40%" cy="30%" r="60%">
      <stop offset="0%" stop-color="#E5C9D7"/>
      <stop offset="100%" stop-color="#83A6CE"/>
    </radialGradient>
    <radialGradient id="thumbGrad" cx="35%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="100%" stop-color="#C48CB3"/>
    </radialGradient>
    <filter id="handShadow" x="-20%" y="-10%" width="150%" height="150%">
      <feDropShadow dx="4" dy="6" stdDeviation="6" flood-color="#284155" flood-opacity="0.45"/>
    </filter>
    <filter id="glowH">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <ellipse cx="55" cy="112" rx="32" ry="7" fill="#284155" opacity="0.3"/>
  <rect x="22" y="54" width="58" height="52" rx="14" fill="url(#palmGrad)" filter="url(#handShadow)"/>
  <ellipse cx="42" cy="62" rx="12" ry="6" fill="rgba(255,255,255,0.22)" transform="rotate(-10 42 62)"/>
  <rect x="28" y="18" width="14" height="42" rx="7" fill="url(#fingerGrad)" filter="url(#handShadow)"/>
  <ellipse cx="35" cy="22" rx="5" ry="3" fill="rgba(255,255,255,0.28)"/>
  <rect x="46" y="12" width="14" height="46" rx="7" fill="url(#fingerGrad)" filter="url(#handShadow)"/>
  <ellipse cx="53" cy="16" rx="5" ry="3" fill="rgba(255,255,255,0.28)"/>
  <rect x="64" y="40" width="13" height="22" rx="6" fill="url(#fingerGrad)" opacity="0.85"/>
  <rect x="72" y="28" width="11" height="34" rx="5" fill="url(#fingerGrad)" filter="url(#handShadow)"/>
  <ellipse cx="77" cy="32" rx="4" ry="2.5" fill="rgba(255,255,255,0.28)"/>
  <rect x="6" y="52" width="22" height="12" rx="6" fill="url(#thumbGrad)" transform="rotate(-30 6 52)" filter="url(#handShadow)"/>
  <ellipse cx="12" cy="52" rx="4" ry="2.5" fill="rgba(255,255,255,0.3)" transform="rotate(-30 12 52)"/>
  <circle cx="35" cy="58" r="3" fill="rgba(40,65,85,0.18)"/>
  <circle cx="53" cy="56" r="3" fill="rgba(40,65,85,0.18)"/>
  <circle cx="70" cy="58" r="2.5" fill="rgba(40,65,85,0.12)"/>
  <circle cx="55" cy="72" r="28" stroke="rgba(131,166,206,0.3)" stroke-width="1.5" fill="none" filter="url(#glowH)"/>
</svg>"""


def svg_ai_robot() -> str:
    """3D clay-style AI assistant robot SVG — SignFlow palette."""
    return """
<svg width="90" height="100" viewBox="0 0 90 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="headGradR" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#83A6CE"/>
      <stop offset="100%" stop-color="#284155"/>
    </radialGradient>
    <radialGradient id="bodyGradR" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#C48CB3"/>
      <stop offset="100%" stop-color="#284155"/>
    </radialGradient>
    <radialGradient id="screenGradR" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#E5C9D7"/>
      <stop offset="100%" stop-color="#C48CB3"/>
    </radialGradient>
    <filter id="botShadowR">
      <feDropShadow dx="3" dy="5" stdDeviation="5" flood-color="#284155" flood-opacity="0.4"/>
    </filter>
  </defs>
  <ellipse cx="45" cy="97" rx="26" ry="5" fill="#284155" opacity="0.25"/>
  <rect x="16" y="52" width="58" height="40" rx="14" fill="url(#bodyGradR)" filter="url(#botShadowR)"/>
  <ellipse cx="35" cy="60" rx="14" ry="6" fill="rgba(255,255,255,0.12)" transform="rotate(-8 35 60)"/>
  <rect x="24" y="60" width="42" height="24" rx="8" fill="url(#screenGradR)" opacity="0.9"/>
  <rect x="26" y="62" width="16" height="5" rx="3" fill="rgba(255,255,255,0.35)"/>
  <circle cx="38" cy="74" r="3" fill="#284155"/>
  <circle cx="48" cy="74" r="3" fill="#83A6CE"/>
  <circle cx="58" cy="74" r="3" fill="#C48CB3"/>
  <rect x="18" y="10" width="54" height="46" rx="18" fill="url(#headGradR)" filter="url(#botShadowR)"/>
  <ellipse cx="36" cy="18" rx="14" ry="7" fill="rgba(255,255,255,0.15)" transform="rotate(-10 36 18)"/>
  <circle cx="32" cy="34" r="9" fill="#284155"/>
  <circle cx="32" cy="34" r="6" fill="#83A6CE"/>
  <circle cx="32" cy="34" r="3.5" fill="#E5C9D7"/>
  <circle cx="30" cy="32" r="1.2" fill="rgba(255,255,255,0.8)"/>
  <circle cx="58" cy="34" r="9" fill="#284155"/>
  <circle cx="58" cy="34" r="6" fill="#83A6CE"/>
  <circle cx="58" cy="34" r="3.5" fill="#E5C9D7"/>
  <circle cx="56" cy="32" r="1.2" fill="rgba(255,255,255,0.8)"/>
  <rect x="42" y="2" width="6" height="12" rx="3" fill="#C48CB3"/>
  <circle cx="45" cy="2" r="5" fill="#E5C9D7"/>
  <circle cx="45" cy="2" r="3" fill="#FFFFFF"/>
  <rect x="4" y="22" width="14" height="22" rx="7" fill="url(#headGradR)" opacity="0.85"/>
  <rect x="72" y="22" width="14" height="22" rx="7" fill="url(#headGradR)" opacity="0.85"/>
  <rect x="2" y="56" width="14" height="28" rx="7" fill="url(#bodyGradR)" opacity="0.8"/>
  <rect x="74" y="56" width="14" height="28" rx="7" fill="url(#bodyGradR)" opacity="0.8"/>
</svg>"""


def svg_microphone() -> str:
    """3D clay-style microphone SVG — SignFlow palette."""
    return """
<svg width="64" height="80" viewBox="0 0 64 80" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="micGradS" cx="40%" cy="30%" r="60%">
      <stop offset="0%" stop-color="#E5C9D7"/>
      <stop offset="100%" stop-color="#C48CB3"/>
    </radialGradient>
    <radialGradient id="baseGradS" cx="40%" cy="35%" r="55%">
      <stop offset="0%" stop-color="#83A6CE"/>
      <stop offset="100%" stop-color="#284155"/>
    </radialGradient>
    <filter id="micShadowS">
      <feDropShadow dx="3" dy="4" stdDeviation="4" flood-color="#284155" flood-opacity="0.35"/>
    </filter>
  </defs>
  <ellipse cx="32" cy="77" rx="18" ry="4" fill="#284155" opacity="0.25"/>
  <rect x="14" y="68" width="36" height="9" rx="5" fill="url(#baseGradS)" filter="url(#micShadowS)"/>
  <ellipse cx="25" cy="70" rx="8" ry="3" fill="rgba(255,255,255,0.15)"/>
  <rect x="28" y="46" width="8" height="24" rx="4" fill="url(#baseGradS)" opacity="0.9"/>
  <rect x="14" y="6" width="36" height="46" rx="18" fill="url(#micGradS)" filter="url(#micShadowS)"/>
  <ellipse cx="26" cy="14" rx="8" ry="12" fill="rgba(255,255,255,0.2)" transform="rotate(-10 26 14)"/>
  <line x1="20" y1="28" x2="44" y2="28" stroke="rgba(40,65,85,0.15)" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="18" y1="34" x2="46" y2="34" stroke="rgba(40,65,85,0.15)" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="20" y1="40" x2="44" y2="40" stroke="rgba(40,65,85,0.15)" stroke-width="1.5" stroke-linecap="round"/>
  <ellipse cx="32" cy="11" rx="8" ry="5" fill="rgba(255,255,255,0.25)"/>
</svg>"""


def svg_speaker() -> str:
    """3D clay-style speaker SVG — SignFlow palette."""
    return """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="spkGradS" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#83A6CE"/>
      <stop offset="100%" stop-color="#284155"/>
    </radialGradient>
    <filter id="spkShadowS">
      <feDropShadow dx="3" dy="4" stdDeviation="4" flood-color="#284155" flood-opacity="0.3"/>
    </filter>
  </defs>
  <ellipse cx="32" cy="61" rx="22" ry="5" fill="#284155" opacity="0.25"/>
  <rect x="6" y="8" width="52" height="50" rx="16" fill="url(#spkGradS)" filter="url(#spkShadowS)"/>
  <ellipse cx="22" cy="16" rx="12" ry="7" fill="rgba(255,255,255,0.15)" transform="rotate(-10 22 16)"/>
  <circle cx="32" cy="34" r="16" fill="#284155"/>
  <circle cx="32" cy="34" r="10" fill="#83A6CE"/>
  <circle cx="32" cy="34" r="5" fill="#C48CB3"/>
  <circle cx="32" cy="34" r="2.5" fill="#E5C9D7"/>
  <circle cx="27" cy="29" r="3" fill="rgba(255,255,255,0.15)"/>
  <circle cx="16" cy="17" r="3" fill="rgba(0,0,0,0.15)"/>
  <circle cx="48" cy="17" r="3" fill="rgba(0,0,0,0.15)"/>
  <circle cx="16" cy="51" r="3" fill="rgba(0,0,0,0.15)"/>
  <circle cx="48" cy="51" r="3" fill="rgba(0,0,0,0.15)"/>
</svg>"""


def svg_analytics() -> str:
    """3D clay-style analytics chart SVG — SignFlow palette."""
    return """
<svg width="80" height="70" viewBox="0 0 80 70" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bar1S" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#83A6CE"/>
      <stop offset="100%" stop-color="#284155"/>
    </linearGradient>
    <linearGradient id="bar2S" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#C48CB3"/>
      <stop offset="100%" stop-color="#284155"/>
    </linearGradient>
    <linearGradient id="bar3S" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#E5C9D7"/>
      <stop offset="100%" stop-color="#83A6CE"/>
    </linearGradient>
    <linearGradient id="bar4S" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#C48CB3"/>
      <stop offset="100%" stop-color="#8a4f7a"/>
    </linearGradient>
    <filter id="barShadowS">
      <feDropShadow dx="2" dy="3" stdDeviation="3" flood-color="#284155" flood-opacity="0.25"/>
    </filter>
  </defs>
  <rect x="4" y="54" width="72" height="10" rx="5" fill="#E5C9D7" opacity="0.6"/>
  <rect x="10" y="26" width="12" height="30" rx="6" fill="url(#bar1S)" filter="url(#barShadowS)"/>
  <ellipse cx="16" cy="29" rx="4" ry="2" fill="rgba(255,255,255,0.3)"/>
  <rect x="26" y="14" width="12" height="42" rx="6" fill="url(#bar2S)" filter="url(#barShadowS)"/>
  <ellipse cx="32" cy="17" rx="4" ry="2" fill="rgba(255,255,255,0.3)"/>
  <rect x="42" y="34" width="12" height="22" rx="6" fill="url(#bar3S)" filter="url(#barShadowS)"/>
  <ellipse cx="48" cy="37" rx="4" ry="2" fill="rgba(255,255,255,0.3)"/>
  <rect x="58" y="20" width="12" height="36" rx="6" fill="url(#bar4S)" filter="url(#barShadowS)"/>
  <ellipse cx="64" cy="23" rx="4" ry="2" fill="rgba(255,255,255,0.3)"/>
  <circle cx="70" cy="10" r="8" fill="url(#bar2S)" opacity="0.7" filter="url(#barShadowS)"/>
  <circle cx="67" cy="7" r="2.5" fill="rgba(255,255,255,0.35)"/>
</svg>"""


def svg_communication_scene() -> str:
    return '''
    <svg viewBox="0 0 960 520" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:960px;height:auto;overflow:visible;">
      <defs>
        <radialGradient id="heroStageGlow" cx="50%" cy="50%" r="65%">
          <stop offset="0%" stop-color="#3a5a75" stop-opacity="0.5"/>
          <stop offset="60%" stop-color="#284155" stop-opacity="0.15"/>
          <stop offset="100%" stop-color="#142431" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="signerGlow" cx="40%" cy="35%" r="60%">
          <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.8"/>
          <stop offset="30%" stop-color="#83A6CE"/>
          <stop offset="85%" stop-color="#284155"/>
          <stop offset="100%" stop-color="#1a2f3f"/>
        </radialGradient>
        <radialGradient id="receiverGlow" cx="40%" cy="35%" r="60%">
          <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.8"/>
          <stop offset="30%" stop-color="#E5C9D7"/>
          <stop offset="80%" stop-color="#C48CB3"/>
          <stop offset="100%" stop-color="#284155"/>
        </radialGradient>
        <radialGradient id="coreSphereGrad" cx="35%" cy="30%" r="70%">
          <stop offset="0%" stop-color="#FFFFFF"/>
          <stop offset="30%" stop-color="#83A6CE"/>
          <stop offset="70%" stop-color="#C48CB3"/>
          <stop offset="100%" stop-color="#284155"/>
        </radialGradient>
        <linearGradient id="beamGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#83A6CE" stop-opacity="0.8"/>
          <stop offset="50%" stop-color="#C48CB3" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#E5C9D7" stop-opacity="0.8"/>
        </linearGradient>
        <linearGradient id="gridGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#83A6CE" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="#284155" stop-opacity="0"/>
        </linearGradient>
        <filter id="softGlow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="6" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="intenseGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="10" result="blur1"/>
          <feGaussianBlur stdDeviation="4" result="blur2"/>
          <feMerge><feMergeNode in="blur1"/><feMergeNode in="blur2"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="cardShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#000000" flood-opacity="0.35"/>
        </filter>
      </defs>
      <ellipse cx="480" cy="380" rx="440" ry="110" fill="url(#heroStageGlow)"/>
      <ellipse cx="480" cy="380" rx="360" ry="85" fill="none" stroke="rgba(131,166,206,0.18)" stroke-width="1.5" stroke-dasharray="8 6"/>
      <ellipse cx="480" cy="380" rx="260" ry="60" fill="none" stroke="rgba(196,140,179,0.22)" stroke-width="1.2"/>
      <line x1="480" y1="380" x2="80" y2="460" stroke="url(#gridGrad)" stroke-width="1.2"/>
      <line x1="480" y1="380" x2="260" y2="480" stroke="url(#gridGrad)" stroke-width="1.2"/>
      <line x1="480" y1="380" x2="700" y2="480" stroke="url(#gridGrad)" stroke-width="1.2"/>
      <line x1="480" y1="380" x2="880" y2="460" stroke="url(#gridGrad)" stroke-width="1.2"/>
      <g opacity="0.6">
        <ellipse cx="140" cy="90" rx="20" ry="8" fill="none" stroke="#83A6CE" stroke-width="3" transform="rotate(-25 140 90)">
          <animateTransform attributeName="transform" type="rotate" values="-25 140 90; 335 140 90" dur="20s" repeatCount="indefinite"/>
        </ellipse>
        <circle cx="830" cy="100" r="14" fill="url(#receiverGlow)" filter="url(#softGlow)">
          <animate attributeName="cy" values="100;85;100" dur="6s" repeatCount="indefinite"/>
        </circle>
        <g transform="translate(340, 100)">
          <polygon points="0,10 14,0 28,10 14,20" fill="rgba(131,166,206,0.4)"/>
          <polygon points="0,10 14,20 14,35 0,25" fill="rgba(131,166,206,0.25)"/>
          <polygon points="14,20 28,10 28,25 14,35" fill="rgba(196,140,179,0.35)"/>
          <animateTransform attributeName="transform" type="translate" values="340,100;340,88;340,100" dur="5s" repeatCount="indefinite"/>
        </g>
      </g>
      <path d="M 270,240 Q 380,180 480,240 T 690,250" fill="none" stroke="url(#beamGrad)" stroke-width="2.5" stroke-dasharray="8 6" opacity="0.65">
        <animate attributeName="stroke-dashoffset" values="0;-42" dur="1.8s" repeatCount="indefinite"/>
      </path>
      <circle r="4.5" fill="#83A6CE" filter="url(#intenseGlow)">
        <animateMotion dur="2.2s" repeatCount="indefinite" path="M 270,240 Q 380,180 480,240 T 690,250"/>
        <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" repeatCount="indefinite"/>
      </circle>
      <circle r="3.5" fill="#C48CB3" filter="url(#intenseGlow)">
        <animateMotion dur="2.2s" begin="0.75s" repeatCount="indefinite" path="M 270,240 Q 380,180 480,240 T 690,250"/>
        <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" begin="0.75s" repeatCount="indefinite"/>
      </circle>
      <circle r="3" fill="#E5C9D7" filter="url(#intenseGlow)">
        <animateMotion dur="2.2s" begin="1.5s" repeatCount="indefinite" path="M 270,240 Q 380,180 480,240 T 690,250"/>
        <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" begin="1.5s" repeatCount="indefinite"/>
      </circle>
      <g id="person-signer" transform="translate(140, 130)">
        <ellipse cx="60" cy="275" rx="55" ry="16" fill="rgba(20,36,49,0.7)"/>
        <path d="M 20,240 C 20,165 40,150 60,150 C 80,150 100,165 100,240 Z" fill="url(#signerGlow)" opacity="0.95"/>
        <rect x="52" y="125" width="16" height="28" rx="6" fill="#83A6CE" opacity="0.9"/>
        <ellipse cx="60" cy="95" rx="26" ry="32" fill="url(#signerGlow)"/>
        <ellipse cx="54" cy="80" rx="10" ry="6" fill="rgba(255,255,255,0.4)" transform="rotate(-15 54 80)"/>
        <rect x="42" y="90" width="36" height="8" rx="4" fill="#83A6CE" filter="url(#softGlow)"/>
        <rect x="46" y="92" width="28" height="4" rx="2" fill="#FFFFFF"/>
        <path d="M 24,165 C 10,195 12,230 20,250" fill="none" stroke="#5b86b5" stroke-width="14" stroke-linecap="round"/>
        <path d="M 95,165 Q 125,185 140,150" fill="none" stroke="#83A6CE" stroke-width="16" stroke-linecap="round"/>
        <path d="M 140,150 Q 155,120 170,110" fill="none" stroke="#9bbde4" stroke-width="14" stroke-linecap="round"/>
        <g transform="translate(160, 75)">
          <ellipse cx="22" cy="30" rx="16" ry="14" fill="#E5C9D7" filter="url(#softGlow)"/>
          <path d="M 10,34 Q -2,30 2,20" fill="none" stroke="#E5C9D7" stroke-width="6" stroke-linecap="round"/>
          <path d="M 16,18 Q 18,6 22,-4" fill="none" stroke="#E5C9D7" stroke-width="5.5" stroke-linecap="round"/>
          <path d="M 24,18 Q 28,6 32,-6" fill="none" stroke="#E5C9D7" stroke-width="5.5" stroke-linecap="round"/>
          <path d="M 31,22 Q 35,16 38,20" fill="none" stroke="#E5C9D7" stroke-width="5" stroke-linecap="round"/>
          <path d="M 36,26 Q 44,18 48,8" fill="none" stroke="#E5C9D7" stroke-width="4.5" stroke-linecap="round"/>
          <g stroke="#83A6CE" stroke-width="1.8" opacity="0.9" filter="url(#softGlow)">
            <line x1="22" y1="40" x2="10" y2="34"/><line x1="22" y1="40" x2="16" y2="18"/>
            <line x1="22" y1="40" x2="24" y2="18"/><line x1="22" y1="40" x2="31" y2="22"/>
            <line x1="22" y1="40" x2="36" y2="26"/><line x1="16" y1="18" x2="19" y2="7"/>
            <line x1="19" y1="7" x2="22" y2="-4"/><line x1="36" y1="26" x2="42" y2="17"/>
            <line x1="42" y1="17" x2="48" y2="8"/>
          </g>
          <circle cx="22" cy="40" r="3" fill="#FFFFFF"/>
          <circle cx="22" cy="-4" r="3.2" fill="#FFFFFF" filter="url(#intenseGlow)"/>
          <circle cx="32" cy="-6" r="3.2" fill="#FFFFFF" filter="url(#intenseGlow)"/>
          <circle cx="48" cy="8" r="3.2" fill="#FFFFFF" filter="url(#intenseGlow)"/>
          <circle cx="22" cy="15" r="26" fill="none" stroke="rgba(131,166,206,0.3)" stroke-width="1.2" stroke-dasharray="4 4">
            <animateTransform attributeName="transform" type="rotate" values="0 22 15;360 22 15" dur="8s" repeatCount="indefinite"/>
          </circle>
        </g>
        <g transform="translate(10, 280)">
          <rect x="0" y="0" width="100" height="26" rx="13" fill="rgba(26,47,63,0.85)" stroke="rgba(131,166,206,0.4)" stroke-width="1"/>
          <circle cx="14" cy="13" r="4" fill="#83A6CE"/>
          <text x="56" y="17" text-anchor="middle" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#FFFFFF">SIGNER</text>
        </g>
      </g>
      <g id="ai-neural-core" transform="translate(480, 240)">
        <ellipse cx="0" cy="0" rx="90" ry="34" fill="none" stroke="rgba(131,166,206,0.35)" stroke-width="1.8" transform="rotate(-30)">
          <animateTransform attributeName="transform" type="rotate" values="-30 0 0;330 0 0" dur="12s" repeatCount="indefinite"/>
        </ellipse>
        <ellipse cx="0" cy="0" rx="80" ry="28" fill="none" stroke="rgba(196,140,179,0.4)" stroke-width="1.6" transform="rotate(45)">
          <animateTransform attributeName="transform" type="rotate" values="45 0 0;-315 0 0" dur="9s" repeatCount="indefinite"/>
        </ellipse>
        <ellipse cx="0" cy="0" rx="70" ry="60" fill="none" stroke="rgba(229,201,215,0.25)" stroke-width="1" stroke-dasharray="6 4">
          <animateTransform attributeName="transform" type="rotate" values="0 0 0;360 0 0" dur="15s" repeatCount="indefinite"/>
        </ellipse>
        <circle cx="0" cy="0" r="38" fill="url(#coreSphereGrad)" filter="url(#intenseGlow)"/>
        <ellipse cx="-12" cy="-12" rx="16" ry="9" fill="rgba(255,255,255,0.6)" transform="rotate(-30 -12 -12)"/>
        <g stroke="#FFFFFF" stroke-width="1.2" opacity="0.85">
          <line x1="-15" y1="-8" x2="10" y2="-12"/><line x1="-15" y1="-8" x2="-4" y2="14"/>
          <line x1="10" y1="-12" x2="16" y2="8"/><line x1="-4" y1="14" x2="16" y2="8"/>
        </g>
        <circle cx="-15" cy="-8" r="3" fill="#FFFFFF"/>
        <circle cx="10" cy="-12" r="3" fill="#E5C9D7"/>
        <circle cx="-4" cy="14" r="3" fill="#83A6CE"/>
        <circle cx="16" cy="8" r="3.5" fill="#FFFFFF"/>
        <g transform="translate(0,52)">
          <rect x="-48" y="0" width="96" height="22" rx="11" fill="rgba(20,36,49,0.9)" stroke="rgba(196,140,179,0.5)" stroke-width="1"/>
          <text x="0" y="15" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" font-weight="800" fill="#E5C9D7">AI CORE</text>
        </g>
        <g transform="translate(-130,-75)">
          <rect x="0" y="0" width="80" height="28" rx="14" fill="rgba(26,47,63,0.85)" stroke="rgba(131,166,206,0.6)" stroke-width="1.2"/>
          <text x="40" y="18" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="700" fill="#83A6CE">Hello</text>
          <animateTransform attributeName="transform" type="translate" values="-130,-75;-130,-85;-130,-75" dur="3.5s" repeatCount="indefinite"/>
        </g>
        <g transform="translate(45,-85)">
          <rect x="0" y="0" width="100" height="28" rx="14" fill="rgba(26,47,63,0.85)" stroke="rgba(196,140,179,0.6)" stroke-width="1.2"/>
          <text x="50" y="18" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="700" fill="#C48CB3">Thank You</text>
          <animateTransform attributeName="transform" type="translate" values="45,-85;45,-95;45,-85" dur="4s" repeatCount="indefinite" begin="0.6s"/>
        </g>
        <g transform="translate(-110,80)">
          <rect x="0" y="0" width="65" height="26" rx="13" fill="rgba(26,47,63,0.85)" stroke="rgba(229,201,215,0.5)" stroke-width="1.2"/>
          <text x="32" y="17" text-anchor="middle" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#E5C9D7">Yes</text>
          <animateTransform attributeName="transform" type="translate" values="-110,80;-110,72;-110,80" dur="3.8s" repeatCount="indefinite" begin="1.2s"/>
        </g>
        <g transform="translate(60,75)">
          <rect x="0" y="0" width="95" height="26" rx="13" fill="rgba(26,47,63,0.85)" stroke="rgba(131,166,206,0.6)" stroke-width="1.2"/>
          <text x="47" y="17" text-anchor="middle" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#83A6CE">I Love You</text>
          <animateTransform attributeName="transform" type="translate" values="60,75;60,66;60,75" dur="4.2s" repeatCount="indefinite" begin="1.8s"/>
        </g>
      </g>
      <g id="person-receiver" transform="translate(710, 130)">
        <ellipse cx="60" cy="275" rx="55" ry="16" fill="rgba(20,36,49,0.7)"/>
        <path d="M 20,240 C 20,165 40,150 60,150 C 80,150 100,165 100,240 Z" fill="url(#receiverGlow)" opacity="0.95"/>
        <rect x="52" y="125" width="16" height="28" rx="6" fill="#C48CB3" opacity="0.9"/>
        <ellipse cx="60" cy="95" rx="26" ry="32" fill="url(#receiverGlow)"/>
        <ellipse cx="52" cy="80" rx="10" ry="6" fill="rgba(255,255,255,0.4)" transform="rotate(-10 52 80)"/>
        <ellipse cx="32" cy="96" rx="5" ry="8" fill="#E5C9D7"/>
        <path d="M 48,106 Q 58,114 68,106" fill="none" stroke="#284155" stroke-width="2.5" stroke-linecap="round"/>
        <path d="M 25,165 Q 40,200 55,190" fill="none" stroke="#C48CB3" stroke-width="14" stroke-linecap="round"/>
        <ellipse cx="58" cy="188" rx="10" ry="8" fill="#E5C9D7"/>
        <path d="M 95,165 C 105,195 102,230 96,250" fill="none" stroke="#a86f99" stroke-width="14" stroke-linecap="round"/>
        <g stroke="#C48CB3" fill="none" opacity="0.8">
          <path d="M 20,85 A 18,18 0 0,0 20,110" stroke-width="2">
            <animate attributeName="opacity" values="0.2;1;0.2" dur="1.5s" repeatCount="indefinite"/>
          </path>
          <path d="M 12,78 A 28,28 0 0,0 12,118" stroke-width="2.2">
            <animate attributeName="opacity" values="0.2;1;0.2" dur="1.5s" begin="0.3s" repeatCount="indefinite"/>
          </path>
          <path d="M 4,70 A 38,38 0 0,0 4,126" stroke-width="2.5">
            <animate attributeName="opacity" values="0.2;1;0.2" dur="1.5s" begin="0.6s" repeatCount="indefinite"/>
          </path>
        </g>
        <g transform="translate(110,150)" opacity="0.85">
          <rect x="0" y="0" width="4" height="20" rx="2" fill="#C48CB3">
            <animate attributeName="height" values="8;24;8" dur="0.9s" repeatCount="indefinite"/>
            <animate attributeName="y" values="16;0;16" dur="0.9s" repeatCount="indefinite"/>
          </rect>
          <rect x="7" y="0" width="4" height="28" rx="2" fill="#83A6CE">
            <animate attributeName="height" values="12;32;12" dur="0.8s" begin="0.15s" repeatCount="indefinite"/>
            <animate attributeName="y" values="12;-8;12" dur="0.8s" begin="0.15s" repeatCount="indefinite"/>
          </rect>
          <rect x="14" y="0" width="4" height="18" rx="2" fill="#E5C9D7">
            <animate attributeName="height" values="6;22;6" dur="1s" begin="0.3s" repeatCount="indefinite"/>
            <animate attributeName="y" values="18;2;18" dur="1s" begin="0.3s" repeatCount="indefinite"/>
          </rect>
        </g>
        <g transform="translate(-40,-45)">
          <rect x="0" y="0" width="135" height="34" rx="17" fill="rgba(26,47,63,0.92)" stroke="rgba(229,201,215,0.55)" stroke-width="1.2"/>
          <text x="67" y="21" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="700" fill="#FFFFFF">I understand! &#x1F4AC;</text>
          <polygon points="40,34 50,34 38,42" fill="rgba(26,47,63,0.92)"/>
          <animateTransform attributeName="transform" type="translate" values="-40,-45;-40,-52;-40,-45" dur="3.5s" repeatCount="indefinite"/>
        </g>
        <g transform="translate(10,280)">
          <rect x="0" y="0" width="105" height="26" rx="13" fill="rgba(26,47,63,0.85)" stroke="rgba(196,140,179,0.4)" stroke-width="1"/>
          <circle cx="14" cy="13" r="4" fill="#C48CB3"/>
          <text x="58" y="17" text-anchor="middle" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#FFFFFF">RECEIVER</text>
        </g>
      </g>
    </svg>
    '''


DARK_THEME_CSS = """
<style>
.stApp {
    background: linear-gradient(150deg, #0e1117 0%, #141720 50%, #1a1f2e 100%) !important;
    color: #e0e0e0 !important;
}
.main .block-container { background: transparent !important; }
.clay-card, .pro-card, .ai-panel, .glass-card {
    background: #1e2535 !important;
    border-color: rgba(131,166,206,0.2) !important;
    color: #e0e0e0 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #1e2535 !important;
    border-color: rgba(131,166,206,0.2) !important;
}
.stTabs [data-baseweb="tab"] { color: #a0b4c8 !important; }
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: #1e2535 !important;
    color: #e0e0e0 !important;
    border-color: rgba(131,166,206,0.3) !important;
}
[data-testid="metric-container"] {
    background: #1e2535 !important;
    border-color: rgba(131,166,206,0.2) !important;
}
[data-testid="metric-container"] label { color: #a0b4c8 !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #e0e0e0 !important; }
.sentence-box {
    background: #1e2535 !important;
    color: #e0e0e0 !important;
    border-color: rgba(131,166,206,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e2535 !important;
    color: #e0e0e0 !important;
    border-color: rgba(131,166,206,0.4) !important;
}
p, span, label, div { color: inherit; }
</style>
"""


def inject_css(theme: str = "dark", large_font: bool = False, high_contrast: bool = False) -> None:
    """
    Inject the SignFlow design system CSS into the Streamlit page.

    Args:
        theme:         'dark' or 'light'.
        large_font:    If True, injects LARGE_FONT_CSS for accessibility.
        high_contrast: If True, injects HIGH_CONTRAST_CSS for accessibility.
    """
    st.markdown(CLAY_CSS, unsafe_allow_html=True)
    if theme == "dark":
        st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    if high_contrast:
        st.markdown(HIGH_CONTRAST_CSS, unsafe_allow_html=True)
    if large_font:
        st.markdown(LARGE_FONT_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Component Helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_glass_card(content_html: str, variant: str = "") -> None:
    """
    Render content inside a clay/glass card.

    Args:
        content_html: Raw HTML string to place inside the card.
        variant:      One of '' | 'blue' | 'violet' | 'cyan'.
    """
    cls = f"clay-card clay-card-{variant}" if variant else "clay-card"
    st.markdown(f'<div class="{cls}">{content_html}</div>', unsafe_allow_html=True)


def render_ai_panel(gesture: str, meaning: str, status: str, confidence: float, speaking: bool) -> None:
    """Render the AI assistant detection panel with SignFlow styling."""
    speak_badge = '<span class="badge badge-blue">🔊 Speaking…</span>' if speaking else ""
    conf_pct = f"{confidence * 100:.0f}%"
    if confidence >= 0.75:
        conf_color = "#83A6CE"
        fill_class = "conf-fill-high"
    elif confidence >= 0.5:
        conf_color = "#d97706"
        fill_class = "conf-fill-medium"
    else:
        conf_color = "#C48CB3"
        fill_class = "conf-fill-low"

    bar_w = int(confidence * 100)
    robot_svg = svg_ai_robot()

    html = f"""
    <div class="ai-panel">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
        <div style="flex-shrink:0;">{robot_svg}</div>
        <div style="flex:1;min-width:0;">
          <div style="font-size:0.68rem;color:#6b8aaa;font-weight:700;text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:6px;">Detected Gesture</div>
          <div class="gesture-big">{gesture}</div>
          <div class="gesture-meaning">{meaning}</div>
        </div>
      </div>
      <div style="background:rgba(40,65,85,0.04);border-radius:14px;padding:14px 16px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
          <span style="color:#6b8aaa;font-size:0.82rem;font-weight:600;">Status</span>
          <span style="color:#284155;font-size:0.82rem;font-weight:700;">{status}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="color:#6b8aaa;font-size:0.82rem;font-weight:600;">Confidence</span>
          <span style="color:{conf_color};font-size:0.92rem;font-weight:800;">{conf_pct}</span>
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
            val_color = "#83A6CE"
        elif pct >= 50:
            fill_cls = "conf-fill-medium"
            val_color = "#f59e0b"
        else:
            fill_cls = "conf-fill-low"
            val_color = "#C48CB3"
        rows.append(f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.83rem;color:#284155;font-weight:600;">{label}</span>
            <span style="font-size:0.83rem;font-weight:800;color:{val_color};">{pct}%</span>
          </div>
          <div class="conf-track">
            <div class="{fill_cls}" style="width:{pct}%;transition:width 0.35s ease;"></div>
          </div>
        </div>""")
    st.markdown(
        '<div class="clay-card">'
        '<div class="section-title-dark" style="margin-bottom:14px;">Confidence Meter</div>'
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
    """Render a pill-shaped status badge. level: 'green'|'blue'|'yellow'|'red'|'violet'"""
    st.markdown(f'<span class="badge badge-{level}">{text}</span>', unsafe_allow_html=True)


def render_detection_status(has_hand: bool, gesture, confidence: float) -> None:
    """Render a contextual detection-state badge above the camera feed."""
    if not has_hand:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-yellow">🖐 Scanning for hand…</span></div>'
    elif gesture and confidence >= 0.75:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-green">✅ Gesture Recognised</span></div>'
    elif gesture:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-blue">🔍 Recognising…</span></div>'
    else:
        html = '<div style="text-align:center;padding:8px;"><span class="badge badge-blue">🖐 Hand Detected</span></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_suggestions(suggestions: list) -> None:
    """Render a row of gesture-suggestion chips."""
    if not suggestions:
        return
    chips = " ".join(
        f'<span class="gesture-chip">○ {s}</span>'
        for s in suggestions[1:]
    )
    first = suggestions[0] if suggestions else ""
    st.markdown(
        f'<div class="clay-card">'
        f'<div style="font-size:0.78rem;color:#6b8aaa;font-weight:600;margin-bottom:10px;">💡 Did you mean?</div>'
        f'<span class="gesture-chip" style="border-color:rgba(131,166,206,0.5);'
        f'background:rgba(131,166,206,0.15);">✓ {first}</span>{chips}</div>',
        unsafe_allow_html=True,
    )


def render_subtitle_strip(text: str) -> None:
    """Render a subtitle strip below the camera panel."""
    if text:
        st.markdown(f'<div class="subtitle-strip">📝 {text}</div>', unsafe_allow_html=True)


def render_loading_sequence(stage: str) -> None:
    """
    Render a loading indicator for a named pipeline stage.

    Args:
        stage: One of 'model' | 'camera' | 'translator' | 'ready', or any custom string.
    """
    stages = {
        "model":      "🧠 Loading Model…",
        "camera":     "📷 Initialising Camera…",
        "translator": "🌍 Loading Translator…",
        "ready":      "✅ Ready",
    }
    msg = stages.get(stage, stage)
    color = "#284155" if stage == "ready" else "#83A6CE"
    st.markdown(
        f'<div class="clay-loading" style="color:{color};">{msg}</div>',
        unsafe_allow_html=True,
    )


def camera_placeholder_html() -> str:
    """
    Return an HTML string for an empty camera-feed placeholder.
    Used before the webcam is started.
    """
    return (
        "<div style='"
        "background: linear-gradient(135deg, #1a2f3f 0%, #284155 100%);"
        "border: 2px dashed rgba(131,166,206,0.35);"
        "border-radius: 22px;"
        "height: 380px;"
        "display: flex;"
        "align-items: center;"
        "justify-content: center;"
        "flex-direction: column;"
        "gap: 14px;"
        "box-shadow: 0 4px 20px rgba(40,65,85,0.15);"
        "position: relative;"
        "overflow: hidden;"
        "'>"
        # Decorative blobs inside the placeholder
        "<div style='"
        "position:absolute;top:-40px;right:-40px;"
        "width:180px;height:180px;"
        "background:radial-gradient(circle,rgba(196,140,179,0.15) 0%,transparent 70%);"
        "border-radius:50%;"
        "pointer-events:none;"
        "'></div>"
        "<div style='"
        "position:absolute;bottom:-30px;left:-30px;"
        "width:150px;height:150px;"
        "background:radial-gradient(circle,rgba(131,166,206,0.1) 0%,transparent 70%);"
        "border-radius:50%;"
        "pointer-events:none;"
        "'></div>"
        "<span style='font-size:3rem;'>📷</span>"
        "<span style='"
        "color:#FFFFFF;"
        "font-weight:700;"
        "font-size:1.05rem;"
        "font-family:Inter,sans-serif;"
        "'>Camera feed will appear here</span>"
        "<span style='"
        "color:rgba(255,255,255,0.5);"
        "font-size:0.84rem;"
        "font-family:Inter,sans-serif;"
        "'>Click ▶ Start Camera to begin</span>"
        "</div>"
    )
