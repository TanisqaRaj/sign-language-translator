# =============================================================================
# app_pages/home.py  –  Premium 3D Animated Landing Page
# Shared by both app.py and app_cloud.py.
# =============================================================================

import streamlit as st
import streamlit.components.v1 as _components

from utils.ui_components import (
    HOME_PAGE_CSS,
    HIDE_SIDEBAR_CSS,
    svg_communication_scene,
    svg_hand_3d,
    svg_ai_robot,
)


# ── AI Flow SVG (inline, no external dependency) ─────────────────────────────
def _svg_ai_flow() -> str:
    """Animated pipeline: Hand → AI → Text → Voice."""
    return """
<svg viewBox="0 0 760 160" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:760px;overflow:visible;">
  <defs>
    <linearGradient id="flowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#83A6CE"/>
      <stop offset="50%"  stop-color="#C48CB3"/>
      <stop offset="100%" stop-color="#E5C9D7"/>
    </linearGradient>
    <filter id="glow2">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="url(#flowGrad)" opacity="0.7"/>
    </marker>
  </defs>

  <!-- Connecting lines -->
  <line x1="138" y1="80" x2="228" y2="80"
        stroke="url(#flowGrad)" stroke-width="2" stroke-dasharray="5,4" opacity="0.6"
        marker-end="url(#arrow)">
    <animate attributeName="stroke-dashoffset" from="0" to="-18" dur="1.2s" repeatCount="indefinite"/>
  </line>
  <line x1="328" y1="80" x2="418" y2="80"
        stroke="url(#flowGrad)" stroke-width="2" stroke-dasharray="5,4" opacity="0.6"
        marker-end="url(#arrow)">
    <animate attributeName="stroke-dashoffset" from="0" to="-18" dur="1.2s" repeatCount="indefinite"/>
  </line>
  <line x1="518" y1="80" x2="608" y2="80"
        stroke="url(#flowGrad)" stroke-width="2" stroke-dasharray="5,4" opacity="0.6"
        marker-end="url(#arrow)">
    <animate attributeName="stroke-dashoffset" from="0" to="-18" dur="1.2s" repeatCount="indefinite"/>
  </line>

  <!-- Travelling particle -->
  <circle r="5" fill="#83A6CE" filter="url(#glow2)" opacity="0.9">
    <animateMotion dur="2.4s" repeatCount="indefinite" calcMode="linear">
      <mpath><path d="M138,80 L628,80"/></mpath>
    </animateMotion>
    <animate attributeName="opacity" values="0;1;1;0" dur="2.4s" repeatCount="indefinite"/>
  </circle>

  <!-- Node 1: Hand -->
  <circle cx="80" cy="80" r="52" fill="rgba(131,166,206,0.12)"
          stroke="rgba(131,166,206,0.4)" stroke-width="1.5">
    <animate attributeName="r" values="52;55;52" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="80" y="68" text-anchor="middle" font-size="28">🖐</text>
  <text x="80" y="110" text-anchor="middle" fill="rgba(255,255,255,0.85)"
        font-size="11" font-weight="700" letter-spacing="1">SIGN</text>

  <!-- Node 2: AI -->
  <circle cx="280" cy="80" r="52" fill="rgba(196,140,179,0.12)"
          stroke="rgba(196,140,179,0.4)" stroke-width="1.5">
    <animate attributeName="r" values="52;56;52" dur="2.5s" repeatCount="indefinite"/>
  </circle>
  <circle r="5" fill="#C48CB3" opacity="0.8">
    <animateMotion dur="2s" repeatCount="indefinite">
      <mpath><path d="M280,28 A52,52 0 1,1 279.9,28"/></mpath>
    </animateMotion>
  </circle>
  <text x="280" y="68" text-anchor="middle" font-size="28">🧠</text>
  <text x="280" y="110" text-anchor="middle" fill="rgba(255,255,255,0.85)"
        font-size="11" font-weight="700" letter-spacing="1">AI CORE</text>

  <!-- Node 3: Text -->
  <circle cx="480" cy="80" r="52" fill="rgba(131,166,206,0.12)"
          stroke="rgba(131,166,206,0.4)" stroke-width="1.5">
    <animate attributeName="r" values="52;55;52" dur="3.5s" repeatCount="indefinite"/>
  </circle>
  <text x="480" y="68" text-anchor="middle" font-size="28">📝</text>
  <text x="480" y="110" text-anchor="middle" fill="rgba(255,255,255,0.85)"
        font-size="11" font-weight="700" letter-spacing="1">TEXT</text>

  <!-- Node 4: Voice -->
  <circle cx="680" cy="80" r="52" fill="rgba(229,201,215,0.12)"
          stroke="rgba(229,201,215,0.4)" stroke-width="1.5">
    <animate attributeName="r" values="52;56;52" dur="2.8s" repeatCount="indefinite"/>
  </circle>
  <circle cx="680" cy="80" r="58" fill="none" stroke="rgba(229,201,215,0.3)" stroke-width="1">
    <animate attributeName="r" values="52;70;52" dur="1.8s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.5;0;0.5" dur="1.8s" repeatCount="indefinite"/>
  </circle>
  <text x="680" y="68" text-anchor="middle" font-size="28">🔊</text>
  <text x="680" y="110" text-anchor="middle" fill="rgba(255,255,255,0.85)"
        font-size="11" font-weight="700" letter-spacing="1">VOICE</text>
</svg>
"""


def show_home() -> None:
    """Render the premium SignFlow Home / landing page."""

    # ── 1. Inject CSS ─────────────────────────────────────────────────────────
    # Hide sidebar + Streamlit's own header bar (hamburger/deploy) so only the
    # glass navbar shows — no double navbar.
    # Note: we target [data-testid="stFooter"] not <footer> so our custom footer
    # HTML element is not hidden.
    st.markdown("""
<style>
/* Hide sidebar on home page */
section[data-testid="stSidebar"]   { display: none !important; }
[data-testid="stSidebarNav"]        { display: none !important; }
/* Hide Streamlit's built-in header bar (hamburger + Deploy button) */
header[data-testid="stHeader"]      { display: none !important; }
#MainMenu                           { display: none !important; }
/* Hide Streamlit's own footer bar (not our custom footer element) */
[data-testid="stFooter"]            { display: none !important; }
/* Remove top padding Streamlit adds for the header */
.main .block-container              { padding-top: 0 !important; }
/* Hide the nav buttons row (they are off-screen functional triggers) */
div[data-testid="stHorizontalBlock"]:has(button[data-testid="baseButton-secondary"][kind="secondary"]) {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)
    st.markdown(f"<style>{HOME_PAGE_CSS}</style>", unsafe_allow_html=True)

    # ── 2. Glass Navbar ────────────────────────────────────────────────────────
    _current        = st.session_state.get("page", "🏠 Home")
    _home_active    = " active" if _current == "🏠 Home"            else ""
    _live_active    = " active" if _current == "📷 Live Translator" else ""
    _history_active = " active" if _current == "📜 History"         else ""
    _about_active   = " active" if _current == "ℹ️ About"           else ""

    st.markdown(f"""
<div class="glass-navbar" id="main-navbar">
  <div class="nav-logo">🤙 SignFlow</div>
  <div class="nav-links">
    <div class="nav-link{_home_active}" id="nl-home">Home</div>
    <div class="nav-link{_live_active}" id="nl-live">Live Translator</div>
    <div class="nav-link{_history_active}" id="nl-history">History</div>
    <div class="nav-link{_about_active}" id="nl-about">About</div>
  </div>
  <div class="nav-cta" id="nl-cta">Start Translating →</div>
</div>
""", unsafe_allow_html=True)

    # ── Hidden functional Streamlit buttons (triggered by navbar JS) ──────────
    # These are hidden off-screen; JavaScript clicks them when a nav link is tapped.
    _hb_col1, _hb_col2, _hb_col3, _hb_col4 = st.columns(4)
    with _hb_col1:
        if st.button("📷 Live Translator", key="nav_live", type="secondary"):
            st.session_state["page"] = "📷 Live Translator"
            st.rerun()
    with _hb_col2:
        if st.button("📜 History", key="nav_history", type="secondary"):
            st.session_state["page"] = "📜 History"
            st.rerun()
    with _hb_col3:
        if st.button("ℹ️ About", key="nav_about", type="secondary"):
            st.session_state["page"] = "ℹ️ About"
            st.rerun()
    with _hb_col4:
        if st.button("▶ Start Translating", key="nav_cta", type="primary"):
            st.session_state["page"] = "📷 Live Translator"
            st.rerun()

    # Navbar JS — wires the visual navbar divs to the hidden Streamlit buttons above.
    # Must use _components.html so the <script> actually executes.
    _components.html("""
<script>
(function() {
  function findBtn(text) {
    var doc = window.parent.document;
    return Array.from(doc.querySelectorAll('button')).find(function(b) {
      return b.innerText.trim().includes(text);
    });
  }
  function initNavbar() {
    var doc = window.parent.document;
    var navbar = doc.getElementById('main-navbar');
    if (!navbar) { setTimeout(initNavbar, 300); return; }
    window.parent.addEventListener('scroll', function() {
      navbar.classList.toggle('scrolled', window.parent.scrollY > 50);
    }, { passive: true });
    var map = [
      ['nl-home',    'Home'],
      ['nl-live',    'Live Translator'],
      ['nl-history', 'History'],
      ['nl-about',   'About'],
      ['nl-cta',     'Start Translating'],
    ];
    map.forEach(function(pair) {
      var el = doc.getElementById(pair[0]);
      if (el) {
        el.style.cursor = 'pointer';
        el.addEventListener('click', function() {
          var btn = findBtn(pair[1]);
          if (btn) btn.click();
        });
      }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNavbar);
  } else {
    initNavbar();
  }
})();
</script>
""", height=0, scrolling=False)

    # ── 3. Hero Section ───────────────────────────────────────────────────────
    # The hero text/stats block is rendered via st.markdown.
    # The SVG scene goes through _components.html (st.markdown strips <svg> tags).
    _scene_svg = svg_communication_scene()

    st.markdown("""
<div class="home-hero-section">
  <div class="hero-bg">
    <div class="hero-blob-1"></div>
    <div class="hero-blob-2"></div>
    <div class="hero-blob-3"></div>
    <div style="position:absolute;top:18%;left:8%;width:6px;height:6px;border-radius:50%;
                background:#83A6CE;opacity:0.5;animation:particle-drift 6s ease-in-out infinite;"></div>
    <div style="position:absolute;top:65%;left:15%;width:4px;height:4px;border-radius:50%;
                background:#C48CB3;opacity:0.4;animation:particle-drift 8s ease-in-out infinite 1s;"></div>
    <div style="position:absolute;top:30%;right:12%;width:5px;height:5px;border-radius:50%;
                background:#E5C9D7;opacity:0.5;animation:particle-drift 7s ease-in-out infinite 2s;"></div>
    <div style="position:absolute;top:75%;right:20%;width:3px;height:3px;border-radius:50%;
                background:#83A6CE;opacity:0.4;animation:particle-drift 9s ease-in-out infinite 0.5s;"></div>
    <div style="position:absolute;top:45%;left:50%;width:4px;height:4px;border-radius:50%;
                background:#C48CB3;opacity:0.35;animation:particle-drift 5s ease-in-out infinite 3s;"></div>
  </div>
  <div style="display:flex;align-items:center;gap:60px;max-width:1100px;
              width:100%;flex-wrap:wrap;position:relative;z-index:1;">
    <div style="flex:1;min-width:280px;animation:fade-in-up 0.8s ease both;">
      <div style="font-size:0.75rem;font-weight:700;letter-spacing:3px;
                  text-transform:uppercase;color:#C48CB3;margin-bottom:20px;">
        AI Sign Language Technology
      </div>
      <h1 style="font-size:3.5rem;font-weight:900;color:#FFFFFF;line-height:1.05;
                 margin:0 0 20px;letter-spacing:-1px;">
        Communication<br>
        <span style="background:linear-gradient(135deg,#83A6CE,#C48CB3,#E5C9D7);
                     background-size:200% auto;
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     animation:gradient-shift 4s ease infinite;">
          Without Barriers.
        </span>
      </h1>
      <p style="font-size:1.1rem;color:rgba(255,255,255,0.7);line-height:1.7;
                margin:0 0 36px;max-width:440px;font-weight:400;">
        Real-time AI translation of sign language into text and speech.<br>
        <span style="color:rgba(255,255,255,0.5);font-size:0.9rem;">
          25+ FPS &nbsp;·&nbsp; MediaPipe &nbsp;·&nbsp; TFLite &nbsp;·&nbsp; Zero barriers.
        </span>
      </p>
      <div style="display:flex;gap:28px;flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:1.8rem;font-weight:900;color:#83A6CE;line-height:1;">25+</div>
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);letter-spacing:1px;
                      text-transform:uppercase;margin-top:2px;">FPS Live</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:1.8rem;font-weight:900;color:#C48CB3;line-height:1;">95%+</div>
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);letter-spacing:1px;
                      text-transform:uppercase;margin-top:2px;">Accuracy</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:1.8rem;font-weight:900;color:#E5C9D7;line-height:1;">10</div>
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);letter-spacing:1px;
                      text-transform:uppercase;margin-top:2px;">Gestures</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:1.8rem;font-weight:900;color:#83A6CE;line-height:1;">13</div>
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);letter-spacing:1px;
                      text-transform:uppercase;margin-top:2px;">Languages</div>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # SVG scene — must use components.html, st.markdown strips <svg> tags
    _components.html(
        f"""<div style="display:flex;justify-content:center;align-items:center;
                        padding:0;background:transparent;">{_scene_svg}</div>""",
        height=320,
        scrolling=False,
    )

    # Hero CTA button
    _cl, _cc, _cr = st.columns([2, 1, 2])
    with _cc:
        if st.button("▶ Start Translating", type="primary",
                     use_container_width=True, key="hero_cta"):
            st.session_state["page"] = "📷 Live Translator"
            st.rerun()

    # ── 4. How It Works ───────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:80px 0 60px;text-align:center;">
  <div style="font-size:0.75rem;font-weight:700;letter-spacing:3px;color:#C48CB3;
              text-transform:uppercase;margin-bottom:16px;">Simple Process</div>
  <h2 style="font-size:2.5rem;font-weight:900;margin:0 0 12px;
             background:linear-gradient(135deg,#284155,#83A6CE);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    How It Works
  </h2>
  <p style="color:#6b8aaa;font-size:1rem;max-width:500px;margin:0 auto;">
    Three seamless steps from gesture to understanding.
  </p>
</div>
""", unsafe_allow_html=True)

    _sc1, _sc2, _sc3 = st.columns(3, gap="large")
    with _sc1:
        st.markdown("""
<div class="step-card-3d">
  <div class="step-number">01</div>
  <div class="step-icon-wrap">🖐</div>
  <div class="step-title">Sign</div>
  <div class="step-desc">
    Perform a hand gesture in front of your camera. Our AI watches every
    movement in real time, detecting 21 precise hand landmarks.
  </div>
</div>""", unsafe_allow_html=True)
    with _sc2:
        st.markdown("""
<div class="step-card-3d">
  <div class="step-number">02</div>
  <div class="step-icon-wrap">🧠</div>
  <div class="step-title">Recognize</div>
  <div class="step-desc">
    AI detects hand landmarks in real-time using MediaPipe, then classifies
    the gesture with a TFLite neural network at &lt;8 ms latency.
  </div>
</div>""", unsafe_allow_html=True)
    with _sc3:
        st.markdown("""
<div class="step-card-3d">
  <div class="step-number">03</div>
  <div class="step-icon-wrap">🔊</div>
  <div class="step-title">Translate</div>
  <div class="step-desc">
    Sign becomes text and voice output instantly. Sentences are built
    word by word and spoken aloud — bridging communication effortlessly.
  </div>
</div>""", unsafe_allow_html=True)

    # ── 5. AI Translation Flow ────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#1a2f40 0%,#284155 50%,#1a2f40 100%);
            border-radius:32px;padding:60px 48px 32px;margin:20px 0;
            position:relative;overflow:hidden;border:1px solid rgba(131,166,206,0.15);">
  <div style="position:relative;z-index:1;text-align:center;margin-bottom:36px;">
    <div style="font-size:0.75rem;font-weight:700;letter-spacing:3px;color:#C48CB3;
                text-transform:uppercase;margin-bottom:14px;">Pipeline</div>
    <h2 style="font-size:2.2rem;font-weight:900;color:#FFFFFF;margin:0 0 10px;">
      AI Translation Flow
    </h2>
    <p style="color:rgba(255,255,255,0.55);font-size:0.95rem;max-width:460px;margin:0 auto;">
      From your hand gesture to spoken words — all within a single frame.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

    # Flow SVG — must use components.html, st.markdown strips <svg> tags
    _components.html(
        f"""<div style="display:flex;justify-content:center;padding:16px 24px 32px;
                        background:linear-gradient(135deg,#1a2f40,#284155);
                        border-radius:0 0 32px 32px;margin-top:-4px;">
              {_svg_ai_flow()}
            </div>""",
        height=200,
        scrolling=False,
    )

    # ── 6. Features ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:60px 0 40px;text-align:center;">
  <div style="font-size:0.75rem;font-weight:700;letter-spacing:3px;color:#83A6CE;
              text-transform:uppercase;margin-bottom:16px;">Capabilities</div>
  <h2 style="font-size:2.5rem;font-weight:900;margin:0 0 12px;
             background:linear-gradient(135deg,#284155,#C48CB3);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Everything You Need
  </h2>
  <p style="color:#6b8aaa;font-size:1rem;max-width:500px;margin:0 auto;">
    A complete sign language translation suite built for real-world use.
  </p>
</div>
""", unsafe_allow_html=True)

    _features = [
        ("🖐", "Real-time Sign Recognition", "Live gesture detection at 25+ FPS with MediaPipe hand tracking.", "#83A6CE"),
        ("📝", "Word Recognition",           "Full word-level gesture translation with sentence builder.",        "#C48CB3"),
        ("🔤", "Character Recognition",      "Spell out anything — A–Z and 0–9 fingerspelling support.",         "#83A6CE"),
        ("💬", "Text Output",                "Clean on-screen text output with confidence scores.",               "#C48CB3"),
        ("🔊", "Voice Output",               "Text-to-Speech reads translations aloud.",                          "#83A6CE"),
        ("📺", "Subtitle Support",           "Cinema-style subtitle overlay synced with live translation.",       "#C48CB3"),
        ("🌍", "13 Languages",               "Interface and voice output in 13 different languages.",             "#83A6CE"),
        ("📊", "Analytics Dashboard",        "Track usage, accuracy trends, and session statistics.",             "#C48CB3"),
    ]
    _fr1 = st.columns(4, gap="medium")
    _fr2 = st.columns(4, gap="medium")
    for _idx, (_icon, _title, _desc, _accent) in enumerate(_features):
        _col = _fr1[_idx] if _idx < 4 else _fr2[_idx - 4]
        with _col:
            st.markdown(f"""
<div class="feature-card-premium">
  <div class="feature-icon-wrap" style="color:{_accent};">{_icon}</div>
  <div style="font-weight:800;font-size:0.92rem;color:#284155;margin-bottom:6px;">{_title}</div>
  <div style="font-size:0.78rem;color:#6b8aaa;line-height:1.55;">{_desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── 7. About Section ──────────────────────────────────────────────────────
    _tech_pills = "".join(
        f'<span style="background:rgba(131,166,206,0.12);border:1px solid rgba(131,166,206,0.3);'
        f'border-radius:20px;padding:6px 14px;font-size:0.78rem;font-weight:600;color:#284155;">{t}</span>'
        for t in ["MediaPipe", "TFLite", "Streamlit", "OpenCV", "pyttsx3", "Python 3.10", "Keras", "NumPy"]
    )
    # svg_hand_3d / svg_ai_robot are small SVGs that Streamlit allows inline
    _hand_svg  = svg_hand_3d()
    _robot_svg = svg_ai_robot()
    st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(131,166,206,0.06),rgba(196,140,179,0.06));
            border:1px solid rgba(131,166,206,0.15);border-radius:32px;
            padding:60px 48px;margin:20px 0;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,#83A6CE,#C48CB3,#E5C9D7);
              border-radius:32px 32px 0 0;"></div>
  <div style="display:flex;align-items:center;gap:60px;flex-wrap:wrap;">
    <div style="flex:1;min-width:280px;">
      <div style="font-size:0.75rem;font-weight:700;letter-spacing:3px;color:#C48CB3;
                  text-transform:uppercase;margin-bottom:16px;">About This Project</div>
      <h2 style="font-size:2.2rem;font-weight:900;margin:0 0 20px;
                 background:linear-gradient(135deg,#284155,#83A6CE);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Technology Designed<br>for Communication
      </h2>
      <p style="color:#6b8aaa;font-size:0.95rem;line-height:1.8;margin:0 0 18px;">
        SignFlow is a final-year computer vision project that translates hand gestures
        into text and speech in real time. It combines <strong style="color:#284155;">
        Google MediaPipe</strong> for precise 21-point hand landmark detection,
        a lightweight <strong style="color:#284155;">TFLite Dense Neural Network</strong>
        for gesture classification, and <strong style="color:#284155;">pyttsx3</strong>
        for natural-sounding voice output.
      </p>
      <p style="color:#6b8aaa;font-size:0.95rem;line-height:1.8;margin:0 0 28px;">
        Built to run entirely on CPU — no GPU required — SignFlow achieves
        25–35 FPS on a standard laptop, with &lt;8 ms inference latency and
        over 95% test accuracy across 10 ASL-inspired gestures.
      </p>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">{_tech_pills}</div>
    </div>
    <div style="flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:24px;">
      <div style="animation:float 4s ease-in-out infinite;">{_hand_svg}</div>
      <div style="animation:float 4s ease-in-out infinite 1s;">{_robot_svg}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 8. Bottom CTA Banner ──────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#284155,#3a5a75,#284155);
            border-radius:28px;padding:52px 48px;margin:20px 0;
            text-align:center;position:relative;overflow:hidden;">
  <div style="position:absolute;inset:0;background:
    radial-gradient(ellipse 60% 80% at 20% 50%,rgba(131,166,206,0.2),transparent),
    radial-gradient(ellipse 60% 80% at 80% 50%,rgba(196,140,179,0.15),transparent);
    pointer-events:none;"></div>
  <div style="position:relative;z-index:1;">
    <div style="font-size:0.75rem;font-weight:700;letter-spacing:3px;color:#C48CB3;
                text-transform:uppercase;margin-bottom:16px;">Ready?</div>
    <h2 style="font-size:2.4rem;font-weight:900;color:#FFFFFF;margin:0 0 16px;">
      Start Communicating Now
    </h2>
    <p style="color:rgba(255,255,255,0.65);font-size:1rem;max-width:460px;margin:0 auto;">
      Open your camera, show your hand, and let SignFlow do the rest.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

    _cl2, _cc2, _cr2 = st.columns([2, 1, 2])
    with _cc2:
        if st.button("🚀 Launch Translator", type="primary",
                     use_container_width=True, key="bottom_cta"):
            st.session_state["page"] = "📷 Live Translator"
            st.rerun()

    # ── 9. Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(180deg,transparent,rgba(20,36,50,0.6));
            border-top:1px solid rgba(131,166,206,0.15);
            padding:40px 0 20px;text-align:center;margin-top:40px;">
  <div style="font-size:1.6rem;margin-bottom:10px;">🤙</div>
  <div style="font-size:1rem;font-weight:800;color:#FFFFFF;margin-bottom:6px;
              letter-spacing:1px;">SignFlow</div>
  <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:20px;">
    AI-Powered Sign Language Translation
  </div>
  <div style="display:flex;justify-content:center;gap:32px;flex-wrap:wrap;margin-bottom:24px;">
    <span style="font-size:0.78rem;color:rgba(255,255,255,0.4);">
      MediaPipe &nbsp;·&nbsp; TFLite &nbsp;·&nbsp; Streamlit
    </span>
    <span style="font-size:0.78rem;color:rgba(255,255,255,0.4);">Python 3.10+</span>
    <span style="font-size:0.78rem;color:rgba(255,255,255,0.4);">
      25+ FPS &nbsp;·&nbsp; &lt;8 ms latency
    </span>
  </div>
  <div style="font-size:0.72rem;color:rgba(255,255,255,0.25);">
    Final Year Computer Vision Project &nbsp;·&nbsp; Built with ❤️ for accessibility
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 10. Audio Toggle ──────────────────────────────────────────────────────
    if "audio_on" not in st.session_state:
        st.session_state["audio_on"] = True

    _audio_icon  = "🔊" if st.session_state["audio_on"] else "🔇"
    _audio_label = "Audio ON" if st.session_state["audio_on"] else "Audio OFF"

    st.markdown(f"""
<div style="position:fixed;bottom:32px;right:32px;z-index:10000;pointer-events:none;">
  <div class="audio-btn" id="audio-toggle">{_audio_icon}</div>
</div>
""", unsafe_allow_html=True)

    _, _, _acol = st.columns([4, 1, 1])
    with _acol:
        if st.button(f"{_audio_icon} {_audio_label}", key="audio_toggle_btn",
                     type="secondary", use_container_width=True):
            st.session_state["audio_on"] = not st.session_state["audio_on"]
            st.rerun()
