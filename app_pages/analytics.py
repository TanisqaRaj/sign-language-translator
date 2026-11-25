# =============================================================================
# pages/analytics.py  –  Analytics Dashboard page
# Shared by both app.py and app_cloud.py.
#
# The two apps store SessionAnalytics under different session-state keys
# ('analytics_obj' vs 'analytics').  Pass the correct key via analytics_key.
#
# app.py uses plotly for charts; app_cloud.py uses st.line_chart.
# Both are supported here via the use_plotly flag.
# =============================================================================

import streamlit as st

from utils.analytics import SessionAnalytics
from utils.ui_components import svg_analytics

# ---------------------------------------------------------------------------
# Premium page-level CSS injected once per render
# ---------------------------------------------------------------------------
_ANALYTICS_CSS = """
<style>
/* ── Hero banner ──────────────────────────────────────────────────── */
.analytics-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0d1b2a 0%, #0f172a 40%, #1a103a 100%);
    border: 1px solid rgba(131,166,206,0.18);
    border-radius: 20px;
    padding: 32px 32px 28px;
    margin-bottom: 28px;
    animation: fade-in-up 0.5s ease both;
}
.analytics-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%,
                    rgba(131,166,206,0.12) 0%, transparent 70%),
                radial-gradient(ellipse 40% 60% at 10% 90%,
                    rgba(196,140,179,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.analytics-hero-inner {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 20px;
}
.analytics-hero-text .hero-page-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0 0%, #83A6CE 50%, #C48CB3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 6px;
}
.analytics-hero-text .hero-page-sub {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.analytics-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(131,166,206,0.12);
    border: 1px solid rgba(131,166,206,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #83A6CE;
    font-weight: 600;
    margin-top: 8px;
    letter-spacing: 0.04em;
}

/* ── Premium KPI cards ────────────────────────────────────────────── */
.kpi-premium, .clay-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, rgba(15,23,42,0.95) 0%, rgba(22,33,62,0.9) 100%);
    border: 1px solid rgba(131,166,206,0.18);
    border-radius: 16px;
    padding: 18px 16px 16px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    animation: fade-in-up 0.45s ease both;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.kpi-premium::after, .clay-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
    background: var(--kpi-accent, linear-gradient(90deg, #83A6CE, #C48CB3));
}
.kpi-premium:hover, .clay-card:hover {
    transform: translateY(-3px);
    border-color: rgba(131,166,206,0.35);
    box-shadow: 0 12px 32px rgba(0,0,0,0.4), 0 0 20px rgba(131,166,206,0.08);
}
.kpi-label {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.kpi-icon {
    font-size: 1.5rem;
    opacity: 0.65;
    float: right;
    margin-top: -4px;
}

/* ── Section titles ───────────────────────────────────────────────── */
.analytics-section-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #83A6CE;
    margin: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.analytics-section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(131,166,206,0.25), transparent);
    border-radius: 1px;
}

/* ── Chart container ──────────────────────────────────────────────── */
.chart-glass {
    background: linear-gradient(145deg, rgba(8,13,28,0.85) 0%, rgba(15,23,42,0.9) 100%);
    border: 1px solid rgba(131,166,206,0.14);
    border-radius: 18px;
    padding: 18px 14px 10px;
    margin-bottom: 16px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    animation: fade-in-up 0.5s ease 0.1s both;
    overflow: hidden;
}
.chart-glass-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* ── Confidence / frequency cards ────────────────────────────────── */
.conf-split-card {
    background: linear-gradient(145deg, rgba(8,13,28,0.9) 0%, rgba(15,23,42,0.95) 100%);
    border: 1px solid rgba(131,166,206,0.15);
    border-radius: 18px;
    padding: 20px;
    animation: fade-in-up 0.5s ease both;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* ── Camera status pill ───────────────────────────────────────────── */
.cam-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(131,166,206,0.15);
    border-radius: 24px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* ── Reset button overrides ───────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(196,140,179,0.08) !important;
    border: 1px solid rgba(196,140,179,0.28) !important;
    color: #C48CB3 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(196,140,179,0.18) !important;
    box-shadow: 0 0 16px rgba(196,140,179,0.2) !important;
}

/* ── Animations ───────────────────────────────────────────────────── */
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


def show_analytics(
    analytics_key: str = "analytics_obj",
    history_key: str = "history_obj",
    use_plotly: bool = True,
) -> None:
    """
    Render the Analytics Dashboard page.

    Parameters
    ----------
    analytics_key : str
        Session-state key holding the SessionAnalytics instance.
        app.py  → 'analytics_obj'
        app_cloud.py → 'analytics'
    history_key : str
        Session-state key holding the GestureHistory instance.
        app.py  → 'history_obj'
        app_cloud.py → 'history'
    use_plotly : bool
        True  → render charts with plotly (app.py).
        False → render charts with st.line_chart (app_cloud.py).
    """
    # Inject premium CSS
    st.markdown(_ANALYTICS_CSS, unsafe_allow_html=True)

    # ── Top Bar with Return to Home ───────────────────────────────────────────
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Return to Home", key="analytics_back_home", type="secondary"):
            st.session_state["page"] = "🏠 Home"
            st.rerun()

    # ── Premium hero header ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="analytics-hero">
      <div class="analytics-hero-inner">
        <div style="flex-shrink:0;opacity:0.85;">{svg_analytics()}</div>
        <div class="analytics-hero-text">
          <div class="hero-page-title">Analytics Dashboard</div>
          <div class="hero-page-sub">Session performance &amp; gesture insights</div>
          <div class="analytics-hero-badge">
            <span style="width:6px;height:6px;border-radius:50%;
                         background:#83A6CE;display:inline-block;"></span>
            LIVE SESSION
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    analytics: SessionAnalytics = st.session_state[analytics_key]
    snap = analytics.snapshot()

    # ── KPI row ────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "Total Gestures",       str(snap.total_gestures),
         "#83A6CE", "linear-gradient(90deg,#83A6CE,#5b9abf)", "🤙"),
        (k2, "Recognition Accuracy", f"{snap.recognition_accuracy*100:.1f}%",
         "#34d399", "linear-gradient(90deg,#34d399,#10b981)", "🎯"),
        (k3, "Avg Confidence",       f"{snap.average_confidence*100:.1f}%",
         "#fbbf24", "linear-gradient(90deg,#fbbf24,#f59e0b)", "📊"),
        (k4, "Current FPS",          f"{snap.current_fps:.1f}",
         "#60a5fa", "linear-gradient(90deg,#60a5fa,#3b82f6)", "⚡"),
        (k5, "Session Time",         analytics.format_duration(),
         "#C48CB3", "linear-gradient(90deg,#C48CB3,#a86f99)", "⏱️"),
    ]
    for col, label, value, color, gradient, icon in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-premium" style="--kpi-accent:{gradient};">
              <span class="kpi-icon">{icon}</span>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};">{value}</div>
            </div>""", unsafe_allow_html=True)

    k6, k7, k8 = st.columns(3)
    kpis2 = [
        (k6, "Most Used Gesture",     snap.most_frequent_gesture or "—",
         "#C48CB3", "linear-gradient(90deg,#C48CB3,#a86f99)", "🏆"),
        (k7, "High Confidence Preds", str(snap.high_confidence_count),
         "#34d399", "linear-gradient(90deg,#34d399,#10b981)", "✅"),
        (k8, "Low Confidence Preds",  str(snap.low_confidence_count),
         "#f87171", "linear-gradient(90deg,#f87171,#ef4444)", "⚠️"),
    ]
    for col, label, value, color, gradient, icon in kpis2:
        with col:
            st.markdown(f"""
            <div class="kpi-premium" style="--kpi-accent:{gradient};">
              <span class="kpi-icon">{icon}</span>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};font-size:1.2rem;">{value}</div>
            </div>""", unsafe_allow_html=True)

    cam_is_active = st.session_state.get("running", False)
    cam_dot_color = "#34d399" if cam_is_active else "#f87171"
    cam_label     = "Active" if cam_is_active else "Inactive"
    st.markdown(
        f'<div class="cam-status-pill">'
        f'<span style="width:8px;height:8px;border-radius:50%;'
        f'background:{cam_dot_color};display:inline-block;'
        f'box-shadow:0 0 6px {cam_dot_color};"></span>'
        f'Camera: {cam_label}</div>',
        unsafe_allow_html=True,
    )

    # ── Confidence breakdown + Gesture frequency ───────────────────────────────
    st.markdown(
        '<div class="analytics-section-title">🎯 Confidence &amp; Frequency</div>',
        unsafe_allow_html=True,
    )
    row1_left, row1_right = st.columns(2, gap="large")

    with row1_left:
        hi  = snap.high_confidence_count
        lo  = snap.low_confidence_count
        tot = snap.total_gestures or 1
        hi_pct = int(hi / tot * 100)
        st.markdown(f"""
        <div class="conf-split-card">
          <div class="chart-glass-title">🎯 Confidence Breakdown</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px;">
            <div style="text-align:center;padding:16px 12px;
                        background:linear-gradient(145deg,#064e3b,#065f46);
                        border-radius:14px;border:1px solid rgba(52,211,153,0.2);">
              <div style="font-size:2rem;font-weight:800;color:#34d399;">{hi}</div>
              <div style="font-size:0.72rem;color:#6ee7b7;font-weight:700;
                          text-transform:uppercase;letter-spacing:0.05em;margin-top:4px;">
                High Confidence</div>
              <div style="font-size:0.68rem;color:rgba(52,211,153,0.7);margin-top:2px;">≥ threshold</div>
            </div>
            <div style="text-align:center;padding:16px 12px;
                        background:linear-gradient(145deg,#450a0a,#3b0606);
                        border-radius:14px;border:1px solid rgba(239,68,68,0.2);">
              <div style="font-size:2rem;font-weight:800;color:#f87171;">{lo}</div>
              <div style="font-size:0.72rem;color:#fca5a5;font-weight:700;
                          text-transform:uppercase;letter-spacing:0.05em;margin-top:4px;">
                Low Confidence</div>
              <div style="font-size:0.68rem;color:rgba(248,113,113,0.7);margin-top:2px;">&lt; threshold</div>
            </div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
              <span style="font-size:0.75rem;color:#64748b;font-weight:600;
                           text-transform:uppercase;letter-spacing:0.05em;">High Conf Rate</span>
              <span style="font-size:0.8rem;color:#34d399;font-weight:800;">{hi_pct}%</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill-high" style="width:{hi_pct}%;transition:width 0.6s ease;"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with row1_right:
        freq = snap.gesture_frequency
        if freq:
            max_count = max(freq.values()) or 1
            color_cycle = [
                "#83A6CE", "#C48CB3", "#60a5fa", "#34d399",
                "#fbbf24", "#f87171", "#06b6d4", "#ec4899",
                "#84cc16", "#f97316",
            ]
            bars_html = ""
            for i, (g, cnt) in enumerate(list(freq.items())[:8]):
                pct = int(cnt / max_count * 100)
                col = color_cycle[i % len(color_cycle)]
                bars_html += f"""
                <div style="margin-bottom:11px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:0.84rem;color:#c7d2fe;font-weight:600;">{g}</span>
                    <span style="font-size:0.8rem;color:{col};font-weight:800;">{cnt}×</span>
                  </div>
                  <div class="conf-track">
                    <div style="width:{pct}%;height:100%;border-radius:8px;
                                background:{col};transition:width 0.5s ease;"></div>
                  </div>
                </div>"""
            st.markdown(
                f'<div class="conf-split-card">'
                f'<div class="chart-glass-title">🤙 Gesture Frequency</div>'
                f'{bars_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div class="conf-split-card" style="text-align:center;padding:44px 20px;">
              <div style="font-size:2.5rem;margin-bottom:12px;opacity:0.3;">🤙</div>
              <div style="color:#334155;font-style:italic;font-size:0.9rem;">
                No gestures recorded yet.</div>
            </div>""", unsafe_allow_html=True)

    # ── FPS chart ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="analytics-section-title">⚡ FPS Over Time</div>',
        unsafe_allow_html=True,
    )
    fps_list = snap.fps_history

    if fps_list:
        st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-glass-title">⚡ Frames Per Second — Live History</div>',
            unsafe_allow_html=True,
        )
        if use_plotly:
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=fps_list, mode="lines", fill="tozeroy",
                    line=dict(color="#83A6CE", width=2.5),
                    fillcolor="rgba(131,166,206,0.08)",
                    name="FPS",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.5)",
                    font=dict(color="#64748b", family="Plus Jakarta Sans"),
                    height=240,
                    margin=dict(l=20, r=20, t=8, b=20),
                    yaxis=dict(
                        gridcolor="rgba(131,166,206,0.08)",
                        color="#475569",
                        tickfont=dict(size=10),
                    ),
                    xaxis=dict(
                        gridcolor="rgba(131,166,206,0.06)",
                        color="#475569",
                        tickfont=dict(size=10),
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                import pandas as pd
                st.line_chart(pd.DataFrame({"FPS": fps_list}),
                              use_container_width=True, height=180)
        else:
            import pandas as pd
            st.line_chart(pd.DataFrame({"FPS": fps_list}),
                          use_container_width=True, height=180)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Confidence distribution pie ────────────────────────────────────────
        if use_plotly and snap.total_gestures > 0:
            try:
                import plotly.graph_objects as go
                st.markdown(
                    '<div class="analytics-section-title">🥧 Confidence Distribution</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="chart-glass-title">🥧 High vs Low Confidence Predictions</div>',
                    unsafe_allow_html=True,
                )
                fig2 = go.Figure(go.Pie(
                    labels=["High Confidence", "Low Confidence"],
                    values=[snap.high_confidence_count, snap.low_confidence_count],
                    hole=0.58,
                    marker=dict(
                        colors=["#83A6CE", "#C48CB3"],
                        line=dict(color="#080D1C", width=2),
                    ),
                    textfont=dict(color="#94a3b8", size=12),
                ))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                    height=280,
                    margin=dict(l=20, r=20, t=8, b=20),
                    legend=dict(
                        font=dict(color="#94a3b8", size=11),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except ImportError:
                pass
    else:
        st.markdown("""
        <div class="chart-glass" style="text-align:center;padding:36px 20px;">
          <div style="font-size:2.5rem;margin-bottom:12px;opacity:0.25;">⚡</div>
          <div style="color:#334155;font-style:italic;font-size:0.9rem;">
            Start the camera to collect FPS data.</div>
        </div>""", unsafe_allow_html=True)

    # ── Reset button ───────────────────────────────────────────────────────────
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset Analytics", key="reset_analytics"):
        analytics.reset()
        st.success("Analytics reset.")
        st.rerun()
