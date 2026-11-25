# =============================================================================
# app_pages/history.py  –  Gesture History Page
# Shared by both app.py and app_cloud.py.
#
# history_key:  app.py → 'history_obj'  |  app_cloud.py → 'history'
# =============================================================================

import streamlit as st
import pandas as pd
from collections import Counter

from utils.gesture_history import GestureHistory


def show_history(history_key: str = "history_obj") -> None:
    """Render the premium SignFlow Gesture History page."""

    history: GestureHistory = st.session_state[history_key]

    # ── Return to Home ─────────────────────────────────────────────────────────
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Return to Home", key="history_back_home", type="secondary"):
            st.session_state["page"] = "🏠 Home"
            st.rerun()

    # ── Page Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:28px;">
      <div style="font-size:0.72rem;font-weight:700;letter-spacing:2.5px;
                  text-transform:uppercase;color:#83A6CE;margin-bottom:8px;">
        Session Data
      </div>
      <h1 style="font-size:2rem;font-weight:900;color:#284155;margin:0 0 6px;
                 letter-spacing:-0.3px;">Translation History</h1>
      <p style="color:#6b8aaa;font-size:0.9rem;margin:0;">
        All gestures recognised in this session.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary Stats ──────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3, gap="medium")

    total     = history.count()
    most_freq = history.most_frequent_gesture() or "—"
    avg_conf  = f"{history.average_confidence() * 100:.1f}%"

    with col_a:
        st.markdown(f"""
        <div class="pro-card" style="text-align:center;padding:22px 18px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:1.5px;
                      text-transform:uppercase;color:#6b8aaa;margin-bottom:8px;">
            Total Gestures
          </div>
          <div style="font-size:2.4rem;font-weight:900;color:#284155;
                      line-height:1;">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="pro-card" style="text-align:center;padding:22px 18px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:1.5px;
                      text-transform:uppercase;color:#6b8aaa;margin-bottom:8px;">
            Most Frequent
          </div>
          <div style="font-size:1.8rem;font-weight:900;
                      background:linear-gradient(135deg,#83A6CE,#C48CB3);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      line-height:1.2;">{most_freq}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class="pro-card" style="text-align:center;padding:22px 18px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:1.5px;
                      text-transform:uppercase;color:#6b8aaa;margin-bottom:8px;">
            Avg Confidence
          </div>
          <div style="font-size:2.4rem;font-weight:900;color:#83A6CE;
                      line-height:1;">{avg_conf}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Action buttons ─────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns(3, gap="small")
    with btn1:
        if not history.is_empty():
            st.download_button(
                "📥 Export CSV",
                data=history.to_csv_bytes(),
                file_name="gesture_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("📥 Export CSV", disabled=True, use_container_width=True)

    with btn2:
        if st.button("🗑 Clear History", use_container_width=True,
                     disabled=history.is_empty()):
            history.clear()
            st.success("History cleared.")
            st.rerun()

    with btn3:
        if st.button("📋 Copy as Text", use_container_width=True,
                     disabled=history.is_empty()):
            st.code(history.to_plain_text(), language=None)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── Empty state ────────────────────────────────────────────────────────────
    if history.is_empty():
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid rgba(131,166,206,0.2);
                    border-radius:22px;text-align:center;padding:56px 24px;
                    box-shadow:0 2px 12px rgba(40,65,85,0.06);">
          <div style="font-size:3.5rem;margin-bottom:16px;">📭</div>
          <div style="font-size:1.1rem;font-weight:700;color:#284155;margin-bottom:8px;">
            No history yet
          </div>
          <div style="font-size:0.9rem;color:#6b8aaa;max-width:320px;margin:0 auto;
                      line-height:1.6;">
            Head over to <strong>Live Translator</strong> and start signing.
            Your recognised gestures will appear here.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Gesture log table ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-title-dark" style="margin-bottom:12px;">
      Gesture Log — newest first
    </div>
    """, unsafe_allow_html=True)

    rows = history.to_display_dicts()

    # Render as styled cards instead of raw dataframe
    for entry in reversed(rows[:30]):
        gesture  = entry.get("Gesture", "—")
        time_str = entry.get("Time", "")
        mode_str = entry.get("Mode", "Word")
        conf_raw = entry.get("Confidence", 0)
        try:
            conf_val = float(str(conf_raw).replace("%", "")) / 100
        except Exception:
            conf_val = 0.0

        conf_display = f"{conf_val * 100:.0f}%"
        conf_color = "#83A6CE" if conf_val >= 0.75 else ("#f59e0b" if conf_val >= 0.5 else "#C48CB3")
        mode_color = "#83A6CE" if mode_str == "Word" else "#C48CB3"

        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid rgba(131,166,206,0.18);
                    border-radius:16px;padding:14px 20px;margin-bottom:8px;
                    display:flex;align-items:center;justify-content:space-between;
                    box-shadow:0 1px 6px rgba(40,65,85,0.05);
                    transition:all 0.18s;flex-wrap:wrap;gap:10px;">
          <div style="display:flex;align-items:center;gap:14px;min-width:0;">
            <div style="width:40px;height:40px;border-radius:10px;
                        background:linear-gradient(135deg,rgba(131,166,206,0.15),rgba(196,140,179,0.15));
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.1rem;flex-shrink:0;">🤙</div>
            <div>
              <div style="font-size:1rem;font-weight:800;color:#284155;">{gesture}</div>
              <div style="font-size:0.74rem;color:#6b8aaa;margin-top:2px;">{time_str}</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <span style="font-size:0.74rem;font-weight:700;color:{mode_color};
                         background:rgba(131,166,206,0.08);padding:3px 10px;
                         border-radius:20px;border:1px solid rgba(131,166,206,0.2);">
              {mode_str}
            </span>
            <span style="font-size:0.82rem;font-weight:800;color:{conf_color};
                         background:rgba(131,166,206,0.06);padding:3px 10px;
                         border-radius:20px;border:1px solid rgba(131,166,206,0.15);">
              {conf_display}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if len(rows) > 30:
        st.markdown(f"""
        <div style="text-align:center;color:#6b8aaa;font-size:0.84rem;
                    padding:8px;font-weight:500;">
          Showing 30 of {len(rows)} entries — export CSV for full history.
        </div>
        """, unsafe_allow_html=True)

    # ── Frequency bars ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title-dark" style="margin-bottom:14px;">Gesture Frequency</div>
    """, unsafe_allow_html=True)

    freq = (
        history.gesture_frequency()
        if hasattr(history, "gesture_frequency")
        else dict(Counter(r.get("Gesture", "") for r in rows))
    )

    if freq:
        max_count = max(freq.values()) or 1
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        for i, (gesture, count) in enumerate(sorted_freq):
            pct = int(count / max_count * 100)
            bar_color = "#83A6CE" if i % 2 == 0 else "#C48CB3"
            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:5px;">
                <span style="font-size:0.9rem;color:#284155;font-weight:600;">{gesture}</span>
                <span style="font-size:0.82rem;color:{bar_color};font-weight:800;
                             background:rgba(131,166,206,0.08);padding:1px 10px;
                             border-radius:20px;">{count}×</span>
              </div>
              <div style="background:rgba(40,65,85,0.07);border-radius:8px;height:10px;
                          overflow:hidden;">
                <div style="width:{pct}%;height:100%;
                            background:linear-gradient(90deg,{bar_color},{bar_color}aa);
                            border-radius:8px;transition:width 0.4s;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
