# =============================================================================
# utils/helpers.py
# Shared small utility helpers used across pages and app entry points.
# =============================================================================

from __future__ import annotations

import streamlit as st

from utils.gesture_history import GestureHistory
from utils.analytics import SessionAnalytics


def init_history_and_analytics(
    history_key: str = "history_obj",
    analytics_key: str = "analytics_obj",
) -> None:
    """
    Lazily initialise GestureHistory and SessionAnalytics objects in
    st.session_state if they are not already present.

    Parameters
    ----------
    history_key : str
        Session-state key under which the GestureHistory instance is stored.
        app.py uses 'history_obj'; app_cloud.py uses 'history'.
    analytics_key : str
        Session-state key under which the SessionAnalytics instance is stored.
        app.py uses 'analytics_obj'; app_cloud.py uses 'analytics'.
    """
    if st.session_state.get(history_key) is None:
        st.session_state[history_key] = GestureHistory()

    if st.session_state.get(analytics_key) is None:
        analytics = SessionAnalytics()
        analytics.start_session()
        st.session_state[analytics_key] = analytics


__all__ = ["init_history_and_analytics"]
