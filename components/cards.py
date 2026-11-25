# =============================================================================
# components/cards.py
# Card / panel UI helpers — thin re-export from utils.ui_components
# =============================================================================

from utils.ui_components import (
    render_glass_card,
    render_ai_panel,
    render_confidence_meter,
    render_translation_cards,
    render_status_badge,
    render_detection_status,
    render_suggestions,
    render_subtitle_strip,
    render_loading_sequence,
    camera_placeholder_html,
)

__all__ = [
    "render_glass_card",
    "render_ai_panel",
    "render_confidence_meter",
    "render_translation_cards",
    "render_status_badge",
    "render_detection_status",
    "render_suggestions",
    "render_subtitle_strip",
    "render_loading_sequence",
    "camera_placeholder_html",
]
