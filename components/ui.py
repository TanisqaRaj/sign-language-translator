# =============================================================================
# components/ui.py
# Convenience single-import surface for all shared UI pieces.
# Import from here instead of reaching directly into utils.ui_components.
# =============================================================================

from utils.ui_components import (
    # CSS
    inject_css,
    CLAY_CSS,
    HIGH_CONTRAST_CSS,
    LARGE_FONT_CSS,
    # SVG illustrations
    svg_hand_3d,
    svg_ai_robot,
    svg_microphone,
    svg_speaker,
    svg_analytics,
    # Component helpers
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
    "inject_css",
    "CLAY_CSS",
    "HIGH_CONTRAST_CSS",
    "LARGE_FONT_CSS",
    "svg_hand_3d",
    "svg_ai_robot",
    "svg_microphone",
    "svg_speaker",
    "svg_analytics",
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
