# =============================================================================
# components/styles.py
# CSS injection — thin re-export from utils.ui_components
# =============================================================================
# Import directly from the canonical source so there is only one copy of CSS.

from utils.ui_components import (
    inject_css,
    CLAY_CSS,
    HIGH_CONTRAST_CSS,
    LARGE_FONT_CSS,
)

__all__ = [
    "inject_css",
    "CLAY_CSS",
    "HIGH_CONTRAST_CSS",
    "LARGE_FONT_CSS",
]
