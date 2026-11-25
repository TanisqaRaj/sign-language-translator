# =============================================================================
# core/speech.py
# Text-to-speech — thin re-export from utils.tts_engine
# =============================================================================
# The local app (app.py) uses TTSEngine (pyttsx3-based, desktop only).
# The cloud app (app_cloud.py) uses the browser Web Speech API via injected JS
# and does NOT use TTSEngine at all.
#
# Only import TTSEngine when running the local app.  The cloud app never
# imports this module.

from utils.tts_engine import TTSEngine

__all__ = ["TTSEngine"]
