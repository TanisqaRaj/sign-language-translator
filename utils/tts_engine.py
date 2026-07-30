# =============================================================================
# utils/tts_engine.py
# Phase 8 – Text-to-Speech Integration (Enhanced)
# =============================================================================
# Thread-safe pyttsx3 wrapper with:
#   • "Speak only when prediction is stable for N frames" logic
#   • Cooldown timer to prevent rapid repeated speech
#   • Configurable voice, rate, and volume
#   • Graceful fallback when audio is unavailable
# =============================================================================

import os
import sys
import threading
import time

import pyttsx3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import TTS_RATE, TTS_VOLUME
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum seconds that must pass before the same word can be spoken again.
# This prevents the TTS from firing on every single stable frame.
REPEAT_COOLDOWN_SECS = 3.0


class TTSEngine:
    """
    Thread-safe, production-ready Text-to-Speech engine.

    Design decisions
    ─────────────────
    1.  **Background thread** – pyttsx3's ``runAndWait()`` blocks.  We
        dispatch every speech request to a daemon thread so the main
        video/inference loop is never stalled.

    2.  **Repeat prevention** – the same word/sentence will not be spoken
        again until ``REPEAT_COOLDOWN_SECS`` seconds have elapsed since the
        last utterance.  This avoids the engine firing on every stable
        frame of a held gesture.

    3.  **Force flag** – the Speak button in the UI can pass
        ``force=True`` to bypass both the duplicate-check and the cooldown.

    4.  **Graceful degradation** – if pyttsx3 fails to initialise (no audio
        device, CI environment, etc.) the engine silently disables itself
        and logs a warning.  The rest of the application continues normally.

    Usage
    ──────
    >>> tts = TTSEngine()
    >>> tts.speak("Hello")          # spoken in background
    >>> tts.speak("Hello")          # skipped — too soon / same word
    >>> tts.speak("Hello", force=True)   # always spoken
    >>> tts.reset_last_spoken()     # clear history (e.g. after Clear button)
    """

    def __init__(self) -> None:
        """Initialise pyttsx3 and apply project-level settings."""
        self._lock          = threading.Lock()
        self._last_spoken   = ""
        self._last_spoken_at= 0.0      # Unix timestamp of last speech
        self._available     = False

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   TTS_RATE)
            self._engine.setProperty("volume", TTS_VOLUME)
            self._available = True
            logger.info(
                "TTS engine ready — rate: %d wpm, volume: %.1f",
                TTS_RATE, TTS_VOLUME,
            )
            self._log_voices()
        except Exception as exc:
            self._engine = None
            logger.warning("TTS engine unavailable: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def speak(self, text: str, force: bool = False) -> None:
        """
        Speak ``text`` in a background thread.

        Parameters
        ----------
        text  : str   Word or sentence to speak.
        force : bool  If True, bypass duplicate-check and cooldown.
                      Use this for the manual Speak / Enter key button.
        """
        if not self._available or not text:
            return

        now = time.monotonic()

        with self._lock:
            duplicate  = (text == self._last_spoken)
            too_soon   = (now - self._last_spoken_at) < REPEAT_COOLDOWN_SECS

            if not force and (duplicate or too_soon):
                logger.debug(
                    "TTS skipped '%s' (duplicate=%s, too_soon=%s)",
                    text, duplicate, too_soon,
                )
                return

            self._last_spoken    = text
            self._last_spoken_at = now

        logger.info("TTS speaking: '%s' (force=%s)", text, force)
        thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        thread.start()

    def speak_when_stable(
        self,
        gesture:       str,
        stable_frames: int,
        threshold:     int,
    ) -> None:
        """
        Convenience method — speak ``gesture`` only when it has been stable
        for exactly ``threshold`` frames.

        Call this once per frame from the inference loop instead of
        managing the frame count yourself.

        Parameters
        ----------
        gesture       : current predicted gesture name
        stable_frames : how many consecutive frames this gesture has been seen
        threshold     : required stable frame count (from config.STABLE_FRAME_COUNT)
        """
        if stable_frames == threshold:
            self.speak(gesture)

    def reset_last_spoken(self) -> None:
        """
        Clear the last-spoken history.

        Call this when the user presses the Clear button so the next
        gesture can be spoken even if it matches the previous sentence.
        """
        with self._lock:
            self._last_spoken    = ""
            self._last_spoken_at = 0.0
        logger.debug("TTS history cleared.")

    def set_voice_by_index(self, index: int) -> None:
        """
        Switch to a different system voice by its list index.

        Parameters
        ----------
        index : int  0-based index into the list returned by ``get_voices()``.
        """
        if not self._available:
            return
        voices = self._engine.getProperty("voices")
        if 0 <= index < len(voices):
            self._engine.setProperty("voice", voices[index].id)
            logger.info("TTS voice changed to index %d (%s)", index, voices[index].name)
        else:
            logger.warning("Voice index %d out of range (total: %d).", index, len(voices))

    def get_voices(self) -> list[str]:
        """
        Return a list of available system voice names.

        Returns
        -------
        list[str]  e.g. ['Microsoft Zira Desktop - English (United States)', …]
        """
        if not self._available:
            return []
        return [v.name for v in self._engine.getProperty("voices")]

    @property
    def last_spoken(self) -> str:
        """Return the most recently spoken text."""
        return self._last_spoken

    @property
    def is_available(self) -> bool:
        """True if the TTS engine initialised successfully."""
        return self._available

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run(self, text: str) -> None:
        """Background thread target — calls pyttsx3 blocking API."""
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            logger.error("TTS playback error: %s", exc)

    def _log_voices(self) -> None:
        """Log available system voices at DEBUG level."""
        voices = self.get_voices()
        for i, name in enumerate(voices):
            logger.debug("  Voice %d: %s", i, name)
