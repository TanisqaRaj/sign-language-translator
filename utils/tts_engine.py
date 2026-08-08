# =============================================================================
# utils/tts_engine.py  –  TTS via PowerShell SpeechSynthesizer (Windows)
# =============================================================================

import os
import subprocess
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import TTS_RATE, TTS_VOLUME
from utils.logger import get_logger

logger = get_logger(__name__)

REPEAT_COOLDOWN_SECS = 3.0


class TTSEngine:

    def __init__(self) -> None:
        self._lock           = threading.Lock()
        self._last_spoken    = ""
        self._last_spoken_at = 0.0
        self._speaking       = False
        self._available      = False
        self._voices         = []
        self._voice_index    = 0

        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Add-Type -AssemblyName System.Speech; "
                 "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                 "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"],
                capture_output=True, text=True, timeout=10,
            )
            self._voices = [v for v in result.stdout.strip().splitlines() if v]
            self._available = True
            logger.info("TTS ready — %d voices found", len(self._voices))
        except Exception as exc:
            logger.warning("TTS unavailable: %s", exc)

    # ── public ────────────────────────────────────────────────────────────────

    def speak(self, text: str, force: bool = False) -> None:
        if not self._available or not text:
            return

        now = time.monotonic()
        with self._lock:
            duplicate = (text == self._last_spoken)
            too_soon  = (now - self._last_spoken_at) < REPEAT_COOLDOWN_SECS
            if not force and (duplicate or too_soon):
                return
            self._last_spoken    = text
            self._last_spoken_at = now

        if self._speaking:
            return

        logger.info("TTS speaking: '%s'", text)
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def speak_when_stable(self, gesture: str, stable_frames: int, threshold: int) -> None:
        if stable_frames == threshold:
            self.speak(gesture)

    def reset_last_spoken(self) -> None:
        with self._lock:
            self._last_spoken    = ""
            self._last_spoken_at = 0.0

    def set_voice_by_index(self, index: int) -> None:
        if 0 <= index < len(self._voices):
            self._voice_index = index

    def get_voices(self) -> list:
        return self._voices

    @property
    def last_spoken(self) -> str:
        return self._last_spoken

    @property
    def is_available(self) -> bool:
        return self._available

    # ── internal ──────────────────────────────────────────────────────────────

    def _run(self, text: str) -> None:
        self._speaking = True
        try:
            # Escape single quotes in text for PowerShell
            safe_text = text.replace("'", "''")
            voice_line = ""
            if self._voices and self._voice_index < len(self._voices):
                voice_name = self._voices[self._voice_index].replace("'", "''")
                voice_line = f"$s.SelectVoice('{voice_name}'); "

            # Rate: PowerShell uses -10 to +10, map from wpm (80-300)
            ps_rate = max(-10, min(10, (TTS_RATE - 150) // 15))
            volume  = int(TTS_VOLUME * 100)

            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Volume = {volume}; "
                f"$s.Rate = {ps_rate}; "
                f"{voice_line}"
                f"$s.Speak('{safe_text}')"
            )
            subprocess.run(
                ["powershell", "-Command", script],
                timeout=15,
            )
        except Exception as exc:
            logger.error("TTS error: %s", exc)
        finally:
            self._speaking = False
