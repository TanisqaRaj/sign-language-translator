# =============================================================================
# utils/gesture_history.py
# Gesture History Management
# =============================================================================
# Maintains a session-level log of every confirmed gesture detection.
# Supports:
#   • Adding new entries (time, gesture, translation, confidence)
#   • Clearing history
#   • Exporting to CSV (in-memory bytes for Streamlit download)
#   • Copying as plain text
#   • Querying the most frequent gesture
# =============================================================================

import csv
import io
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HistoryEntry:
    """One row in the gesture history log."""
    timestamp:       str    # ISO-format string, e.g. "2026-08-06 20:50:13"
    english_text:    str    # Raw English gesture name, e.g. "Hello"
    translated_text: str    # Translated text in selected language
    confidence:      float  # Model confidence 0.0–1.0
    language:        str    # Target language display name, e.g. "Hindi"


class GestureHistory:
    """
    In-memory ordered list of HistoryEntry records for the current session.

    Thread-safety note: Streamlit runs in a single thread per session,
    so no explicit lock is needed here. If used in a multi-threaded context,
    wrap mutating calls in a threading.Lock.
    """

    def __init__(self, max_entries: int = 500) -> None:
        """
        Parameters
        ----------
        max_entries : int  Maximum number of entries to keep (oldest are dropped).
        """
        self._entries: list[HistoryEntry] = []
        self._max = max_entries

    # ──────────────────────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────────────────────

    def add(
        self,
        english_text:    str,
        translated_text: str,
        confidence:      float,
        language:        str = "English",
    ) -> None:
        """
        Append a new gesture detection record.

        Parameters
        ----------
        english_text    : Raw English prediction (e.g. "Hello")
        translated_text : Translated text in the selected language
        confidence      : Model confidence score (0.0 – 1.0)
        language        : Target language display name
        """
        entry = HistoryEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            english_text=english_text,
            translated_text=translated_text,
            confidence=round(confidence, 4),
            language=language,
        )
        self._entries.append(entry)

        # Trim oldest entries if over capacity
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max :]

        logger.debug("History: added '%s' (%.1f%%)", english_text, confidence * 100)

    def clear(self) -> None:
        """Remove all history entries."""
        self._entries.clear()
        logger.info("Gesture history cleared.")

    def pop_last(self) -> Optional[HistoryEntry]:
        """Remove and return the most recent entry, or None if empty."""
        if self._entries:
            return self._entries.pop()
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Read operations
    # ──────────────────────────────────────────────────────────────────────────

    def get_all(self) -> list[HistoryEntry]:
        """Return all entries, oldest first."""
        return list(self._entries)

    def get_recent(self, n: int = 20) -> list[HistoryEntry]:
        """Return the ``n`` most recent entries, newest first."""
        return list(reversed(self._entries[-n:]))

    def count(self) -> int:
        """Total number of history entries."""
        return len(self._entries)

    def is_empty(self) -> bool:
        """True if no entries have been recorded."""
        return len(self._entries) == 0

    def most_frequent_gesture(self) -> Optional[str]:
        """
        Return the English gesture name that appears most often in history.
        Returns None if history is empty.
        """
        if not self._entries:
            return None
        freq: dict[str, int] = {}
        for e in self._entries:
            freq[e.english_text] = freq.get(e.english_text, 0) + 1
        return max(freq, key=lambda k: freq[k])

    def gesture_frequency(self) -> dict[str, int]:
        """
        Return a dict mapping each gesture to how many times it was detected.

        Returns
        -------
        dict[str, int]  e.g. {"Hello": 5, "Thank You": 3, …}
        """
        freq: dict[str, int] = {}
        for e in self._entries:
            freq[e.english_text] = freq.get(e.english_text, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    def average_confidence(self) -> float:
        """Return the mean confidence over all history entries, or 0.0 if empty."""
        if not self._entries:
            return 0.0
        return sum(e.confidence for e in self._entries) / len(self._entries)

    # ──────────────────────────────────────────────────────────────────────────
    # Export operations
    # ──────────────────────────────────────────────────────────────────────────

    def to_csv_bytes(self) -> bytes:
        """
        Serialise history to CSV and return as UTF-8 encoded bytes.

        Suitable for Streamlit's ``st.download_button(data=…)``.

        Returns
        -------
        bytes  CSV file content with header row.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Timestamp", "English", "Translated", "Language", "Confidence (%)"])
        for e in self._entries:
            writer.writerow([
                e.timestamp,
                e.english_text,
                e.translated_text,
                e.language,
                f"{e.confidence * 100:.1f}",
            ])
        return buffer.getvalue().encode("utf-8")

    def to_plain_text(self) -> str:
        """
        Serialise history to a human-readable plain-text string.

        Suitable for clipboard copy.
        """
        if not self._entries:
            return "No gesture history recorded."
        lines = ["=== Gesture History ==="]
        for e in self._entries:
            lines.append(
                f"[{e.timestamp}] {e.english_text} → {e.translated_text} "
                f"({e.language}) | {e.confidence * 100:.1f}%"
            )
        return "\n".join(lines)

    def to_display_dicts(self) -> list[dict]:
        """
        Convert entries to a list of plain dicts for use with st.dataframe().

        Returns
        -------
        list[dict]  Each dict has keys: Time, Gesture, Translation, Language, Confidence
        """
        return [
            {
                "Time":        e.timestamp,
                "Gesture":     e.english_text,
                "Translation": e.translated_text,
                "Language":    e.language,
                "Confidence":  f"{e.confidence * 100:.1f}%",
            }
            for e in reversed(self._entries)   # newest first for display
        ]
