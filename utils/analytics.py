# =============================================================================
# utils/analytics.py
# Session Analytics Tracker
# =============================================================================
# Tracks real-time statistics for the current session:
#   • Total gestures detected
#   • High / low confidence counts
#   • Average confidence
#   • FPS history (for chart)
#   • Session start time / active duration
#   • Per-gesture frequency map
#
# All data lives in memory; nothing is persisted to disk.
# =============================================================================

import time
import collections
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import LOW_CONFIDENCE_THRESHOLD, ANALYTICS_MAX_HISTORY
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionSnapshot:
    """A point-in-time snapshot of session analytics."""
    total_gestures:            int
    high_confidence_count:     int
    low_confidence_count:      int
    average_confidence:        float          # 0.0 – 1.0
    recognition_accuracy:      float          # high_conf / total, 0.0 – 1.0
    most_frequent_gesture:     Optional[str]
    session_duration_seconds:  float
    current_fps:               float
    gesture_frequency:         dict[str, int]
    fps_history:               list[float]    # recent FPS values for chart


class SessionAnalytics:
    """
    Lightweight, in-memory analytics tracker for a single app session.

    Usage
    ──────
    analytics = SessionAnalytics()
    analytics.start_session()

    # In the webcam loop, once per confirmed gesture:
    analytics.record_gesture("Hello", confidence=0.96)
    analytics.record_fps(28.5)

    # Retrieve a snapshot for the dashboard:
    snap = analytics.snapshot()
    """

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._total_gestures    = 0
        self._high_conf_count   = 0
        self._low_conf_count    = 0
        self._confidence_sum    = 0.0
        self._gesture_freq: dict[str, int] = {}

        # Rolling FPS history (keeps last ANALYTICS_MAX_HISTORY samples)
        self._fps_history: collections.deque = collections.deque(
            maxlen=ANALYTICS_MAX_HISTORY
        )

        # Confidence samples for average calculation
        self._conf_samples: collections.deque = collections.deque(
            maxlen=ANALYTICS_MAX_HISTORY
        )

        self._current_fps = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Session lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def start_session(self) -> None:
        """Mark the start of a new session. Resets all counters."""
        self._start_time        = time.monotonic()
        self._total_gestures    = 0
        self._high_conf_count   = 0
        self._low_conf_count    = 0
        self._confidence_sum    = 0.0
        self._gesture_freq      = {}
        self._fps_history.clear()
        self._conf_samples.clear()
        self._current_fps       = 0.0
        logger.info("Analytics: new session started.")

    def reset(self) -> None:
        """Alias for start_session — resets everything."""
        self.start_session()

    # ──────────────────────────────────────────────────────────────────────────
    # Data ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def record_gesture(self, gesture: str, confidence: float) -> None:
        """
        Record one confirmed gesture detection.

        Parameters
        ----------
        gesture    : English gesture name, e.g. "Hello"
        confidence : Model confidence 0.0 – 1.0
        """
        self._total_gestures += 1
        self._confidence_sum += confidence
        self._conf_samples.append(confidence)

        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            self._high_conf_count += 1
        else:
            self._low_conf_count += 1

        self._gesture_freq[gesture] = self._gesture_freq.get(gesture, 0) + 1

        logger.debug(
            "Analytics: gesture='%s' conf=%.2f total=%d",
            gesture, confidence, self._total_gestures,
        )

    def record_fps(self, fps: float) -> None:
        """
        Record the current frames-per-second reading.

        Parameters
        ----------
        fps : float  Instantaneous or rolling FPS value.
        """
        self._current_fps = fps
        self._fps_history.append(round(fps, 1))

    # ──────────────────────────────────────────────────────────────────────────
    # Derived properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def total_gestures(self) -> int:
        return self._total_gestures

    @property
    def high_confidence_count(self) -> int:
        return self._high_conf_count

    @property
    def low_confidence_count(self) -> int:
        return self._low_conf_count

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @property
    def session_duration(self) -> float:
        """Elapsed seconds since start_session() was called."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def average_confidence(self) -> float:
        """Mean confidence over all recorded gestures."""
        if not self._conf_samples:
            return 0.0
        return sum(self._conf_samples) / len(self._conf_samples)

    @property
    def recognition_accuracy(self) -> float:
        """Fraction of gestures that met the confidence threshold."""
        if self._total_gestures == 0:
            return 0.0
        return self._high_conf_count / self._total_gestures

    @property
    def most_frequent_gesture(self) -> Optional[str]:
        """Most-detected gesture, or None if none recorded."""
        if not self._gesture_freq:
            return None
        return max(self._gesture_freq, key=lambda k: self._gesture_freq[k])

    @property
    def gesture_frequency(self) -> dict[str, int]:
        """Dict mapping gesture name → occurrence count."""
        return dict(
            sorted(self._gesture_freq.items(), key=lambda x: x[1], reverse=True)
        )

    @property
    def fps_history_list(self) -> list[float]:
        """Most-recent FPS samples as a plain list."""
        return list(self._fps_history)

    # ──────────────────────────────────────────────────────────────────────────
    # Snapshot
    # ──────────────────────────────────────────────────────────────────────────

    def snapshot(self) -> SessionSnapshot:
        """
        Return an immutable snapshot of all current analytics.

        Use this to pass data to the UI without holding references to the
        mutable analytics object.
        """
        return SessionSnapshot(
            total_gestures=self._total_gestures,
            high_confidence_count=self._high_conf_count,
            low_confidence_count=self._low_conf_count,
            average_confidence=self.average_confidence,
            recognition_accuracy=self.recognition_accuracy,
            most_frequent_gesture=self.most_frequent_gesture,
            session_duration_seconds=self.session_duration,
            current_fps=self._current_fps,
            gesture_frequency=self.gesture_frequency,
            fps_history=self.fps_history_list,
        )

    def format_duration(self) -> str:
        """
        Format session duration as a human-readable string.

        Returns
        -------
        str  e.g. "2m 34s" or "45s"
        """
        secs = int(self.session_duration)
        if secs < 60:
            return f"{secs}s"
        mins = secs // 60
        secs = secs % 60
        return f"{mins}m {secs:02d}s"
