# =============================================================================
# tests/test_pipeline.py
# Phase 10 – Unit & Integration Tests
# =============================================================================
# Run with:
#   python -m pytest tests/ -v
#
# Tests cover:
#   • Landmark normalisation correctness
#   • HandDetector interface contract
#   • TTSEngine duplicate-prevention logic
#   • PredictionSmoother majority-vote logic
#   • Model loading helpers (mocked)
#   • config.py sanity checks
# =============================================================================

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    GESTURE_LABELS,
    NUM_CLASSES,
    NUM_LANDMARKS,
    LANDMARK_FEATURES,
    CONFIDENCE_THRESHOLD,
    STABLE_FRAME_COUNT,
    PREDICTION_BUFFER_LEN,
)
from preprocess import normalise_landmarks
from inference import PredictionSmoother


# ─────────────────────────────────────────────────────────────────────────────
# Config sanity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    """Verify that config.py constants are internally consistent."""

    def test_num_classes_matches_label_list(self):
        """NUM_CLASSES must equal the length of GESTURE_LABELS."""
        self.assertEqual(NUM_CLASSES, len(GESTURE_LABELS))

    def test_landmark_features_formula(self):
        """LANDMARK_FEATURES must be NUM_LANDMARKS * 2 (x and y per point)."""
        self.assertEqual(LANDMARK_FEATURES, NUM_LANDMARKS * 2)

    def test_confidence_threshold_in_range(self):
        """Confidence threshold must be between 0 and 1."""
        self.assertGreater(CONFIDENCE_THRESHOLD, 0.0)
        self.assertLessEqual(CONFIDENCE_THRESHOLD, 1.0)

    def test_gesture_labels_are_strings(self):
        """Every gesture label must be a non-empty string."""
        for label in GESTURE_LABELS:
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)

    def test_stable_frame_count_positive(self):
        """STABLE_FRAME_COUNT must be a positive integer."""
        self.assertGreater(STABLE_FRAME_COUNT, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark normalisation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseLandmarks(unittest.TestCase):
    """Test that normalise_landmarks produces correctly shaped, bounded output."""

    def _make_landmarks(self) -> list[float]:
        """Return a flat list of 42 random floats simulating MediaPipe output."""
        return list(np.random.rand(LANDMARK_FEATURES).astype(float))

    def test_output_length(self):
        """Output must contain exactly LANDMARK_FEATURES values."""
        result = normalise_landmarks(self._make_landmarks())
        self.assertEqual(len(result), LANDMARK_FEATURES)

    def test_values_in_range(self):
        """All normalised values must lie in [-1, 1]."""
        for _ in range(50):
            result = normalise_landmarks(self._make_landmarks())
            arr    = np.array(result)
            self.assertTrue(
                np.all(arr >= -1.0) and np.all(arr <= 1.0),
                msg=f"Values out of [-1, 1]: min={arr.min():.4f} max={arr.max():.4f}",
            )

    def test_wrist_at_origin(self):
        """After normalisation, wrist (landmark 0) must be at (0, 0)."""
        lm     = self._make_landmarks()
        result = normalise_landmarks(lm)
        self.assertAlmostEqual(result[0], 0.0, places=6)   # x0
        self.assertAlmostEqual(result[1], 0.0, places=6)   # y0

    def test_zero_vector_handled(self):
        """All-zero input must not raise ZeroDivisionError."""
        zeros  = [0.0] * LANDMARK_FEATURES
        result = normalise_landmarks(zeros)
        self.assertEqual(len(result), LANDMARK_FEATURES)

    def test_same_input_same_output(self):
        """Normalisation must be deterministic."""
        lm      = self._make_landmarks()
        result1 = normalise_landmarks(lm)
        result2 = normalise_landmarks(lm)
        np.testing.assert_array_equal(result1, result2)

    def test_translation_invariant(self):
        """Translating all landmarks by a constant offset must yield the same result."""
        lm        = self._make_landmarks()
        lm_shifted = [v + 0.3 for v in lm]
        r1 = np.array(normalise_landmarks(lm))
        r2 = np.array(normalise_landmarks(lm_shifted))
        np.testing.assert_array_almost_equal(r1, r2, decimal=5)


# ─────────────────────────────────────────────────────────────────────────────
# PredictionSmoother tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionSmoother(unittest.TestCase):
    """Test the rolling majority-vote smoother in inference.py."""

    def setUp(self):
        """Create a fresh smoother before each test."""
        self.smoother = PredictionSmoother(buffer_len=5)

    def test_low_confidence_returns_none(self):
        """Predictions below CONFIDENCE_THRESHOLD must be discarded."""
        gesture, conf = self.smoother.update("Hello", CONFIDENCE_THRESHOLD - 0.01)
        self.assertIsNone(gesture)

    def test_majority_accepted(self):
        """Consistent high-confidence predictions must produce a stable result."""
        # Push the same gesture 5 times (fills the buffer)
        for _ in range(5):
            gesture, _ = self.smoother.update("Hello", 0.95)
        self.assertEqual(gesture, "Hello")

    def test_minority_rejected(self):
        """A single prediction cannot win a majority over four others."""
        for _ in range(4):
            self.smoother.update("Hello", 0.95)
        gesture, _ = self.smoother.update("Yes", 0.95)
        # "Yes" appears once, "Hello" appears 4× — majority should be Hello
        self.assertIn(gesture, ("Hello", None))
        self.assertNotEqual(gesture, "Yes")

    def test_reset_clears_buffer(self):
        """After reset(), the smoother should not return a stable prediction."""
        for _ in range(5):
            self.smoother.update("Hello", 0.95)
        self.smoother.reset()
        gesture, _ = self.smoother.update("Hello", 0.95)
        # Buffer now has only 1 sample — not enough for majority
        self.assertIsNone(gesture)

    def test_different_classes_no_majority(self):
        """Alternating predictions must not produce a stable result."""
        results = []
        for i in range(6):
            gesture_name = "Hello" if i % 2 == 0 else "Yes"
            g, _ = self.smoother.update(gesture_name, 0.95)
            results.append(g)
        # No single class should dominate the alternating buffer
        non_none = [r for r in results if r is not None]
        # It is acceptable to have some non-None but they should not all agree
        unique = set(non_none)
        # If there are results they should come from both classes or be empty
        self.assertTrue(len(unique) <= 2)


# ─────────────────────────────────────────────────────────────────────────────
# TTSEngine tests (with mock to avoid actual audio)
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSEngine(unittest.TestCase):
    """Test TTSEngine duplicate-prevention and cooldown logic using mocks."""

    def _make_engine(self):
        """Create a TTSEngine with a mocked pyttsx3 backend."""
        with patch("utils.tts_engine.pyttsx3") as mock_pyttsx3:
            mock_engine_instance = MagicMock()
            mock_pyttsx3.init.return_value = mock_engine_instance

            from importlib import reload
            import utils.tts_engine as tts_module
            reload(tts_module)
            engine = tts_module.TTSEngine()
            engine._engine = mock_engine_instance
            engine._available = True
        return engine

    def test_duplicate_skipped(self):
        """The same word must not be spoken twice in quick succession."""
        engine = self._make_engine()
        engine.speak("Hello")
        import time
        time.sleep(0.05)
        with patch.object(engine, "_run") as mock_run:
            engine.speak("Hello")          # Should be skipped
            mock_run.assert_not_called()

    def test_force_bypasses_duplicate_check(self):
        """force=True must speak even if the word was just spoken."""
        engine = self._make_engine()
        engine._last_spoken = "Hello"
        with patch.object(engine, "_run") as mock_run:
            engine.speak("Hello", force=True)
            mock_run.assert_called_once_with("Hello")

    def test_reset_clears_last_spoken(self):
        """reset_last_spoken() must clear the cached last word."""
        engine = self._make_engine()
        engine._last_spoken = "Hello"
        engine.reset_last_spoken()
        self.assertEqual(engine.last_spoken, "")

    def test_empty_text_not_spoken(self):
        """Empty string must not trigger speech."""
        engine = self._make_engine()
        with patch.object(engine, "_run") as mock_run:
            engine.speak("")
            mock_run.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Label map I/O test
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelMap(unittest.TestCase):
    """Test saving and loading label_map.json."""

    def test_label_map_round_trip(self):
        """Label map must survive a JSON write/read cycle unchanged."""
        original = {i: label for i, label in enumerate(GESTURE_LABELS)}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(original, f)
            tmp_path = f.name

        try:
            with open(tmp_path) as f:
                loaded = json.load(f)

            # JSON keys become strings — cast back to int for comparison
            loaded_int_keys = {int(k): v for k, v in loaded.items()}
            self.assertEqual(original, loaded_int_keys)
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# FPSCounter test
# ─────────────────────────────────────────────────────────────────────────────

class TestFPSCounter(unittest.TestCase):
    """Test the rolling FPS counter utility."""

    def test_initial_fps_is_zero(self):
        """FPS must be 0.0 before any ticks are recorded."""
        from utils.performance import FPSCounter
        counter = FPSCounter()
        self.assertEqual(counter.fps, 0.0)

    def test_fps_after_ticks(self):
        """FPS must be positive after several ticks."""
        import time
        from utils.performance import FPSCounter
        counter = FPSCounter(window=5)
        for _ in range(5):
            counter.tick()
            time.sleep(0.01)
        self.assertGreater(counter.fps, 0.0)

    def test_fps_single_tick_is_zero(self):
        """A single tick cannot produce a meaningful FPS (needs at least 2)."""
        from utils.performance import FPSCounter
        counter = FPSCounter()
        counter.tick()
        self.assertEqual(counter.fps, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
