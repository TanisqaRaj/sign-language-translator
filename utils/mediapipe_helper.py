# ─────────────────────────────────────────────────────────────────────────────
# utils/mediapipe_helper.py
# Wraps MediaPipe Hands so the same initialisation code is not duplicated
# across collection, preprocessing, and inference scripts.
#
# Two-hand detection (MAX_NUM_HANDS = 2 in config.py):
#
#   find_landmarks()     — BACKWARD-COMPATIBLE, unchanged API.
#                          Returns 42-float list for the dominant hand
#                          (right preferred; falls back to left if only
#                          left is visible).  All existing callers
#                          (preprocess, collect_data, inference, app) work
#                          without any modification.
#
#   find_all_landmarks() — NEW.  Returns full per-hand data for every
#                          detected hand in the frame, keyed by handedness
#                          label ("Left" / "Right").  Used by the Streamlit
#                          apps to draw both skeletons and show the
#                          "Hands Detected" debug strip.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import MAX_NUM_HANDS, MIN_DETECTION_CONF, MIN_TRACKING_CONF, NUM_LANDMARKS


# ── Public data class returned by find_all_landmarks() ───────────────────────

@dataclass
class HandResult:
    """
    Per-hand detection result.

    Attributes
    ----------
    label       : "Left" or "Right" (MediaPipe's handedness label, already
                  mirror-corrected for a flipped webcam feed).
    landmarks   : flat list of 42 floats [x0,y0,…,x20,y20] in normalised
                  image coordinates [0,1].  These are the RAW pixel-space
                  values — call preprocess.normalise_landmarks() before
                  feeding to a model.
    score       : MediaPipe handedness confidence (0–1).
    """
    label:     str
    landmarks: list[float]
    score:     float


@dataclass
class MultiHandResult:
    """
    Container for all detected hands in one frame.

    Attributes
    ----------
    hands            : list of HandResult, one per detected hand.
    left             : HandResult for the left hand, or None.
    right            : HandResult for the right hand, or None.
    dominant         : HandResult for the "dominant" hand used for inference.
                       Selection priority:
                         1. Right hand  (preferred; ISL is mostly right-hand dominant)
                         2. Left hand   (when only the left hand is visible)
                         3. None        (no hand detected at all)
    count            : total number of detected hands (0, 1 or 2).
    left_detected    : bool — True if left hand is present.
    right_detected   : bool — True if right hand is present.
    """
    hands:          list[HandResult] = field(default_factory=list)
    left:           HandResult | None = None
    right:          HandResult | None = None
    dominant:       HandResult | None = None
    count:          int = 0
    left_detected:  bool = False
    right_detected: bool = False


# ── HandDetector ──────────────────────────────────────────────────────────────

class HandDetector:
    """
    Thin wrapper around MediaPipe Hands.

    Two public methods
    ------------------
    find_landmarks(frame, draw)
        Backward-compatible single-hand API.
        Returns (landmarks_flat | None, annotated_frame).

    find_all_landmarks(frame, draw)
        Two-hand-aware API.
        Returns (MultiHandResult, annotated_frame).
    """

    def __init__(self, static_image_mode: bool = False) -> None:
        """Initialise the MediaPipe Hands solution with project-level config.

        Parameters
        ----------
        static_image_mode : bool
            True  → static image mode (for preprocessing images, higher accuracy)
            False → video stream mode (for real-time inference, faster tracking)
        """
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=MAX_NUM_HANDS,          # 2 — allows both hands
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_flat(self, hand_landmarks) -> list[float]:
        """Flatten 21 MediaPipe landmarks → [x0,y0,…,x20,y20] (42 floats)."""
        flat: list[float] = []
        for lm in hand_landmarks.landmark:
            flat.extend([lm.x, lm.y])   # z ignored (2-D approach)
        return flat

    def _draw_hand(self, frame_bgr: np.ndarray, hand_landmarks) -> None:
        """Draw a single hand skeleton onto frame_bgr in-place."""
        self._mp_draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_style.get_default_hand_landmarks_style(),
            self._mp_style.get_default_hand_connections_style(),
        )

    # ── Public: backward-compatible single-hand method ────────────────────────

    def find_landmarks(
        self,
        frame_bgr: np.ndarray,
        draw: bool = True,
    ) -> tuple[list[float] | None, np.ndarray]:
        """
        Detect hand landmarks in a BGR frame (backward-compatible API).

        Behaviour with two hands present
        ---------------------------------
        Returns landmarks for the *dominant* hand only (right hand preferred;
        falls back to left if the right hand is not visible).  This keeps the
        42-feature model input identical to how the models were trained.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame from OpenCV.
        draw : bool
            Whether to draw ALL detected hand skeletons on the returned frame
            (both hands are drawn when visible, but only the dominant hand's
            landmarks are returned for inference).

        Returns
        -------
        landmarks : list[float] | None
            Flattened list of 42 floats (x, y for 21 landmarks) for the
            dominant hand, or None if no hand was detected.
        annotated_frame : np.ndarray
            Frame with hand skeleton(s) drawn (or unmodified if draw=False).
        """
        mhr, annotated = self.find_all_landmarks(frame_bgr, draw=draw)

        if mhr.dominant is None:
            return None, annotated

        return mhr.dominant.landmarks, annotated

    # ── Public: two-hand-aware method ─────────────────────────────────────────

    def find_all_landmarks(
        self,
        frame_bgr: np.ndarray,
        draw: bool = True,
    ) -> tuple[MultiHandResult, np.ndarray]:
        """
        Detect up to two hand landmarks in a BGR frame.

        Handles all five edge cases correctly:
          Case 1 — only left hand visible
          Case 2 — only right hand visible
          Case 3 — both hands visible simultaneously
          Case 4 — one hand temporarily disappears (other continues correctly)
          Case 5 — MediaPipe changes detection order between frames
                   (handedness label is used, NOT array index)

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame from OpenCV.
        draw : bool
            Whether to draw all detected hand skeletons onto the frame.

        Returns
        -------
        result : MultiHandResult
            Full per-hand information for all detected hands.
        annotated_frame : np.ndarray
            Frame with hand skeleton(s) drawn.
        """
        # MediaPipe expects RGB
        frame_rgb = frame_bgr[:, :, ::-1]   # faster than cv2.cvtColor
        results   = self.hands.process(frame_rgb)

        mhr = MultiHandResult()

        if not results.multi_hand_landmarks:
            return mhr, frame_bgr

        # ── Iterate over every detected hand ──────────────────────────────────
        # IMPORTANT: We use results.multi_handedness[i].classification[0].label
        # instead of the array index so that Left/Right identification is
        # stable regardless of detection order changes between frames.
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            # MediaPipe handedness label is from the model's perspective.
            # Because we flip the webcam frame (cv2.flip(img, 1)), Left/Right
            # is already mirror-corrected and matches the user's actual hand.
            label = handedness.classification[0].label   # "Left" or "Right"
            score = handedness.classification[0].score

            flat = self._extract_flat(hand_lm)
            hr   = HandResult(label=label, landmarks=flat, score=score)
            mhr.hands.append(hr)

            if label == "Left":
                mhr.left          = hr
                mhr.left_detected = True
            else:  # "Right"
                mhr.right          = hr
                mhr.right_detected = True

            # Draw skeleton for this hand
            if draw:
                self._draw_hand(frame_bgr, hand_lm)

        mhr.count = len(mhr.hands)

        # ── Choose dominant hand for single-model inference ───────────────────
        # Priority: Right > Left  (ISL is predominantly right-hand dominant,
        # and the training data was collected right-hand-first).
        # When only the left hand is visible the left hand is used so
        # left-handed users and two-hand gestures requiring the left hand
        # are still supported.
        if mhr.right is not None:
            mhr.dominant = mhr.right
        elif mhr.left is not None:
            mhr.dominant = mhr.left
        # else: dominant remains None (no hand detected)

        return mhr, frame_bgr

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
