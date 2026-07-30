# ─────────────────────────────────────────────────────────────────────────────
# utils/mediapipe_helper.py
# Wraps MediaPipe Hands so the same initialisation code is not duplicated
# across collection, preprocessing, and inference scripts.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import MAX_NUM_HANDS, MIN_DETECTION_CONF, MIN_TRACKING_CONF, NUM_LANDMARKS


class HandDetector:
    """
    Thin wrapper around MediaPipe Hands.

    Usage
    -----
    detector = HandDetector()
    landmarks, annotated_frame = detector.find_landmarks(frame)
    """

    def __init__(self) -> None:
        """Initialise the MediaPipe Hands solution with project-level config."""
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,          # Video-stream mode (faster)
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        )

    def find_landmarks(
        self,
        frame_bgr: np.ndarray,
        draw: bool = True,
    ) -> tuple[list[float] | None, np.ndarray]:
        """
        Detect hand landmarks in a BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame from OpenCV.
        draw : bool
            Whether to draw the hand skeleton on the returned frame.

        Returns
        -------
        landmarks : list[float] | None
            Flattened list of 42 normalised floats (x, y for 21 landmarks),
            or None if no hand was detected.
        annotated_frame : np.ndarray
            Frame with hand skeleton drawn (or unmodified if draw=False).
        """
        # MediaPipe expects RGB
        frame_rgb = frame_bgr[:, :, ::-1]          # faster than cv2.cvtColor
        results   = self.hands.process(frame_rgb)

        landmarks_flat: list[float] | None = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]  # Take first detected hand

            if draw:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    hand,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_style.get_default_hand_landmarks_style(),
                    self._mp_style.get_default_hand_connections_style(),
                )

            # Extract and flatten x, y coordinates (ignore z for 2-D approach)
            landmarks_flat = []
            for lm in hand.landmark:
                landmarks_flat.extend([lm.x, lm.y])

        return landmarks_flat, frame_bgr

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
