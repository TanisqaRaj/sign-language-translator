# =============================================================================
# utils/movenet_helper.py
# MoveNet single-pose body-pose detector — TF SavedModel version.
#
# Downloads MoveNet Lightning via tf.keras.utils.get_file (no tensorflow_hub
# dependency needed — avoids the pkg_resources / Python 3.12 compatibility
# issue with tensorflow-hub 0.16.1).
#
# MoveNet keypoint ordering (0-indexed):
#   0  nose          1  left_eye        2  right_eye
#   3  left_ear      4  right_ear       5  left_shoulder
#   6  right_shoulder 7 left_elbow      8  right_elbow
#   9  left_wrist   10  right_wrist    11  left_hip
#  12  right_hip    13  left_knee      14  right_knee
#  15  left_ankle   16  right_ankle
# =============================================================================

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MOVENET_THRESHOLD,
    NUM_POSE_KEYPOINTS,
    POSE_FEATURES,
)

# MoveNet Lightning TFLite model URL (much lighter than the SavedModel)
_MOVENET_TFLITE_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "tflite/movenet/movenet_singlepose_lightning_tflite_float16_version4.zip"
)
_MOVENET_TFLITE_FILENAME = "movenet_lightning_fp16.tflite"


class MoveNetDetector:
    """
    Thin wrapper around MoveNet Lightning TFLite model for real-time pose
    estimation.  Uses a TFLite interpreter directly — no tensorflow_hub
    required, fully compatible with Python 3.12.

    The TFLite model (~2 MB) is downloaded once and cached in the models/
    directory.  Subsequent runs load from disk instantly.

    Usage
    -----
    detector = MoveNetDetector()
    pose_features, annotated_frame = detector.detect(frame_bgr)

    pose_features is a list of 34 floats: [x0,y0, x1,y1, ..., x16,y16]
    normalised to [0, 1] in frame coordinates.
    Returns None when no keypoint meets MOVENET_THRESHOLD.
    """

    # Where to cache the downloaded .tflite file
    _CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )

    def __init__(self) -> None:
        """Download (if needed) and load the MoveNet Lightning TFLite model."""
        import tensorflow as tf

        self._tf = tf
        model_path = self._ensure_model()

        # TF 2.20+ deprecates tf.lite.Interpreter — use ai_edge_litert if available
        try:
            from ai_edge_litert.interpreter import Interpreter
            self._interpreter = Interpreter(model_path=model_path)
        except ImportError:
            # Fallback: tf.lite.Interpreter still works in TF 2.21 with a warning
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._interpreter = tf.lite.Interpreter(model_path=model_path)

        self._interpreter.allocate_tensors()

        self._input_idx   = self._interpreter.get_input_details()[0]["index"]
        self._output_idx  = self._interpreter.get_output_details()[0]["index"]
        self._input_size  = self._interpreter.get_input_details()[0]["shape"][1]  # 192
        # Detect whether model expects uint8 (quantized) or float32
        self._input_dtype = self._interpreter.get_input_details()[0]["dtype"]

        # Warm-up pass
        dummy = np.zeros((1, self._input_size, self._input_size, 3), dtype=self._input_dtype)
        self._interpreter.set_tensor(self._input_idx, dummy)
        self._interpreter.invoke()
        print(f"  MoveNet Lightning ready (input: {self._input_size}x{self._input_size}, dtype: {self._input_dtype.__name__}).")

    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_model(self) -> str:
        """
        Return local path to the MoveNet TFLite file.

        Download priority:
        1. Already present in models/ → use immediately
        2. Try downloading from multiple known URLs
        3. If all fail → print manual download instructions and raise

        Returns
        -------
        str : local path to the .tflite file
        """
        import tensorflow as tf
        import urllib.request

        os.makedirs(self._CACHE_DIR, exist_ok=True)
        local_path = os.path.join(self._CACHE_DIR, _MOVENET_TFLITE_FILENAME)

        if os.path.exists(local_path):
            return local_path

        # Multiple candidate URLs — try each in order
        candidate_urls = [
            # TFHub lite-model direct download (version 4)
            "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
            # GitHub mirror (version 3 — same architecture, works fine)
            "https://github.com/SARIT42/MoveNet-Pose-Detector/raw/main/lite-model_movenet_singlepose_lightning_3.tflite",
            # Kaggle models CDN (version 1)
            "https://www.kaggle.com/models/google/movenet/TfLite/singlepose-lightning/1/download/1.tflite",
        ]

        print("  Downloading MoveNet Lightning TFLite (~2 MB)…")
        for url in candidate_urls:
            try:
                print(f"    Trying: {url[:70]}…")
                urllib.request.urlretrieve(url, local_path)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 100_000:
                    print(f"  ✅ MoveNet model saved: {local_path}")
                    return local_path
                else:
                    os.remove(local_path)  # incomplete file
            except Exception as e:
                print(f"    Failed: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                continue

        # All URLs failed — give clear manual instructions
        print("\n" + "="*60)
        print("  [WARNING] Auto-download failed. Manual step required:")
        print("  1. Open this URL in your browser:")
        print("     https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning")
        print("  2. Download any .tflite file")
        print(f"  3. Rename it to: {_MOVENET_TFLITE_FILENAME}")
        print(f"  4. Place it in:  {self._CACHE_DIR}")
        print("  5. Re-run the script")
        print("="*60 + "\n")
        raise FileNotFoundError(
            f"MoveNet TFLite model not found. "
            f"Please place '{_MOVENET_TFLITE_FILENAME}' in '{self._CACHE_DIR}'."
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Resize + pad a BGR frame for MoveNet input.

        Returns
        -------
        np.ndarray  shape (1, input_size, input_size, 3)  dtype uint8
        """
        import cv2
        h, w = frame_bgr.shape[:2]
        size = self._input_size

        # Resize with letterbox padding to preserve aspect ratio
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame_bgr[:, :, ::-1], (new_w, new_h))  # BGR→RGB

        # Pad to square
        pad_h = size - new_h
        pad_w = size - new_w
        top,  bottom = pad_h // 2, pad_h - pad_h // 2
        left, right  = pad_w // 2, pad_w - pad_w // 2
        padded = np.pad(resized, ((top, bottom), (left, right), (0, 0)),
                        mode="constant", constant_values=0)

        return padded[np.newaxis].astype(self._input_dtype)   # (1, size, size, 3)

    # ─────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        frame_bgr: np.ndarray,
        draw: bool = True,
    ) -> tuple[list[float] | None, np.ndarray]:
        """
        Run MoveNet on a single BGR frame and return pose features.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame from OpenCV.
        draw : bool
            If True, draws keypoint circles and skeleton lines on the frame.

        Returns
        -------
        pose_features : list[float] | None
            Flat list of 34 floats [x0,y0, x1,y1, ..., x16,y16] normalised
            to [0, 1] in frame coordinates.  Returns None if all keypoints
            are below MOVENET_THRESHOLD.
        annotated_frame : np.ndarray
            Frame with optional pose overlay.
        """
        import cv2

        input_tensor = self._preprocess(frame_bgr)
        self._interpreter.set_tensor(self._input_idx, input_tensor)
        self._interpreter.invoke()

        # Output shape: [1, 1, 17, 3]  → (y, x, confidence)
        keypoints = self._interpreter.get_tensor(self._output_idx)[0, 0]  # (17, 3)

        h, w = frame_bgr.shape[:2]
        pose_features: list[float] = []
        all_low_conf = True

        for kp in keypoints:
            ky, kx, kconf = float(kp[0]), float(kp[1]), float(kp[2])
            if kconf >= MOVENET_THRESHOLD:
                all_low_conf = False
                pose_features.extend([kx, ky])
            else:
                pose_features.extend([0.0, 0.0])

        if all_low_conf:
            return None, frame_bgr

        if draw:
            self._draw_pose(frame_bgr, keypoints, w, h)

        return pose_features, frame_bgr

    # ─────────────────────────────────────────────────────────────────────────

    def _draw_pose(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,   # shape (17, 3)
        w: int,
        h: int,
    ) -> None:
        """Draw keypoint dots and skeleton edges on the frame in-place."""
        import cv2

        POSE_COLOR     = (0, 165, 255)   # Orange
        SKELETON_COLOR = (255, 100, 0)   # Blue-ish

        EDGES = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]

        for i, j in EDGES:
            yi, xi, ci = keypoints[i]
            yj, xj, cj = keypoints[j]
            if ci >= MOVENET_THRESHOLD and cj >= MOVENET_THRESHOLD:
                cv2.line(frame,
                         (int(xi * w), int(yi * h)),
                         (int(xj * w), int(yj * h)),
                         SKELETON_COLOR, 2)

        for kp in keypoints:
            ky, kx, kconf = float(kp[0]), float(kp[1]), float(kp[2])
            if kconf >= MOVENET_THRESHOLD:
                cx, cy = int(kx * w), int(ky * h)
                cv2.circle(frame, (cx, cy), 5, POSE_COLOR, -1)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def normalise_pose(pose_features: list[float]) -> list[float]:
        """
        Normalise 34 raw pose features relative to the torso midpoint.

        Strategy
        --------
        1. Compute torso centre = mean of shoulders (5,6) + hips (11,12).
        2. Subtract centre from all (x,y) pairs → position-invariant.
        3. Scale by max absolute value → values in [-1, 1].

        Zero-confidence keypoints (0.0, 0.0) are kept as-is.

        Parameters
        ----------
        pose_features : list of 34 floats

        Returns
        -------
        list of 34 normalised floats
        """
        coords = np.array(pose_features, dtype=np.float32).reshape(NUM_POSE_KEYPOINTS, 2)

        torso_indices = [5, 6, 11, 12]
        torso_pts = coords[torso_indices]

        if np.any(torso_pts != 0):
            centre = torso_pts[torso_pts.any(axis=1)].mean(axis=0)
            coords = coords - centre

        scale = np.max(np.abs(coords))
        if scale > 0:
            coords = coords / scale

        return coords.flatten().tolist()
