# =============================================================================
# core/recognition.py
# Gesture / character recognition helpers — thin re-export from inference.py
# and preprocess.py.
# =============================================================================
# All recognition algorithms, confidence filtering, stability logic, debounce
# logic, and PredictionSmoother live in the original inference.py / preprocess.py.
# Nothing is changed here — we only provide a clean import surface.

from inference import (
    predict,
    predict_character,
    PredictionSmoother,
)
from preprocess import normalise_landmarks

__all__ = [
    "predict",
    "predict_character",
    "PredictionSmoother",
    "normalise_landmarks",
]
