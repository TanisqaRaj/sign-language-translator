# =============================================================================
# core/model_loader.py
# Model loading — thin re-export from inference.py
# =============================================================================
# All actual loading logic lives in inference.py (the original working module).
# This module gives callers a clean core.model_loader import path without
# duplicating any code.

from inference import (
    load_tflite_model,
    load_label_map,
    load_scaler,
    load_character_tflite_model,
    load_character_label_map,
    load_character_scaler,
)

__all__ = [
    "load_tflite_model",
    "load_label_map",
    "load_scaler",
    "load_character_tflite_model",
    "load_character_label_map",
    "load_character_scaler",
]
