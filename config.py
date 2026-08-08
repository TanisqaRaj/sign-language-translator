# =============================================================================
# config.py  –  Central Configuration
# Sign Language Translator – Enhanced Edition
# =============================================================================
# All constants, paths, and feature flags live here.
# Nothing else needs to change when you tweak a setting.
# =============================================================================

import os

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join(BASE_DIR, "dataset")
LANDMARKS_CSV    = os.path.join(BASE_DIR, "landmarks.csv")
IMAGES_PER_CLASS = 300
IMG_SIZE         = (224, 224)

# ── Gesture Classes ───────────────────────────────────────────────────────────
GESTURE_LABELS = [
    "Hello",
    "Thank You",
    "Yes",
    "No",
    "Please",
    "Sorry",
    "Good",
    "Bad",
    "Help",
    "I Love You",
]
NUM_CLASSES = len(GESTURE_LABELS)

# Human-readable meanings for the AI Assistant panel
GESTURE_MEANINGS = {
    "Hello":     "A greeting gesture",
    "Thank You": "Expressing gratitude",
    "Yes":       "Affirmative response",
    "No":        "Negative response",
    "Please":    "Polite request",
    "Sorry":     "Apology",
    "Good":      "Positive acknowledgement",
    "Bad":       "Negative acknowledgement",
    "Help":      "Asking for assistance",
    "I Love You":"Expressing love / ILY sign",
}

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MAX_NUM_HANDS        = 1
MIN_DETECTION_CONF   = 0.7
MIN_TRACKING_CONF    = 0.5
NUM_LANDMARKS        = 21
LANDMARK_FEATURES    = NUM_LANDMARKS * 2   # 42

# ── MoveNet ───────────────────────────────────────────────────────────────────
MOVENET_MODEL_URL  = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
MOVENET_INPUT_SIZE = 192
MOVENET_THRESHOLD  = 0.2
NUM_POSE_KEYPOINTS = 17
POSE_FEATURES      = NUM_POSE_KEYPOINTS * 2   # 34

POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye",  "right_eye",
    "left_ear",  "right_ear",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]

# ── Hybrid Feature Vector ─────────────────────────────────────────────────────
HYBRID_FEATURES = LANDMARK_FEATURES + POSE_FEATURES   # 76

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_DIR         = os.path.join(BASE_DIR, "models")
MODEL_KERAS_PATH  = os.path.join(MODEL_DIR, "gesture_model.keras")
MODEL_TFLITE_PATH = os.path.join(MODEL_DIR, "gesture_model.tflite")
LABEL_MAP_PATH    = os.path.join(MODEL_DIR, "label_map.json")
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.pkl")

# ── Training Hyperparameters ──────────────────────────────────────────────────
EPOCHS           = 100
BATCH_SIZE       = 32
LEARNING_RATE    = 0.001
VALIDATION_SPLIT = 0.20
TEST_SPLIT       = 0.10
DROPOUT_RATE     = 0.4
RANDOM_SEED      = 42

# ── Real-Time Inference ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.75
STABLE_FRAME_COUNT    = 15
PREDICTION_BUFFER_LEN = 10
CAMERA_INDEX          = 0

# Confidence level below which "Did you mean?" suggestions appear
LOW_CONFIDENCE_THRESHOLD = 0.70

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_RATE   = 150
TTS_VOLUME = 1.0

# ── Translation Languages ─────────────────────────────────────────────────────
# Maps display name → ISO 639-1 code used by deep-translator
SUPPORTED_LANGUAGES = {
    "English":    "en",
    "Hindi":      "hi",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Japanese":   "ja",
    "Korean":     "ko",
    "Chinese":    "zh-CN",
    "Arabic":     "ar",
    "Tamil":      "ta",
    "Telugu":     "te",
    "Bengali":    "bn",
    "Marathi":    "mr",
}

DEFAULT_LANGUAGE = "English"

# ── UI / Theme ────────────────────────────────────────────────────────────────
# Available themes: "dark" | "light"
DEFAULT_THEME = "dark"

# Font size modes: "normal" | "large"
DEFAULT_FONT_SIZE = "normal"

# High contrast mode
DEFAULT_HIGH_CONTRAST = False

# Auto-speak translated text when a gesture is confirmed
AUTO_SPEAK_DEFAULT = True

# Show live subtitle strip
SUBTITLE_ENABLED_DEFAULT = True

# ── Navigation Pages ─────────────────────────────────────────────────────────
NAV_PAGES = [
    "🏠 Home",
    "📷 Live Translator",
    "📜 History",
    "📊 Analytics",
    "⚙️ Settings",
    "ℹ️ About",
]

# ── Analytics ────────────────────────────────────────────────────────────────
# How many recent session data points to keep in memory
ANALYTICS_MAX_HISTORY = 500

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
