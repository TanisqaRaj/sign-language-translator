# ─────────────────────────────────────────────────────────────────────────────
# config.py
# Central configuration file for the Sign Language Translator project.
# All constants, paths, and hyperparameters are defined here so you only
# need to change one file when tweaking the project.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join(BASE_DIR, "dataset")      # Raw gesture images
LANDMARKS_CSV    = os.path.join(BASE_DIR, "landmarks.csv") # Extracted landmark data
IMAGES_PER_CLASS = 300          # Number of images to collect per gesture
IMG_SIZE         = (224, 224)   # Resize target for CNN (if used)

# ── Gesture Classes ───────────────────────────────────────────────────────────
# Add or remove gesture labels here — everything else auto-updates.
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

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MAX_NUM_HANDS        = 1        # Detect only one hand for speed
MIN_DETECTION_CONF   = 0.7      # Minimum hand detection confidence
MIN_TRACKING_CONF    = 0.5      # Minimum hand tracking confidence
NUM_LANDMARKS        = 21       # MediaPipe Hands always returns 21 landmarks
LANDMARK_FEATURES    = NUM_LANDMARKS * 2   # x, y per landmark = 42 features

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_DIR         = os.path.join(BASE_DIR, "models")
MODEL_KERAS_PATH  = os.path.join(MODEL_DIR, "gesture_model.keras")
MODEL_TFLITE_PATH = os.path.join(MODEL_DIR, "gesture_model.tflite")
LABEL_MAP_PATH    = os.path.join(MODEL_DIR, "label_map.json")
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.pkl")

# ── Training Hyperparameters ──────────────────────────────────────────────────
EPOCHS          = 100
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
VALIDATION_SPLIT= 0.20          # 20% of data for validation
TEST_SPLIT      = 0.10          # 10% of data for testing
DROPOUT_RATE    = 0.4
RANDOM_SEED     = 42

# ── Real-Time Inference ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.75    # Minimum confidence to accept a prediction
STABLE_FRAME_COUNT    = 15      # Frames a prediction must hold before TTS fires
PREDICTION_BUFFER_LEN = 10      # Rolling buffer for smoothing predictions
CAMERA_INDEX          = 0       # Webcam device index

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_RATE   = 150                # Speech rate (words per minute)
TTS_VOLUME = 1.0                # Volume: 0.0 – 1.0

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
