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

# ── MoveNet (Body Pose) ───────────────────────────────────────────────────────
# MoveNet Lightning: fast single-pose model, suitable for real-time on CPU.
# TF-Hub URL — loaded once at startup, cached locally after first download.
MOVENET_MODEL_URL  = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
MOVENET_INPUT_SIZE = 192      # MoveNet Lightning expects 192×192 RGB input
MOVENET_THRESHOLD  = 0.2      # Keypoints below this confidence are zeroed out

# MoveNet returns 17 keypoints; we keep (x, y) per keypoint = 34 pose features
NUM_POSE_KEYPOINTS = 17
POSE_FEATURES      = NUM_POSE_KEYPOINTS * 2   # 34 features

# Human-readable keypoint names (MoveNet ordering, 0-indexed)
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
# MediaPipe hand landmarks (42) + MoveNet body pose (34) = 76 total features.
# All downstream scripts (preprocess, train, inference) use HYBRID_FEATURES.
HYBRID_FEATURES    = LANDMARK_FEATURES + POSE_FEATURES   # 42 + 34 = 76

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
