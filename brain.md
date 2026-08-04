# 🧠 brain.md — Project Master Context
> Real-Time Sign Language Translator  
> Last updated: August 2026  
> Status: **Hybrid Pipeline Implemented — Ready to Run**

---

## 1. Project ka Goal

Ek real-time application banana jo:
1. Webcam se live video capture kare
2. Haath ke landmarks detect kare (MediaPipe)
3. Body pose detect kare (MoveNet)
4. Dono ko combine karke gesture classify kare (TFLite Dense NN)
5. Predicted word screen pe show kare
6. pyttsx3 se bol ke sunaye (Text-to-Speech)

**Target:** 10 ASL-inspired gestures, 25+ FPS, CPU-only, browser UI via Streamlit.

---

## 2. Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Language | Python | 3.10+ |
| Computer Vision | OpenCV | 4.9 |
| Hand Detection | MediaPipe Hands | 0.10 |
| Body Pose | MoveNet Lightning (TF-Hub) | v4 |
| Deep Learning | TensorFlow / Keras | 2.16 |
| TF-Hub | tensorflow-hub | 0.16.1 |
| Web UI | Streamlit | 1.35 |
| Text-to-Speech | pyttsx3 | 2.90 |
| Data | NumPy, Pandas, scikit-learn | latest |

---

## 3. 10 Supported Gestures

| # | Gesture | Description |
|---|---------|-------------|
| 1 | Hello | Open palm wave |
| 2 | Thank You | Flat hand from chin forward |
| 3 | Yes | Fist nodding |
| 4 | No | Index + middle fingers shaking |
| 5 | Please | Flat hand on chest, circular |
| 6 | Sorry | Fist on chest, circular |
| 7 | Good | Thumbs up |
| 8 | Bad | Thumbs down |
| 9 | Help | Fist on open palm, lift |
| 10 | I Love You | ILY handshape |

Nayi gesture add karni ho to sirf `config.py` mein `GESTURE_LABELS` mein naam add karo — baaki sab auto-update ho jaata hai.

---

## 4. Folder Structure

```
sign-language-translator/
├── brain.md               ← YEH FILE — project ka master context
├── config.py              ← Sab constants aur paths ek jagah
├── requirements.txt       ← Pinned dependencies
│
├── collect_data.py        ← Phase 1: Gesture images collect karo
├── preprocess.py          ← Phase 2: Hybrid features extract karo (76)
├── train_model.py         ← Phase 3: DNN model train karo
├── convert_tflite.py      ← Phase 4: Keras → TFLite convert karo
├── inference.py           ← Phase 5: Real-time OpenCV window
├── app.py                 ← Phase 5 (alt): Streamlit web UI
│
├── utils/
│   ├── mediapipe_helper.py  ← HandDetector class (MediaPipe wrapper)
│   ├── movenet_helper.py    ← MoveNetDetector class (MoveNet wrapper) ← NEW
│   ├── tts_engine.py        ← Thread-safe TTS wrapper
│   ├── performance.py       ← FPSCounter, ThreadedCapture
│   └── logger.py            ← Rotating file + console logger
│
├── dataset/               ← Gesture image folders (collect_data.py se banta hai)
├── models/                ← gesture_model.keras, .tflite, scaler.pkl, label_map.json
├── logs/                  ← app.log
├── docs/                  ← accuracy_plot.png, loss_plot.png, confusion_matrix.png
└── tests/
    └── test_pipeline.py   ← Unit + integration tests
```

---

## 5. Hybrid Pipeline — Core Idea

### Pehle kya tha (v1 — MediaPipe only)
```
Webcam Frame
    ↓
MediaPipe Hands → 21 landmarks → 42 features (x,y × 21)
    ↓
Normalise (wrist = origin, scale to [-1,1])
    ↓
TFLite DNN (42 → 256 → 128 → 64 → 10)
    ↓
Gesture + Confidence
```

### Ab kya hai (v2 — Hybrid: MediaPipe + MoveNet)
```
Webcam Frame
    ↓
    ├──→ MediaPipe Hands → 21 landmarks → normalise → 42 hand features
    │
    └──→ MoveNet Lightning → 17 body keypoints → normalise → 34 pose features
    ↓
Concatenate → 76-feature Hybrid Vector  [hand(42) + pose(34)]
    ↓
StandardScaler (same as training)
    ↓
TFLite DNN (76 → 256 → 128 → 64 → 10)
    ↓
PredictionSmoother (rolling majority vote, buffer=10)
    ↓
Gesture (stable for 15 frames) → Sentence Builder → TTS
```

---

## 6. Feature Vector Detail

```
Index 0  – 41  : MediaPipe hand landmarks
                 hx0,hy0, hx1,hy1, ..., hx20,hy20
                 (wrist-relative, scaled to [-1,1])

Index 42 – 75  : MoveNet body pose keypoints
                 px0,py0, px1,py1, ..., px16,py16
                 (torso-centred, scaled to [-1,1])
                 Zero-filled if person not visible in frame

Total          : 76 features per sample
```

### MoveNet 17 Keypoints (order)
```
0=nose  1=left_eye  2=right_eye  3=left_ear  4=right_ear
5=left_shoulder   6=right_shoulder
7=left_elbow      8=right_elbow
9=left_wrist     10=right_wrist
11=left_hip      12=right_hip
13=left_knee     14=right_knee
15=left_ankle    16=right_ankle
```

---

## 7. Model Architecture

```
Input(76)  ← HYBRID_FEATURES = 42 + 34
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.4)
    ↓
Dense(128, relu) → BatchNorm → Dropout(0.3)
    ↓
Dense(64,  relu) → Dropout(0.2)
    ↓
Dense(10, softmax)   ← NUM_CLASSES = 10
```

**Why DNN and not CNN or LSTM?**
- CNN — 2D spatial data ke liye hai (images). 1D vector pe kaam nahi karta.
- LSTM — temporal sequences ke liye hai. Hum static pose classify karte hain.
- DNN — tabular/structured data ke liye perfect. 76-feature vector = tabular data.

**Why TFLite?**
- Full Keras se 3× faster inference
- Float16 quantisation se ~50% chhota model
- Android/iOS/Raspberry Pi pe bhi chalta hai bina changes ke

---

## 8. Key Config Values (config.py)

```python
GESTURE_LABELS       = [10 gestures]
IMAGES_PER_CLASS     = 300          # images per gesture

# MediaPipe
LANDMARK_FEATURES    = 42           # 21 × 2 (x,y)
MIN_DETECTION_CONF   = 0.7

# MoveNet
MOVENET_MODEL_URL    = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
MOVENET_INPUT_SIZE   = 192
MOVENET_THRESHOLD    = 0.2          # keypoints below this → zeroed
POSE_FEATURES        = 34           # 17 × 2 (x,y)

# Hybrid
HYBRID_FEATURES      = 76           # 42 + 34

# Training
EPOCHS               = 100
BATCH_SIZE           = 32
LEARNING_RATE        = 0.001
DROPOUT_RATE         = 0.4

# Inference
CONFIDENCE_THRESHOLD = 0.75         # model ka min confidence
STABLE_FRAME_COUNT   = 15           # kitne frames tak gesture hold karo
PREDICTION_BUFFER_LEN= 10           # rolling smoother buffer size
```

---

## 9. Phases — Step by Step

### Phase 1 — Collect Data
**Script:** `collect_data.py`  
**Command:** `python collect_data.py`

- Webcam se 300 images/gesture collect karo
- Menu se ek gesture choose karo ya sab ek saath
- Blurry frames automatically reject hote hain
- Images `dataset/<GestureName>/` mein save hoti hain
- Config: `IMAGES_PER_CLASS = 300` in config.py

---

### Phase 2 — Extract Hybrid Features
**Script:** `preprocess.py`  
**Command:** `python preprocess.py`

- Har image pe MediaPipe + MoveNet dono chalata hai
- 76-feature hybrid vector extract karta hai
- `landmarks.csv` save karta hai (76 features + label per row)
- `landmarks.npy` save karta hai (fast loading ke liye)
- `models/label_map.json` save karta hai

**Output CSV columns:**
```
label, hx0, hy0, ..., hx20, hy20, px0, py0, ..., px16, py16
```
*Note: MoveNet first run pe ~12 MB download karega (cached baad mein)*

---

### Phase 3 — Train Model
**Script:** `train_model.py`  
**Command:** `python train_model.py`

- `landmarks.csv` load karta hai
- Train/Val/Test split karta hai (70/20/10, stratified)
- StandardScaler fit karta hai (training data pe only)
- DNN train karta hai (76-feature input, 10-class output)
- Best model save karta hai: `models/gesture_model.keras`
- Scaler save karta hai: `models/scaler.pkl`
- Plots save karta hai: `docs/accuracy_plot.png`, `loss_plot.png`, `confusion_matrix.png`

**Callbacks:**
- `ModelCheckpoint` — best val_accuracy pe save
- `EarlyStopping` — patience=15, val_loss monitor
- `ReduceLROnPlateau` — patience=7, factor=0.5

---

### Phase 4 — Convert to TFLite
**Script:** `convert_tflite.py`  
**Command:** `python convert_tflite.py`

- Keras model → TFLite convert karta hai
- Float16 quantisation apply karta hai (~50% size reduction)
- Saves: `models/gesture_model.tflite`

---

### Phase 5 — Real-Time Inference
**Option A (Streamlit UI):** `streamlit run app.py`  
**Option B (OpenCV window):** `python inference.py`

**Per-frame flow:**
1. Webcam frame capture → flip (mirror)
2. MediaPipe Hands → 42 hand features
3. MoveNet Lightning → 34 pose features
4. Concatenate → 76-feature vector
5. StandardScaler transform
6. TFLite forward pass
7. PredictionSmoother majority vote
8. Stable 15 frames → word added to sentence
9. HUD overlay draw karo
10. Key press handle karo (Q/C/S)

**Controls:**
- `Q` — quit
- `C` — sentence clear karo
- `S` — current sentence speak karo (force)

**HUD mein kya dikhta hai:**
- Gesture name + confidence
- Stability progress bar
- Pose indicator dot (green = MoveNet active, red = not detected)
- FPS counter
- Running sentence at bottom

---

## 10. Files Changed for Hybrid Pipeline

| File | Change |
|------|--------|
| `config.py` | MoveNet constants + `HYBRID_FEATURES=76` add kiya |
| `utils/movenet_helper.py` | **Naya file** — MoveNetDetector class |
| `preprocess.py` | Rewrite — 76-feature hybrid extraction |
| `train_model.py` | Input dim 42 → 76, HYBRID_FEATURES import |
| `inference.py` | Rewrite — dono detectors real-time mein |
| `requirements.txt` | `tensorflow-hub==0.16.1` add kiya |

---

## 11. Install & Run Commands

```bash
# 1. Dependencies install karo
pip install -r requirements.txt

# 2. Data collect karo (webcam chahiye)
python collect_data.py

# 3. Hybrid features extract karo
#    (pehli baar MoveNet ~12 MB download karega)
python preprocess.py

# 4. Model train karo
python train_model.py

# 5. TFLite convert karo
python convert_tflite.py

# 6. Real-time inference chalao
streamlit run app.py
# OR
python inference.py

# 7. Tests chalao
python -m pytest tests/ -v
```

---

## 12. Expected Performance

| Metric | Expected Value |
|--------|---------------|
| Test accuracy | >95% |
| Inference latency | ~10–15 ms (TFLite CPU) |
| MediaPipe latency | ~15 ms |
| MoveNet latency | ~8–12 ms (192×192) |
| End-to-end FPS | 20–30 FPS (CPU) |
| Model size (Keras) | ~2 MB |
| Model size (TFLite) | ~1 MB |

*Note: Hybrid pipeline MoveNet ki wajah se ~5-8 FPS kam hoga vs v1 (42-feature). Still real-time.*

---

## 13. Known Limitations

- **Static poses only** — motion-based gestures (waving, circling) supported nahi hain
- **Single hand** — ek haath detect hota hai (`MAX_NUM_HANDS=1`)
- **Single person** — MoveNet `singlepose` hai — ek hi person detect karta hai
- **Self-collected data** — performance vary kar sakta hai different hand shapes pe
- **Background** — bohot cluttered background MediaPipe confuse kar sakta hai

---

## 14. Future Plans

- [ ] LSTM layer add karo for motion-based gestures (temporal sequence)
- [ ] Two-hand support (`MAX_NUM_HANDS=2`, double feature vector)
- [ ] Data augmentation (flip, rotate, jitter) for better generalisation
- [ ] Full ASL alphabet (26 letters + digits)
- [ ] Mobile deployment via TFLite Android
- [ ] MoveNet Thunder (heavier but more accurate) for better pose
- [ ] Confidence calibration (temperature scaling)

---

## 15. Normalisation Logic

### Hand Landmarks (MediaPipe)
```python
# Wrist (landmark 0) → origin
coords = coords - coords[0]
# Scale max abs → 1.0
coords = coords / max(abs(coords))
# Result: wrist at (0,0), all values in [-1, 1]
```

### Body Pose (MoveNet)
```python
# Torso centre = mean of shoulders + hips (indices 5,6,11,12)
centre = mean(torso_keypoints)
coords = coords - centre
# Scale max abs → 1.0
coords = coords / max(abs(coords))
# Result: torso-centred, all values in [-1, 1]
# Low-confidence keypoints (<0.2) → zeroed out
```

---

## 16. Prediction Pipeline Detail

```
Raw prediction → confidence check (>0.75?)
    ↓ YES
PredictionSmoother.update(gesture, confidence)
    ↓
Deque (last 10 predictions) → majority vote
    ↓ majority found
stable_frames counter++
    ↓ stable_frames == 15
sentence.append(gesture)  if not duplicate
    ↓
TTSEngine.speak(gesture)  — background thread, no blocking
```

---

*brain.md — always update karo jab bhi koi major change ho.*
