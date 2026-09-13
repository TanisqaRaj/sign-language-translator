# =============================================================================
# Dockerfile – Sign Language Translator (production)
#
# Uses app_cloud.py (streamlit-webrtc browser webcam, no pyttsx3)
# and requirements_cloud.txt (tflite-runtime, NOT full TensorFlow).
# This keeps the final image under ~1 GB instead of ~3 GB.
#
# Build:
#   docker build -t sign-lang-translator .
#
# Run locally:
#   docker run -p 8501:8501 sign-lang-translator
#
# Run with env overrides:
#   docker run -p 8501:8501 --env-file .env sign-lang-translator
# =============================================================================

FROM python:3.10-slim

# ---------------------------------------------------------------------------
# 1. System dependencies
#
#    OpenCV:
#      libgl1, libglib2.0-0, libsm6, libxext6, libxrender-dev
#
#    TFLite runtime:
#      libgomp1 — OpenMP, required by tflite-runtime on Linux
#
#    aiortc / PyAV (video codec support for streamlit-webrtc):
#      ffmpeg        — runtime codecs (H.264 decode, VP8/VP9)
#      libavcodec-dev, libavformat-dev, libavdevice-dev — headers for PyAV build
#      libvpx-dev    — VP8/VP9 codec (WebRTC default video codec)
#      libopus-dev   — Opus audio codec (WebRTC default audio codec)
#      libsrtp2-dev  — SRTP encryption required by WebRTC
#
#    HTTPS / WebRTC STUN:
#      ca-certificates
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ca-certificates \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavdevice-dev \
        libvpx-dev \
        libopus-dev \
        libsrtp2-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Create a non-root user for security
# ---------------------------------------------------------------------------
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/bash appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# 3. Install Python dependencies FIRST (layer cache benefit)
#    requirements_cloud.txt uses tflite-runtime instead of full TensorFlow
#    → image is ~800 MB lighter
# ---------------------------------------------------------------------------
COPY requirements_cloud.txt ./requirements_cloud.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_cloud.txt

# ---------------------------------------------------------------------------
# 4. Copy application source
#    .dockerignore prevents dataset/, logs/, models/*.keras, __pycache__, etc.
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# 5. Ensure the logs directory is writable
# ---------------------------------------------------------------------------
RUN mkdir -p logs \
    && chown -R appuser:appgroup /app

# ---------------------------------------------------------------------------
# 6. Switch to non-root user
# ---------------------------------------------------------------------------
USER appuser

# ---------------------------------------------------------------------------
# 7. Environment variables
#    These are safe defaults; override at runtime with --env-file .env
# ---------------------------------------------------------------------------
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# ---------------------------------------------------------------------------
# 8. Health check
#    Hits Streamlit's built-in /_stcore/health endpoint every 30 s.
#
#    start-period=60s: gives the container time to:
#      - load mediapipe (first-time model download: ~5-10 s)
#      - load tflite word model (~5 s)
#      - load tflite character model (~5 s)
#      - start Streamlit server (~5 s)
#    Total cold-start is typically 25-40 s; 60 s gives comfortable headroom.
#
#    Without a long enough start-period the container is marked "unhealthy"
#    before Streamlit is ready, which causes CI smoke tests and CD health
#    checks to fail even though the app will eventually start correctly.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# ---------------------------------------------------------------------------
# 9. Start the cloud-compatible Streamlit app
# ---------------------------------------------------------------------------
CMD ["streamlit", "run", "app_cloud.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
