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
#    - libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender-dev : OpenCV
#    - libgomp1 : required by tflite-runtime on Linux
#    - ca-certificates : HTTPS for deep-translator, WebRTC STUN
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ca-certificates \
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
# 5. Ensure the models directory is readable and logs directory writable
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
#    First check starts after a 15 s warm-up period.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# ---------------------------------------------------------------------------
# 9. Start the cloud-compatible Streamlit app
# ---------------------------------------------------------------------------
CMD ["streamlit", "run", "app_cloud.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
