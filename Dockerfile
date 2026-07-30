# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile – Sign Language Translator (local Docker deployment)
#
# Build:  docker build -t sign-lang-translator .
# Run:    docker run -p 8501:8501 sign-lang-translator
#
# NOTE: Webcam access inside Docker requires --device /dev/video0 on Linux.
#       pyttsx3 TTS is disabled in container (no audio device).
#       For cloud deployment use app_cloud.py instead of app.py.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit config – disable file watcher (not needed in container)
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
