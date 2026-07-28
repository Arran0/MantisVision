# Serves ml/src/api (the inference service) — not used for training.
# Deploy target: Render (or any Docker host) with Root Directory set to `ml`.
FROM python:3.12-slim

WORKDIR /app

# opencv-python (pulled in transitively by grad-cam, used when ENABLE_GRADCAM=true)
# links against these X11/GL shared libraries at import time; python:3.12-slim
# doesn't ship them, so `import cv2` fails with "libxcb.so.1: cannot open
# shared object file" the first time a request actually triggers Grad-CAM.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY config.py .
COPY src ./src

ENV PORT=8000
# Keep native math libraries single-threaded and Grad-CAM off by default so the
# service fits a 512 MB free tier. Raise these (and ENABLE_GRADCAM=true) on a
# larger instance. A single uvicorn worker keeps only one model copy in memory.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    ENABLE_GRADCAM=false \
    PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
