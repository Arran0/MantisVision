# Serves ml/src/api (the inference service) — not used for training.
# Deploy target: Render (or any Docker host) with Root Directory set to `ml`.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY config.py .
COPY src ./src

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
