FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend:/app/src \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend
COPY src /app/src
COPY data/ingredient_normalization.csv data/ingredient_patterns.csv /app/data/
# Run as an unprivileged user; caches live in volumes owned by it.
RUN useradd --create-home --uid 10001 app && mkdir -p /app/.cache /home/app/.cache/huggingface \
    && chown -R app:app /app/.cache /home/app/.cache
USER app
WORKDIR /app/backend
EXPOSE 8000
# One worker: rate limits, circuit breaker and recommendation caches are per process.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'"]
