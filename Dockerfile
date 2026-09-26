# Nomo backend v3. Works on Render (Docker runtime), Hugging Face Spaces, Railway, Fly.io, any VM.
# Listens on $PORT (Render injects it; default 8765). Logs: one JSON object per line on stdout.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY csrc ./csrc
COPY nomo ./nomo
RUN pip install --no-cache-dir pybind11 && pip install --no-cache-dir ".[server]"

ENV PORT=8765 \
    NOMO_LOG_DIR=/tmp/nomo-logs \
    NOMO_MAX_ACTIVE_RUNS=2 \
    PYTHONUNBUFFERED=1
EXPOSE 8765
# Single process: runs, telemetry rings and the admin log rings live in memory.
# --no-access-log: Nomo writes its own structured access log (stream "access").
CMD ["sh", "-c", "uvicorn nomo.telemetry.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
