# Nomo telemetry/search backend. Any container host with WebSocket support works
# (Render, Railway, Fly.io, Cloud Run, a VM). Listens on $PORT (default 8765).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY csrc ./csrc
COPY nomo ./nomo
RUN pip install --no-cache-dir pybind11 && pip install --no-cache-dir ".[server]"

ENV PORT=8765 NOMO_MAX_ACTIVE_RUNS=2 NOMO_CORS_ORIGINS=*
EXPOSE 8765
# One process only: runs and their telemetry ring buffers live in memory.
CMD ["sh", "-c", "uvicorn nomo.telemetry.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
