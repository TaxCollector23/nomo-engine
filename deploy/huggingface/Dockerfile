# Hugging Face Docker Space for the Nomo backend (free CPU tier, no card required).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
# Spaces run containers as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user pyproject.toml setup.py README.md ./
COPY --chown=user csrc ./csrc
COPY --chown=user nomo ./nomo
RUN pip install --no-cache-dir --user pybind11 && pip install --no-cache-dir --user ".[server,coreml]"

ENV PORT=7860 MALLOC_ARENA_MAX=2 NOMO_MAX_ACTIVE_RUNS=2 NOMO_LOG_DIR=/tmp/nomo-logs PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "-c", "uvicorn nomo.telemetry.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips=* --no-access-log"]
