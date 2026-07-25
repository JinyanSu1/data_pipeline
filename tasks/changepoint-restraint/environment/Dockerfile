# Agent-facing environment: CPU only, python + numpy, nothing else needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# curl/ca-certificates/git are here for the agent CLI's own installer, not for
# the task: the task itself is fully offline.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy>=1.26,<3"

WORKDIR /app
COPY app/ /app/

CMD ["sleep", "infinity"]
