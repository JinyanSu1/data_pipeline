FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# stdlib + numpy only.  Nothing else is needed and nothing else is installed:
# the task is CPU-only and must be solvable offline.
RUN pip install --no-cache-dir "numpy==2.3.4"

WORKDIR /app
COPY app/ /app/

RUN chmod 755 /app && chmod 644 /app/solution.py /app/run_dev.py /app/dev_alerts.json

CMD ["sleep", "infinity"]
