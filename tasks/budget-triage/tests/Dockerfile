FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# util-linux provides setpriv, which drops the child that executes the agent's
# rank() to nobody:nogroup.  Root never imports agent code.
RUN apt-get update \
    && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*

# The agent may legitimately use numpy in its solution.
RUN pip install --no-cache-dir "numpy==2.3.4"

COPY . /tests/

# /tests is traversable but not listable by the unprivileged child: it can
# exec /tests/predict.py by exact path and nothing else.  The hidden labels and
# the scoring code are root-only.
#
# run_verifier.py re-applies all of this at runtime, before any agent code
# runs, because some Harbor providers re-upload tests/ into the container at
# verifier start and would otherwise reset these bits.
RUN chmod 0711 /tests \
    && chmod 0644 /tests/predict.py \
    && chmod 0600 /tests/reward.py /tests/grade.py \
                  /tests/run_verifier.py /tests/degenerate.py \
    && chmod 0700 /tests/test.sh \
    && chmod 0600 /tests/test_alerts.json \
    && chown -R root:root /tests \
    && mkdir -p /app && chmod 0755 /app

WORKDIR /tests
CMD ["sleep", "infinity"]
