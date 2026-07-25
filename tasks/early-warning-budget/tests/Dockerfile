FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# util-linux provides setpriv, which the verifier uses to run the agent's
# detector as nobody:nogroup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy==2.5.1"

# The verifier runs in a separate container built from THIS directory, and
# harbor skips the /tests upload in that mode, so the image must own /tests.
COPY . /tests/

# Everything root-owned; the ground truth is root-only.  The unprivileged
# child needs to read predict.py, nothing else.
RUN chown -R root:root /tests \
    && chmod 755 /tests \
    && chmod 644 /tests/*.py \
    && chmod 755 /tests/test.sh \
    && chmod 600 /tests/test_episodes.json \
    && mkdir -p /app /logs/verifier \
    && python3 -c "import numpy, sys; print('numpy', numpy.__version__, 'python', sys.version)" \
    && command -v setpriv \
    && id nobody

WORKDIR /tests
