# Separate verifier environment.
#
# In separate-verifier mode harbor does NOT upload tests/ at run time: it uses
# this directory as the build context and expects the image to already own
# /tests. Hence the COPY below -- without it there is no /tests/test.sh to run.
#
# util-linux provides setpriv, which the verifier uses to run the agent's code
# as `nobody`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy>=1.26,<3"

COPY . /tests

# /tests stays traversable: the untrusted child has to read predict.py.
# The labels do not: root only. run_verifier.py re-applies this at run time.
RUN chmod 755 /tests \
    && chmod 700 /tests/test.sh \
    && chmod 644 /tests/predict.py /tests/grade.py /tests/reward.py \
                 /tests/run_verifier.py /tests/degenerate.py \
    && chown root:root /tests/test_series.json \
    && chmod 600 /tests/test_series.json

WORKDIR /app

CMD ["sleep", "infinity"]
