FROM python:3.12-slim

# Numpy is the only third-party dependency the task allows; it is installed at
# build time so the task itself runs entirely offline on CPU.
RUN pip install --no-cache-dir "numpy>=1.26,<3"

WORKDIR /app
COPY app/ /app/
