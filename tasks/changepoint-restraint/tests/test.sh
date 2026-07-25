#!/bin/sh
# Harbor entry point. Runs as root in the separate verifier container.
set -eu

mkdir -p /logs/verifier
chmod 700 /logs/verifier

exec python3 /tests/run_verifier.py
