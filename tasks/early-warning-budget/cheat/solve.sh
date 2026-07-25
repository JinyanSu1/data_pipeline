#!/bin/sh
# Adversarial: install the cheating detector as the agent's submission.
# Expected reward: 0.0.
set -eu

SRC="$(dirname "$0")/solution.py"
[ -f "$SRC" ] || SRC=/solution/solution.py

mkdir -p /app
cp "$SRC" /app/solution.py
chmod 644 /app/solution.py
echo "installed cheat detector at /app/solution.py"
