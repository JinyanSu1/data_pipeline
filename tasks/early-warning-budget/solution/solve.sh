#!/bin/sh
# Oracle: install the reference detector as the agent's submission.
set -eu

SRC="$(dirname "$0")/solution.py"
[ -f "$SRC" ] || SRC=/solution/solution.py

mkdir -p /app
cp "$SRC" /app/solution.py
chmod 644 /app/solution.py
echo "installed reference detector at /app/solution.py"
