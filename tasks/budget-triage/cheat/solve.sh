#!/bin/sh
# Install the adversarial 'solution'.  Must score 0.0.
#
# Also does, from the agent container, everything a cheating agent would try
# before the verifier ever starts: pre-write a reward file, and leave a stray
# copy of itself around.  None of it survives, because the verifier runs in a
# separate container with a fresh /logs/verifier.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

mkdir -p /app
cp "$here/solution.py" /app/solution.py
chmod 644 /app/solution.py

# Pre-forge a reward in the agent container (separate from the verifier's).
mkdir -p /logs/verifier 2>/dev/null || true
printf '1.0' > /logs/verifier/reward.txt 2>/dev/null || true
printf '{"reward": 1.0}' > /logs/verifier/reward.json 2>/dev/null || true

echo "installed cheat solution at /app/solution.py"
