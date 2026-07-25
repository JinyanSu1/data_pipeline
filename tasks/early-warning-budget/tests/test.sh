#!/bin/sh
# Verifier entrypoint.  Runs as root in the separate verifier container.
#
# The reward file is (re)written unconditionally on every code path, so a
# reward.txt the agent may have planted in the bind-mounted /logs during its
# own phase is always overwritten, and a crash still leaves a valid scalar.
set -u

mkdir -p /logs/verifier
chmod 700 /logs/verifier

# Harbor prefers reward.json over reward.txt; make sure only reward.txt exists.
rm -f /logs/verifier/reward.json /logs/verifier/rewards.json
printf '0.0\n' > /logs/verifier/reward.txt

# Ground truth stays root-only.
chmod 600 /tests/test_episodes.json 2>/dev/null || true

python3 /tests/run_verifier.py
status=$?

if [ ! -s /logs/verifier/reward.txt ]; then
    printf '0.0\n' > /logs/verifier/reward.txt
fi
chmod 644 /logs/verifier/reward.txt 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true

echo "run_verifier.py exited with status ${status}"
exit 0
