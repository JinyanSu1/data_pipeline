#!/bin/sh
# Verifier entrypoint.  Runs as root in the separate verifier container.
#
# Always (re)writes /logs/verifier/reward.txt, on every code path, so a reward
# file the agent may have left behind is never trusted.
set -u

mkdir -p /logs/verifier
chmod 700 /logs/verifier
printf '0.000000' > /logs/verifier/reward.txt

/usr/local/bin/python3 /tests/run_verifier.py
status=$?

if [ ! -s /logs/verifier/reward.txt ]; then
    printf '0.000000' > /logs/verifier/reward.txt
fi

if [ "$status" -ne 0 ]; then
    echo "run_verifier.py exited with status $status" >&2
fi

# The reward file is the contract; a non-zero verifier exit would abort the
# trial before the reward is read, so exit clean.
exit 0
