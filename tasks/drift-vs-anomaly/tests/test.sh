#!/bin/bash
# Entry point for the Harbor verifier.  Runs as root in the separate verifier
# container.  Must always leave a scalar reward in /logs/verifier/reward.txt.
set -u

mkdir -p /logs/verifier
chmod 700 /logs/verifier
printf '0.0' > /logs/verifier/reward.txt

# Hidden labels are root-only; the unprivileged prediction child must not be
# able to read them.  /tests itself stays traversable so it can read predict.py.
chmod 755 /tests 2>/dev/null || true
chmod 600 /tests/test_episodes.json 2>/dev/null || true

python3 /tests/run_verifier.py
rc=$?

if [ ! -s /logs/verifier/reward.txt ]; then
    printf '0.0' > /logs/verifier/reward.txt
fi

# Never let a stray reward.json (which Harbor would prefer over reward.txt)
# survive into grading.
rm -f /logs/verifier/reward.json

exit "${rc}"
