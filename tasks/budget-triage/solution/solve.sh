#!/bin/sh
# Oracle solution: install the reference ranker as /app/solution.py.
#
# The reference is self-contained -- its logistic-regression and severity
# coefficients are baked into the file -- because the verifier container gets
# /app/solution.py and nothing else.  Regenerate them with:
#   python3 solution/solution.py --refit environment/app/dev_alerts.json
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

mkdir -p /app
cp "$here/solution.py" /app/solution.py
chmod 644 /app/solution.py

echo "installed reference solution at /app/solution.py"
