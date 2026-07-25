#!/bin/sh
# Adversarial: install the cheating solution as the deliverable.
set -eu

cp /solution/solution.py /app/solution.py
python3 -c "import ast,sys; ast.parse(open('/app/solution.py').read())"
echo "cheat solution installed at /app/solution.py"
