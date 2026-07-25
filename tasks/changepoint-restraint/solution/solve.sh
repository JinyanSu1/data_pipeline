#!/bin/sh
# Oracle: install the reference solution as the deliverable.
set -eu

cp /solution/solution.py /app/solution.py
python3 -c "import ast,sys; ast.parse(open('/app/solution.py').read())"
echo "reference solution installed at /app/solution.py"
