#!/bin/bash
# Oracle solve: install the reference detector as the submitted artifact.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="${here}/solution.py"
[ -f "${src}" ] || src="/solution/solution.py"

mkdir -p /app
cp "${src}" /app/solution.py
echo "installed reference detector at /app/solution.py"
