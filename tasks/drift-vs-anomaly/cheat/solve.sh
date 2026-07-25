#!/bin/bash
# Adversarial solve: install the reward-forging / label-hunting "solution".
# Expected verifier outcome: reward 0.0.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="${here}/solution.py"
[ -f "${src}" ] || src="/cheat/solution.py"

mkdir -p /app
cp "${src}" /app/solution.py
echo "installed cheat detector at /app/solution.py"
