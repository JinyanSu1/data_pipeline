#!/usr/bin/env python3
"""Unprivileged bridge to the agent's /app/solution.py.

Run by run_verifier.py under::

    setpriv --reuid nobody --regid nogroup --clear-groups python3 /tests/predict.py \
        <payload.json> <out_fd>

This is the ONLY process that ever imports agent code.  It is deliberately
tiny and deliberately unprivileged:

  * the payload it is handed contains only id / features / age_days / period,
    never a label;
  * the hidden dataset (/tests/test_alerts.json) is mode 0600 root, so this
    process cannot read it;
  * /logs/verifier is mode 0700 root, so this process cannot forge a reward;
  * the result is written through a file descriptor inherited from root whose
    directory is mode 0700 root, so this process cannot redirect, replace or
    symlink-swap the output path.

It also pins `random` and `numpy.random` to seed 0 before touching agent code,
so that the same submitted artifact always yields the same reward.

Anything the agent's rank() prints goes to stdout/stderr and is ignored.
"""

from __future__ import annotations

import json
import os
import random
import sys
import traceback

SOLUTION_DIR = "/app"


def emit(fd: int, payload: dict) -> None:
    os.write(fd, json.dumps(payload).encode("utf-8"))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: predict.py <payload.json> <out_fd>", file=sys.stderr)
        return 2
    payload_path = sys.argv[1]
    out_fd = int(sys.argv[2])

    try:
        with open(payload_path, "r") as fh:
            alerts = json.load(fh)
    except Exception as exc:  # pragma: no cover - infrastructure failure
        emit(out_fd, {"ok": False, "error": f"could not read payload: {exc!r}"})
        return 1

    # Pin every RNG the agent is likely to reach for, ONCE, before any agent
    # code is imported or called.  Reason: the same submitted artifact must
    # produce the same reward on every run.  A solution that calls
    # random.random() or numpy.random.* without seeding would otherwise score
    # differently run to run, which makes the eval non-reproducible.
    # (Note: this is unrelated to the `floor` term in the reward, which is the
    # ANALYTIC expected utility of a random order, never a sampled one.)
    random.seed(0)
    try:
        import numpy as _np

        _np.random.seed(0)
    except ImportError:
        pass

    try:
        if SOLUTION_DIR not in sys.path:
            sys.path.insert(0, SOLUTION_DIR)
        import solution  # noqa: PLC0415  (agent-supplied module)

        if not hasattr(solution, "rank"):
            emit(out_fd, {"ok": False, "error": "/app/solution.py defines no rank()"})
            return 1
        order = solution.rank(alerts)
    except BaseException:
        emit(
            out_fd,
            {
                "ok": False,
                "error": "rank() raised",
                "traceback": traceback.format_exc(limit=20),
            },
        )
        return 1

    # Normalise whatever came back into plain JSON ints without trusting
    # __iter__/__index__ side effects beyond a single pass.
    try:
        if hasattr(order, "tolist"):  # numpy array / pandas index
            order = order.tolist()
        order = list(order)
        clean: list[int] = []
        for v in order:
            if isinstance(v, bool):
                clean.append(int(v))
            elif isinstance(v, int):
                clean.append(v)
            elif isinstance(v, float) and float(v).is_integer():
                clean.append(int(v))
            else:
                clean.append(int(v))  # last resort; raises for junk
    except BaseException as exc:
        emit(out_fd, {"ok": False, "error": f"rank() returned unusable data: {exc!r}"})
        return 1

    emit(out_fd, {"ok": True, "order": clean})
    return 0


if __name__ == "__main__":
    sys.exit(main())
