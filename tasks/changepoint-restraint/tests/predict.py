#!/usr/bin/env python3
"""Untrusted side of the verifier.

Runs as `nobody` (dropped from root by run_verifier.py via setpriv) and is the
ONLY place where the agent's `/app/solution.py` is imported.  It reads a
label-free payload (series values only), calls `find_changepoints` once per
series, and writes back nothing but predicted index lists.

Usage: predict.py <payload.json> <predictions.json>
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import traceback

SOLUTION_PATH = "/app/solution.py"
PER_SERIES_NOTE = "exception"


def seed_everything():
    """Make the run reproducible before any agent code is imported or called.

    The same artifact must yield the same reward on every run; a solution that
    calls random.random() or numpy.random.* without seeding would otherwise
    score differently run to run.  Seeded once here at child start, not per
    series.
    """
    random.seed(0)
    try:
        import numpy as _np
    except ImportError:
        return
    _np.random.seed(0)


def load_solution(path):
    spec = importlib.util.spec_from_file_location("agent_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_solution"] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "find_changepoints", None)
    if not callable(fn):
        raise AttributeError("solution.py does not define find_changepoints(series)")
    return fn


def coerce(result):
    """Best-effort conversion to a list of ints; the root re-sanitises anyway."""
    out = []
    if result is None:
        return out
    try:
        iterator = list(result)
    except TypeError:
        return out
    for item in iterator[:100000]:
        try:
            if isinstance(item, bool):
                continue
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: predict.py <payload.json> <predictions.json>", file=sys.stderr)
        return 2
    payload_path, out_path = sys.argv[1], sys.argv[2]

    with open(payload_path) as fh:
        payload = json.load(fh)

    seed_everything()

    predictions = {}
    errors = {}
    try:
        find_changepoints = load_solution(SOLUTION_PATH)
    except Exception:
        traceback.print_exc()
        for item in payload.get("series", []):
            predictions[item["id"]] = []
        with open(out_path, "w") as fh:
            json.dump({"predictions": predictions, "errors": {"__import__": "failed"}}, fh)
        return 0

    for item in payload.get("series", []):
        sid = item["id"]
        series = [list(row) for row in item["values"]]
        try:
            predictions[sid] = coerce(find_changepoints(series))
        except Exception:
            traceback.print_exc()
            errors[sid] = PER_SERIES_NOTE
            predictions[sid] = []

    with open(out_path, "w") as fh:
        json.dump({"predictions": predictions, "errors": errors}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
