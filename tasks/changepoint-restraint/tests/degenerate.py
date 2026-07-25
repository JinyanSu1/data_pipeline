#!/usr/bin/env python3
"""Anti-degenerate check: score the strategies that require no understanding.

    python3 tests/degenerate.py [--data tests/test_series.json] [--json]

Every strategy below must come out at (or essentially at) zero reward.  The
two ends of the range are the point of the task:

    empty       return []                       -> recall 0, so F1 0
    all         return every index              -> gate fires, reward 0
    grid8       ~8 evenly spaced indices         -> gate fires, reward 0
    midpoint    [len(series)//2]                -> gate fires, reward 0
    random      uniformly random indices        -> gate fires, reward 0

`[]` scoring 0 while `spray` also scores 0 is deliberate: silence and noise are
both failures, and only a detector that decides *per series* how many change
points there are can score.

The random strategy is seeded so this file prints a number, not a coin flip;
the verifier seeds `random` and `numpy.random` the same way before the agent's
code runs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grade  # noqa: E402
import reward as reward_mod  # noqa: E402

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_series.json")
GRID_POINTS = 8
RANDOM_SEED = 0
RANDOM_K = 4


def strat_empty(series, rng):
    return []


def strat_all(series, rng):
    return list(range(len(series)))


def strat_grid8(series, rng):
    n = len(series)
    step = n / (GRID_POINTS + 1.0)
    return [int(round(step * (i + 1))) for i in range(GRID_POINTS)]


def strat_midpoint(series, rng):
    return [len(series) // 2]


def strat_random(series, rng):
    n = len(series)
    return sorted(rng.sample(range(n), min(RANDOM_K, n)))


STRATEGIES = {
    "empty": strat_empty,
    "all": strat_all,
    "grid8": strat_grid8,
    "midpoint": strat_midpoint,
    "random": strat_random,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with open(args.data) as fh:
        data = json.load(fh)
    tol = int(data.get("tolerance", grade.TOL))
    series_meta = [
        {
            "id": s["id"],
            "length": int(s["length"]),
            "change_points": [int(c) for c in s["change_points"]],
            "is_null": bool(s.get("is_null", len(s["change_points"]) == 0)),
        }
        for s in data["series"]
    ]

    results = {}
    print(f"{'strategy':<10}{'f1':>8}{'prec':>8}{'rec':>8}{'null_fp':>9}{'gate':>7}{'reward':>9}")
    for name, fn in STRATEGIES.items():
        rng = random.Random(RANDOM_SEED)
        preds = {
            s["id"]: fn([list(row) for row in s["values"]], rng) for s in data["series"]
        }
        metrics = grade.score(series_meta, preds, tol=tol)
        value, gate_passed = reward_mod.compute_reward(metrics)
        results[name] = {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "null_fp_rate": metrics["null_fp_rate"],
            "null_gate_passed": gate_passed,
            "reward": value,
        }
        print(
            f"{name:<10}{metrics['f1']:>8.4f}{metrics['precision']:>8.4f}"
            f"{metrics['recall']:>8.4f}{metrics['null_fp_rate']:>9.2f}"
            f"{('pass' if gate_passed else 'FAIL'):>7}{value:>9.4f}"
        )

    worst = max(r["reward"] for r in results.values())
    print("")
    print(f"max degenerate reward = {worst:.4f}")
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if worst < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
