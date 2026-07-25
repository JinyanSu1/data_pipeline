#!/usr/bin/env python3
"""Turn an ordering into a reward file and a metrics file.

Separated from run_verifier.py so the scoring path can be exercised offline
(``python3 tests/grade.py --self-test``) without any sandboxing machinery.
Pure stdlib; never imports agent code.
"""

from __future__ import annotations

import json
import os
from typing import Any

import reward as reward_mod

LOG_DIR = "/logs/verifier"
REWARD_PATH = os.path.join(LOG_DIR, "reward.txt")
METRICS_PATH = os.path.join(LOG_DIR, "metrics.json")


def load_dataset(path: str) -> tuple[list[dict], int]:
    with open(path, "r") as fh:
        data = json.load(fh)
    alerts = data["alerts"]
    k = int(data.get("meta", {}).get("k", reward_mod.K_BUDGET))
    return alerts, k


def grade(alerts: list[dict], order: list, k: int) -> dict[str, Any]:
    return reward_mod.score(alerts, order, k=k)


def write_outputs(metrics: dict[str, Any], log_dir: str = LOG_DIR) -> float:
    """Write the scalar reward and the diagnostics.  Returns the reward."""
    os.makedirs(log_dir, exist_ok=True)
    value = float(metrics.get("normalised", 0.0))
    if not (value == value) or value in (float("inf"), float("-inf")):
        value = 0.0
    value = min(1.0, max(0.0, value))
    with open(os.path.join(log_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")
    # Scalar float only.  Never a reward.json.
    with open(os.path.join(log_dir, "reward.txt"), "w") as fh:
        fh.write(f"{value:.6f}")
    return value


def summary(metrics: dict[str, Any]) -> str:
    lines = [
        f"valid_ordering              : {metrics['valid_ordering']}",
        f"achieved                    : {metrics['achieved']:.4f}",
        f"optimal                     : {metrics['optimal']:.4f}",
        f"floor                       : {metrics['floor']:.4f}",
        f"normalised (reward)         : {metrics['normalised']:.4f}",
        f"precision@k                 : {metrics['precision_at_k']:.4f}",
        f"incidents caught / available: "
        f"{metrics['incidents_caught']} / {metrics['incidents_available']}",
    ]
    mean_age = metrics["mean_age_of_caught_incidents"]
    lines.append(
        "mean age of caught incidents: "
        + ("n/a" if mean_age is None else f"{mean_age:.2f} days")
    )
    if metrics.get("error"):
        lines.append(f"error                       : {metrics['error']}")
    lines.append("per period:")
    for row in metrics["per_period"]:
        if not row.get("valid", False):
            lines.append(f"  period {row['period']:2d}: INVALID ({row.get('error')})")
            continue
        lines.append(
            f"  period {row['period']:2d}: n={row['n_alerts']:4d} "
            f"achieved={row['achieved']:8.3f} optimal={row['optimal']:8.3f} "
            f"floor={row['floor']:7.3f} norm={row['normalised']:.4f} "
            f"p@k={row['precision_at_k']:.2f} "
            f"caught={row['incidents_caught']:2d}/{row['incidents_available']:2d}"
        )
    return "\n".join(lines)


def _self_test() -> None:
    """Exercise the scoring path on the dev split with a handful of orderings."""
    import random
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "test_alerts.json")
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(here), "environment", "app", "dev_alerts.json"
        )
    alerts, k = load_dataset(path)
    ids = [a["id"] for a in alerts]

    identity = list(ids)
    rnd = list(ids)
    random.Random(0).shuffle(rnd)
    oracle = [
        a["id"]
        for a in sorted(alerts, key=lambda a: -reward_mod.utility(a))
    ]

    for name, order in (("identity", identity), ("random", rnd), ("oracle", oracle)):
        m = grade(alerts, order, k)
        print(f"{name:9s} reward={m['normalised']:.6f} achieved={m['achieved']:.3f}")

    for name, bad in (
        ("duplicate", identity[:-1] + [identity[0]]),
        ("missing", identity[:-1]),
        ("unknown", identity[:-1] + [10 ** 9]),
    ):
        m = grade(alerts, bad, k)
        assert m["normalised"] == 0.0 and not m["valid_ordering"], name
        print(f"gate {name:9s} -> reward=0.0  ({m['error']})")
    print("self-test ok", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        ap.error("grade.py is a library; use --self-test or run run_verifier.py")
