#!/usr/bin/env python3
"""Self-scoring harness for the dev split.

    python3 /app/run_dev.py                 score /app/solution.py on dev
    python3 /app/run_dev.py --solution X.py score another file
    python3 /app/run_dev.py --baselines     also print reference points

This reproduces the verifier's arithmetic exactly (same T_MAX, COST, k, same
optimum construction, same analytic floor).  The only differences are that
the verifier uses a larger, hidden 12-period split and runs your rank() under
an unprivileged user.

Nothing here is imported by the verifier -- it is a convenience for you.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

# --- scoring rule (kept identical to tests/reward.py) ----------------------
T_MAX = 30.0
COST = 0.1
PAYLOAD_KEYS = ("id", "features", "age_days", "period")

DEV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_alerts.json")


def utility(alert: dict) -> float:
    if alert["is_incident"]:
        return float(alert["severity"]) * max(
            0.0, 1.0 - float(alert["age_days"]) / T_MAX
        )
    return -COST


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def score(alerts: list[dict], order: list, k: int) -> dict:
    ids = [int(a["id"]) for a in alerts]
    if sorted(int(v) for v in order) != sorted(ids):
        return {"valid_ordering": False, "normalised": 0.0}

    rank_of = {int(v): i for i, v in enumerate(order)}
    by_period: dict[int, list[dict]] = {}
    for a in alerts:
        by_period.setdefault(int(a["period"]), []).append(a)

    tot_a = tot_o = tot_f = 0.0
    caught = investigated = available = 0
    caught_ages: list[float] = []
    rows = []
    for p in sorted(by_period):
        pool = by_period[p]
        chosen = sorted(pool, key=lambda a: rank_of[int(a["id"])])[:k]
        utils = [utility(a) for a in pool]
        a_val = sum(utility(a) for a in chosen)
        o_val = sum(sorted(utils, reverse=True)[:k])
        f_val = min(k, len(pool)) * (sum(utils) / len(pool))
        hits = [a for a in chosen if a["is_incident"]]
        caught_ages.extend(float(a["age_days"]) for a in hits)
        tot_a += a_val
        tot_o += o_val
        tot_f += f_val
        caught += len(hits)
        investigated += len(chosen)
        available += sum(1 for a in pool if a["is_incident"])
        den = o_val - f_val
        rows.append(
            {
                "period": p,
                "n": len(pool),
                "achieved": a_val,
                "optimal": o_val,
                "floor": f_val,
                "normalised": clamp((a_val - f_val) / den, 0.0, 1.0) if den > 1e-12 else 0.0,
                "precision_at_k": len(hits) / len(chosen) if chosen else 0.0,
                "caught": len(hits),
                "available": sum(1 for a in pool if a["is_incident"]),
            }
        )

    den = tot_o - tot_f
    return {
        "valid_ordering": True,
        "achieved": tot_a,
        "optimal": tot_o,
        "floor": tot_f,
        "normalised": clamp((tot_a - tot_f) / den, 0.0, 1.0) if den > 1e-12 else 0.0,
        "precision_at_k": caught / investigated if investigated else 0.0,
        "mean_age_of_caught_incidents": (
            sum(caught_ages) / len(caught_ages) if caught_ages else None
        ),
        "incidents_caught": caught,
        "incidents_available": available,
        "per_period": rows,
    }


def blind(alerts: list[dict]) -> list[dict]:
    """Exactly what the verifier hands to rank(): no labels."""
    return [{key: a[key] for key in PAYLOAD_KEYS} for a in alerts]


def load_rank(path: str):
    spec = importlib.util.spec_from_file_location("_candidate_solution", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "rank"):
        raise SystemExit(f"{path} defines no rank()")
    return mod.rank


def report(name: str, metrics: dict) -> None:
    if not metrics["valid_ordering"]:
        print(f"{name}: INVALID ORDERING -> reward 0.0")
        print("  The ordering must be a permutation of exactly the ids given:")
        print("  no duplicates, nothing missing, nothing unknown.")
        return
    print(f"--- {name} ---")
    print(f"  achieved                     {metrics['achieved']:10.4f}")
    print(f"  optimal                      {metrics['optimal']:10.4f}")
    print(f"  floor (random, analytic)     {metrics['floor']:10.4f}")
    print(f"  normalised score             {metrics['normalised']:10.4f}")
    print(f"  precision@k                  {metrics['precision_at_k']:10.4f}")
    age = metrics["mean_age_of_caught_incidents"]
    print(
        "  mean age of caught incidents "
        + ("       n/a" if age is None else f"{age:10.2f} days")
    )
    print(
        f"  incidents caught / available {metrics['incidents_caught']:5d}"
        f" / {metrics['incidents_available']}"
    )
    print("  per period:")
    for r in metrics["per_period"]:
        print(
            f"    period {r['period']}: n={r['n']:4d} "
            f"achieved={r['achieved']:8.3f} optimal={r['optimal']:8.3f} "
            f"floor={r['floor']:7.3f} norm={r['normalised']:.4f} "
            f"p@k={r['precision_at_k']:.2f} caught={r['caught']:2d}/{r['available']:2d}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default=os.path.join(os.path.dirname(DEV_PATH), "solution.py"))
    ap.add_argument("--data", default=DEV_PATH)
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()

    with open(args.data, "r") as fh:
        data = json.load(fh)
    alerts = data["alerts"]
    k = int(data.get("meta", {}).get("k", 20))
    print(
        f"dev split: {len(alerts)} alerts over "
        f"{len({a['period'] for a in alerts})} periods, k={k}, "
        f"T_MAX={T_MAX}, COST={COST}\n"
    )

    rank_fn = load_rank(args.solution)
    order = list(rank_fn(blind(alerts)))
    report(os.path.basename(args.solution), score(alerts, order, k))

    if args.baselines:
        print()
        oracle = [
            a["id"] for a in sorted(alerts, key=lambda a: (-utility(a), a["id"]))
        ]
        report("ORACLE (uses hidden labels; upper bound = 1.0)", score(alerts, oracle, k))
        print()
        report(
            "identity order (the starter)",
            score(alerts, [a["id"] for a in alerts], k),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
