#!/usr/bin/env python3
"""Anti-degeneracy harness for budget-triage.

Scores a family of label-blind heuristics against the hidden test split and
prints their rewards.  Every one of them must come out at ~0: if any of them
scores materially above zero the task can be passed without doing the actual
work (estimating p(incident) x severity x time-decay).

Also checks two structural properties of the scoring rule:

  --verify-optimal   brute-forces the best top-k subset on small pools and
                     confirms it equals the sort-by-true-utility construction,
                     i.e. `optimal` really is a provable optimum.
  (always)           an oracle ranker scores exactly 1.0, and a ranker that
                     identifies incidents *perfectly* but ignores age_days
                     scores clearly below 1.0 -- proof that the time decay is
                     load-bearing.

Usage:
    python3 tests/degenerate.py
    python3 tests/degenerate.py --verify-optimal
    python3 tests/degenerate.py --solution ../solution/solution.py
    python3 tests/degenerate.py --data ../environment/app/dev_alerts.json
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grade as grade_mod  # noqa: E402
import reward as reward_mod  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "test_alerts.json")


# --------------------------------------------------------------------------
# Label-blind degenerate strategies (only id / features / age_days / period)
# --------------------------------------------------------------------------
def s_identity(alerts):
    return [a["id"] for a in alerts]


def s_reverse(alerts):
    return [a["id"] for a in reversed(alerts)]


def s_random(alerts, seed=0):
    ids = [a["id"] for a in alerts]
    random.Random(seed).shuffle(ids)
    return ids


def s_age_asc(alerts):
    return [a["id"] for a in sorted(alerts, key=lambda a: (a["age_days"], a["id"]))]


def s_age_desc(alerts):
    return [a["id"] for a in sorted(alerts, key=lambda a: (-a["age_days"], a["id"]))]


def s_feature0_desc(alerts):
    return [a["id"] for a in sorted(alerts, key=lambda a: (-a["features"][0], a["id"]))]


def s_feature0_asc(alerts):
    return [a["id"] for a in sorted(alerts, key=lambda a: (a["features"][0], a["id"]))]


# --------------------------------------------------------------------------
# Label-using reference points (NOT achievable by an agent; diagnostics only)
# --------------------------------------------------------------------------
def s_oracle(alerts):
    return [
        a["id"]
        for a in sorted(alerts, key=lambda a: (-reward_mod.utility(a), a["id"]))
    ]


def s_perfect_cls_ignore_age(alerts):
    """Knows is_incident and severity exactly, but ignores age_days."""
    return [
        a["id"]
        for a in sorted(
            alerts,
            key=lambda a: (0 if a["is_incident"] else 1, -a["severity"], a["id"]),
        )
    ]


def s_perfect_cls_only(alerts):
    """Knows is_incident exactly; ignores both severity and age_days."""
    return [
        a["id"]
        for a in sorted(alerts, key=lambda a: (0 if a["is_incident"] else 1, a["id"]))
    ]


def load_external_rank(path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_external_solution", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    def strategy(alerts):
        blind = [
            {k: a[k] for k in ("id", "features", "age_days", "period")} for a in alerts
        ]
        return list(mod.rank(blind))

    return strategy


# --------------------------------------------------------------------------
# Optimality proof
# --------------------------------------------------------------------------
def verify_optimal(alerts, trials: int = 200, pool_n: int = 14, sub_k: int = 5) -> bool:
    """Brute-force max over all k-subsets vs. the sort-by-utility construction.

    The agent's per-period score depends only on WHICH k alerts land in the
    top k, so the achievable maximum is max over all C(n, k) subsets.  If the
    greedy "sort by u, take k" matches that on every sampled pool, then
    `optimal` as computed in reward.py is a true optimum, not a heuristic.
    """
    rng = random.Random(12345)
    ok = True
    for t in range(trials):
        pool = rng.sample(alerts, pool_n)
        utils = [reward_mod.utility(a) for a in pool]
        greedy = sum(sorted(utils, reverse=True)[:sub_k])
        brute = max(
            sum(c) for c in itertools.combinations(utils, sub_k)
        )
        if abs(greedy - brute) > 1e-9:
            print(f"  MISMATCH on trial {t}: greedy={greedy!r} brute={brute!r}")
            ok = False
    print(
        f"  brute-forced {trials} random pools of n={pool_n}, k={sub_k} "
        f"(C({pool_n},{sub_k})={len(list(itertools.combinations(range(pool_n), sub_k)))}"
        f" subsets each): "
        + ("ALL MATCH -> `optimal` is a provable optimum" if ok else "MISMATCH")
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--solution", default=None, help="also score this rank() file")
    ap.add_argument("--verify-optimal", action="store_true")
    ap.add_argument("--random-seeds", type=int, default=8)
    args = ap.parse_args()

    alerts, k = grade_mod.load_dataset(args.data)
    print(f"data: {args.data}")
    print(
        f"      {len(alerts)} alerts, "
        f"{len(reward_mod.group_by_period(alerts))} periods, k={k}, "
        f"T_MAX={reward_mod.T_MAX}, COST={reward_mod.COST}"
    )
    print()

    rows: list[tuple[str, float, float, float]] = []

    def run(name, order):
        m = reward_mod.score(alerts, order, k=k)
        rows.append(
            (name, m["normalised"], m["precision_at_k"], m["achieved"])
        )
        return m

    print("=== label-blind degenerate strategies (all must be ~0) ===")
    run("identity (as given)", s_identity(alerts))
    run("reverse", s_reverse(alerts))
    for seed in range(args.random_seeds):
        run(f"random seed={seed}", s_random(alerts, seed))
    run("sort age_days ascending", s_age_asc(alerts))
    run("sort age_days descending", s_age_desc(alerts))
    run("sort feature[0] descending", s_feature0_desc(alerts))
    run("sort feature[0] ascending", s_feature0_asc(alerts))

    blind_end = len(rows)

    print("=== label-using reference points (diagnostics, not achievable) ===")
    run("ORACLE (true utility)", s_oracle(alerts))
    run("perfect classifier, ignores age", s_perfect_cls_ignore_age(alerts))
    run("perfect classifier only", s_perfect_cls_only(alerts))

    if args.solution:
        print(f"=== external solution: {args.solution} ===")
        strategy = load_external_rank(args.solution)
        run(f"solution: {os.path.basename(args.solution)}", strategy(alerts))

    print()
    print(f"{'strategy':<34} {'reward':>9} {'prec@k':>8} {'achieved':>10}")
    print("-" * 64)
    for name, norm, prec, ach in rows:
        print(f"{name:<34} {norm:9.4f} {prec:8.3f} {ach:10.2f}")
    print()

    blind = rows[:blind_end]
    worst = max(blind, key=lambda r: r[1])
    print(f"worst-case label-blind degenerate: {worst[0]} at reward {worst[1]:.4f}")

    rand_rewards = [r[1] for r in blind if r[0].startswith("random seed=")]
    print(
        f"random over {len(rand_rewards)} seeds: "
        f"min={min(rand_rewards):.4f} max={max(rand_rewards):.4f} "
        f"mean={sum(rand_rewards) / len(rand_rewards):.4f}"
    )

    oracle_reward = next(r[1] for r in rows if r[0].startswith("ORACLE"))
    ignore_age = next(r[1] for r in rows if r[0].startswith("perfect classifier, ignores"))
    print(f"oracle reward: {oracle_reward:.6f}  (must be 1.0)")
    print(
        f"perfect-classifier-ignoring-age reward: {ignore_age:.4f}  "
        f"(must be clearly < 1.0 -- this is the proof that time decay matters)"
    )

    if args.verify_optimal:
        print()
        print("=== optimality proof ===")
        verify_optimal(alerts)

    problems = []
    if worst[1] > 0.15:
        problems.append(f"degenerate {worst[0]} scores {worst[1]:.4f} (> 0.15)")
    if abs(oracle_reward - 1.0) > 1e-9:
        problems.append(f"oracle scores {oracle_reward} (should be exactly 1.0)")
    if ignore_age > 0.90:
        problems.append(
            f"perfect-classifier-ignoring-age scores {ignore_age:.4f} (>0.90): "
            "the time decay is not doing enough work; increase age dispersion"
        )
    print()
    if problems:
        for p in problems:
            print("FAIL:", p)
        return 1
    print("OK: all degenerate strategies are ~0, oracle is 1.0, age matters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
