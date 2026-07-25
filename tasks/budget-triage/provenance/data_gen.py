#!/usr/bin/env python3
"""Deterministic generator for the budget-triage alert pools.

Emits two JSON files drawn from the *same* dataset-level hyper-distribution
but from disjoint seeds:

  environment/app/dev_alerts.json   6 periods, labels visible to the agent
  tests/test_alerts.json           12 periods, labels hidden (root-only, 0600)

Design notes (why each knob exists)
-----------------------------------
* ``is_incident`` is a Bernoulli draw from a logistic model over a *sparse*
  weighted combination of standardised features plus two pairwise interactions
  and one quadratic term.  Sparse + interactions => learnable but not trivial;
  Bernoulli sampling => an irreducible Bayes error, so no ranker can classify
  perfectly from the features.
* Raw features are shifted/scaled per feature index, so a solver that forgets
  to standardise is handicapped.
* ``severity`` is log-linear in a *different* (partially overlapping) sparse
  weight vector, with a fat noise term.  It correlates with the features but
  cannot be read off them.
* ``age_days`` is a per-period mixture (exponential + uniform) with a
  per-period coupling coefficient ``delta`` whose *sign is random*.  Age is
  therefore partly informative about incident-ness and partly not, and
  "sort by age ascending" has no consistent edge across periods.
* Per-period randomisation of n, incident rate, the weight vectors (jittered
  around the dataset-level hyper-mean) and the age distribution makes the
  mapping learnable across periods but not memorisable period-by-period.

Run:  python3 provenance/data_gen.py           (writes both files)
      python3 provenance/data_gen.py --stats   (also prints a summary)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Task constants.  These are duplicated in tests/reward.py and
# environment/app/run_dev.py; they are part of the published task definition
# and must never drift.
# ---------------------------------------------------------------------------
N_FEATURES = 12
K_BUDGET = 20
T_MAX = 30.0
COST = 0.1

# Dataset-level ("hyper") seed.  Shared by dev and test so the two are drawn
# from one distribution; the per-period seeds are disjoint.
HYPER_SEED = 20260724
DEV_SEED = 101
TEST_SEED = 90210
N_DEV_PERIODS = 6
N_TEST_PERIODS = 12


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# ---------------------------------------------------------------------------
# Hyper-distribution
# ---------------------------------------------------------------------------
def make_hyper(seed: int = HYPER_SEED) -> dict[str, Any]:
    """Dataset-level structure shared by every period of both splits."""
    rng = np.random.default_rng(seed)
    d = N_FEATURES

    # Sparse incident weights: 5 of 12 features carry signal.
    inc_idx = rng.choice(d, size=5, replace=False)
    w_mu = np.zeros(d)
    w_mu[inc_idx] = rng.choice([-1.0, 1.0], size=5) * rng.uniform(0.75, 1.75, size=5)

    # Two pairwise interactions and one quadratic term.  At least one leg of
    # each interaction is drawn from the *non*-linear-signal features, so the
    # marginal association of that feature with the label is weak.
    pool = list(range(d))
    pair_a = tuple(int(v) for v in rng.choice(pool, size=2, replace=False))
    pair_b = tuple(int(v) for v in rng.choice(pool, size=2, replace=False))
    while pair_b == pair_a or pair_b == pair_a[::-1]:
        pair_b = tuple(int(v) for v in rng.choice(pool, size=2, replace=False))
    inter_c = rng.choice([-1.0, 1.0], size=2) * rng.uniform(0.55, 1.05, size=2)
    quad_idx = int(rng.choice(pool))
    quad_c = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.7))

    # Severity weights: 4 features, partially overlapping the incident set.
    sev_idx = rng.choice(d, size=4, replace=False)
    v_mu = np.zeros(d)
    v_mu[sev_idx] = rng.choice([-1.0, 1.0], size=4) * rng.uniform(0.10, 0.24, size=4)

    # Raw-feature location/scale, so the raw columns are not standardised.
    feat_loc = rng.uniform(-0.6, 0.6, size=d)
    feat_scale = rng.uniform(0.65, 1.7, size=d)

    return {
        "w_mu": w_mu,
        "inc_idx": np.sort(inc_idx),
        "pair_a": pair_a,
        "pair_b": pair_b,
        "inter_c": inter_c,
        "quad_idx": quad_idx,
        "quad_c": quad_c,
        "v_mu": v_mu,
        "sev_idx": np.sort(sev_idx),
        "feat_loc": feat_loc,
        "feat_scale": feat_scale,
    }


def _solve_bias(scores: np.ndarray, target_rate: float) -> float:
    """Bisect the intercept so that mean sigmoid(scores + b) == target_rate."""
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(sigmoid(scores + mid).mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# One period
# ---------------------------------------------------------------------------
def make_period(
    hyper: dict[str, Any],
    rng: np.random.Generator,
    period: int,
    first_id: int,
    *,
    n: int | None = None,
    rate: float | None = None,
    age_shift: float = 0.0,
    age_stretch: float = 1.0,
) -> list[dict[str, Any]]:
    d = N_FEATURES
    if n is None:
        n = int(rng.integers(350, 701))
    if rate is None:
        rate = float(rng.uniform(0.05, 0.10))

    # --- features -----------------------------------------------------------
    zx = rng.normal(0.0, 1.0, size=(n, d))
    raw = hyper["feat_loc"] + hyper["feat_scale"] * zx

    # --- per-period weight jitter around the hyper-mean ----------------------
    w_p = hyper["w_mu"] + rng.normal(0.0, 0.22, size=d)
    v_p = hyper["v_mu"] + rng.normal(0.0, 0.06, size=d)
    ic = hyper["inter_c"] * rng.normal(1.0, 0.18, size=2)
    qc = hyper["quad_c"] * float(rng.normal(1.0, 0.18))

    a1, a2 = hyper["pair_a"]
    b1, b2 = hyper["pair_b"]
    q = hyper["quad_idx"]

    score = (
        zx @ w_p
        + ic[0] * zx[:, a1] * zx[:, a2]
        + ic[1] * zx[:, b1] * zx[:, b2]
        + qc * (zx[:, q] ** 2 - 1.0)
    )
    bias = _solve_bias(score, rate)
    logit = score + bias
    p = sigmoid(logit)
    is_incident = rng.random(n) < p

    # --- severity: log-linear in a different sparse direction, noisy ---------
    # Deliberately moderate dispersion: severity must matter, but it must not
    # dominate the time-decay factor, or ignoring age_days would be nearly
    # free.  See README.md ("why these numbers").
    log_sev = 0.55 + zx @ v_p + rng.normal(0.0, 0.22, size=n)
    severity = np.clip(np.exp(log_sev), 0.15, 6.0)

    # --- age: per-period mixture, plus a random-sign coupling to the logit ---
    # Wide, so the decay factor max(0, 1 - age/30) spans the whole [0, 1]
    # range with a substantial point mass at 0 (fully aged-out alerts).
    mix_w = float(rng.uniform(0.30, 0.65))
    exp_scale = float(rng.uniform(8.0, 20.0))
    uni_hi = float(rng.uniform(38.0, 58.0))
    comp_exp = rng.exponential(exp_scale, size=n)
    comp_uni = rng.uniform(0.0, uni_hi, size=n)
    age = np.where(rng.random(n) < mix_w, comp_exp, comp_uni)

    # Coupling with a random sign: age is partly informative about
    # incident-ness in any single period, but the direction flips between
    # periods, so "sort by age" has no consistent edge.  Kept small on
    # purpose -- see README.md.
    delta = float(rng.normal(0.0, 0.7))
    zscore = (score - score.mean()) / (score.std() + 1e-9)
    age = age * age_stretch + delta * zscore + age_shift
    age = np.clip(age, 0.0, 60.0)

    alerts: list[dict[str, Any]] = []
    for i in range(n):
        alerts.append(
            {
                "id": first_id + i,
                "period": period,
                "features": [round(float(v), 5) for v in raw[i]],
                "age_days": round(float(age[i]), 3),
                "is_incident": bool(is_incident[i]),
                "severity": round(float(severity[i]), 5),
            }
        )
    return alerts


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def make_split(
    hyper: dict[str, Any],
    seed: int,
    n_periods: int,
    *,
    overrides: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    overrides = overrides or {}
    alerts: list[dict[str, Any]] = []
    next_id = 0
    for period in range(n_periods):
        kw = overrides.get(period, {})
        block = make_period(hyper, rng, period, next_id, **kw)
        next_id += len(block)
        alerts.extend(block)
    return {
        "meta": {
            "k": K_BUDGET,
            "t_max": T_MAX,
            "cost": COST,
            "n_features": N_FEATURES,
            "n_periods": n_periods,
            "hyper_seed": HYPER_SEED,
            "split_seed": seed,
        },
        "alerts": alerts,
    }


def summarise(data: dict[str, Any]) -> str:
    rows = []
    by_period: dict[int, list[dict[str, Any]]] = {}
    for a in data["alerts"]:
        by_period.setdefault(a["period"], []).append(a)
    for period in sorted(by_period):
        blk = by_period[period]
        inc = [a for a in blk if a["is_incident"]]
        ages = np.array([a["age_days"] for a in blk])
        rows.append(
            f"  period {period:2d}: n={len(blk):4d} incidents={len(inc):3d} "
            f"rate={len(inc) / len(blk):.3f} "
            f"age mean={ages.mean():5.1f} p90={np.percentile(ages, 90):5.1f} "
            f"frac_aged_out={float((ages >= T_MAX).mean()):.2f}"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--outdir", default=None, help="task root (default: parent of this file's dir)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = args.outdir or os.path.dirname(here)

    hyper = make_hyper()

    dev = make_split(hyper, DEV_SEED, N_DEV_PERIODS)

    # The test split adds two deliberate distribution stresses:
    #   period 4  -> unusually low incident rate (1.2%)
    #   period 9  -> unusually old alerts (shifted + stretched age distribution)
    test = make_split(
        hyper,
        TEST_SEED,
        N_TEST_PERIODS,
        overrides={
            4: {"rate": 0.012, "n": 520},
            9: {"age_shift": 16.0, "age_stretch": 1.15, "n": 560},
        },
    )

    dev_path = os.path.join(root, "environment", "app", "dev_alerts.json")
    test_path = os.path.join(root, "tests", "test_alerts.json")
    os.makedirs(os.path.dirname(dev_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    with open(dev_path, "w") as fh:
        json.dump(dev, fh, separators=(",", ":"))
        fh.write("\n")
    with open(test_path, "w") as fh:
        json.dump(test, fh, separators=(",", ":"))
        fh.write("\n")

    print(f"wrote {dev_path}  ({len(dev['alerts'])} alerts)")
    print(f"wrote {test_path} ({len(test['alerts'])} alerts)")
    if args.stats:
        print("dev:")
        print(summarise(dev))
        print("test:")
        print(summarise(test))
        print("hyper: informative feature idx =", list(hyper["inc_idx"]))
        print("hyper: severity feature idx    =", list(hyper["sev_idx"]))
        print("hyper: interactions            =", hyper["pair_a"], hyper["pair_b"])
        print("hyper: quadratic idx           =", hyper["quad_idx"])


if __name__ == "__main__":
    main()
