#!/usr/bin/env python3
"""Bayes-optimal ranking for budget-triage: the real ceiling for any agent.

    .venv/bin/python analysis/bayes_optimal_budget_triage.py

The reward normalises against an `optimal` computed *from the hidden labels*.
Nothing without those labels can reach it, so "codex scored 0.64" leaves the
important question open: 0.64 of a range whose top is unreachable by how much?

This computes the score of an agent that knows the generative process exactly --
every per-period weight, the bias, the severity coefficients -- but, like any
real submission, does not know which alerts actually turned out to be incidents.
That is the true ceiling. The gap between it and 1.0 is irreducible; the gap
between an agent and it is what remains to be won.

The per-period parameters are drawn inside data_gen.make_period and never
saved, so they are recovered by replaying the generator's RNG call sequence
verbatim. Any drift in that sequence would silently yield the wrong parameters,
so the replay is checked against the real tests/test_alerts.json before it is
used -- see verify(). If that check fails, nothing below is meaningful.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

TASK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "budget-triage")
sys.path.insert(0, os.path.join(TASK, "provenance"))
sys.path.insert(0, os.path.join(TASK, "tests"))

import data_gen as G  # noqa: E402
import reward as R  # noqa: E402

SEV_LOG_SIGMA = 0.22  # rng.normal(0.0, 0.22) on log_sev in make_period
SEV_LO, SEV_HI = 0.15, 6.0  # np.clip bounds on severity
SEV_LOG_BASE = 0.55


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def expected_clipped_lognormal(mu: np.ndarray, sigma: float) -> np.ndarray:
    """E[clip(exp(mu + N(0, sigma)), SEV_LO, SEV_HI)], exactly.

    The clip is not cosmetic: with sigma=0.22 the upper bound bites on the
    strongest alerts, which are precisely the ones a ranker puts first. Ignoring
    it is the single modelling choice that separated codex's 0.64 runs from its
    0.51 runs.
    """
    a = (math.log(SEV_LO) - mu) / sigma
    b = (math.log(SEV_HI) - mu) / sigma
    below = SEV_LO * _norm_cdf(a)
    above = SEV_HI * (1.0 - _norm_cdf(b))
    middle = np.exp(mu + 0.5 * sigma**2) * (_norm_cdf(b - sigma) - _norm_cdf(a - sigma))
    return below + middle + above


def replay_period(hyper, rng, period, first_id, *, n=None, rate=None,
                  age_shift=0.0, age_stretch=1.0):
    """make_period, re-executed with the latent state kept.

    The RNG calls below must stay in the same order and shape as
    data_gen.make_period. verify() is what proves they did.
    """
    d = G.N_FEATURES
    if n is None:
        n = int(rng.integers(350, 701))
    if rate is None:
        rate = float(rng.uniform(0.05, 0.10))

    zx = rng.normal(0.0, 1.0, size=(n, d))
    raw = hyper["feat_loc"] + hyper["feat_scale"] * zx

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
    bias = G._solve_bias(score, rate)
    logit = score + bias
    p = G.sigmoid(logit)
    is_incident = rng.random(n) < p

    log_sev_mean = SEV_LOG_BASE + zx @ v_p
    log_sev = log_sev_mean + rng.normal(0.0, SEV_LOG_SIGMA, size=n)
    severity = np.clip(np.exp(log_sev), SEV_LO, SEV_HI)

    mix_w = float(rng.uniform(0.30, 0.65))
    exp_scale = float(rng.uniform(8.0, 20.0))
    uni_hi = float(rng.uniform(38.0, 58.0))
    comp_exp = rng.exponential(exp_scale, size=n)
    comp_uni = rng.uniform(0.0, uni_hi, size=n)
    age = np.where(rng.random(n) < mix_w, comp_exp, comp_uni)

    delta = float(rng.normal(0.0, 0.7))
    zscore = (score - score.mean()) / (score.std() + 1e-9)
    age = age * age_stretch + delta * zscore + age_shift
    age = np.clip(age, 0.0, 60.0)

    alerts = [
        {
            "id": first_id + i,
            "period": period,
            "features": [round(float(v), 5) for v in raw[i]],
            "age_days": round(float(age[i]), 3),
            "is_incident": bool(is_incident[i]),
            "severity": round(float(severity[i]), 5),
        }
        for i in range(n)
    ]
    # p and log_sev_mean are the whole point: they are what a Bayes-optimal
    # ranker would know and an agent must estimate.
    return alerts, {"p": p, "log_sev_mean": log_sev_mean}


def replay_test_split():
    hyper = G.make_hyper()
    rng = np.random.default_rng(G.TEST_SEED)
    overrides = {
        4: {"rate": 0.012, "n": 520},
        9: {"age_shift": 16.0, "age_stretch": 1.15, "n": 560},
    }
    alerts, latent = [], []
    next_id = 0
    for period in range(G.N_TEST_PERIODS):
        block, lat = replay_period(hyper, rng, period, next_id, **overrides.get(period, {}))
        next_id += len(block)
        alerts.extend(block)
        latent.append(lat)
    return alerts, latent


def verify(replayed: list[dict]) -> None:
    """Refuse to report a ceiling derived from the wrong parameters."""
    with open(os.path.join(TASK, "tests", "test_alerts.json")) as fh:
        real = json.load(fh)["alerts"]
    if len(real) != len(replayed):
        raise SystemExit(f"replay produced {len(replayed)} alerts, real set has {len(real)}")
    for a, b in zip(real, replayed):
        for key in ("id", "period", "age_days", "is_incident", "severity", "features"):
            if a[key] != b[key]:
                raise SystemExit(
                    f"replay diverged at alert {a['id']} on {key!r}: "
                    f"{a[key]!r} != {b[key]!r}. The recovered per-period "
                    "parameters are not the ones that made this dataset."
                )
    print(f"replay verified: {len(real)} alerts identical to tests/test_alerts.json")


def main() -> int:
    replayed, latent = replay_test_split()
    verify(replayed)

    by_period: dict[int, list[int]] = {}
    for i, a in enumerate(replayed):
        by_period.setdefault(a["period"], []).append(i)

    order: list[int] = []
    for period in sorted(by_period):
        idx = by_period[period]
        lat = latent[period]
        p = lat["p"]
        e_sev = expected_clipped_lognormal(lat["log_sev_mean"], SEV_LOG_SIGMA)
        decay = np.array([R.decay(replayed[i]["age_days"]) for i in idx])
        # Exactly the quantity reward.utility() takes an expectation of.
        ev = p * e_sev * decay - (1.0 - p) * R.COST
        for rank in np.argsort(-ev):
            order.append(replayed[idx[rank]]["id"])

    metrics = R.score(replayed, order, k=R.K_BUDGET)
    print()
    print(f"Bayes-optimal reward : {metrics['normalised']:.4f}")
    print(f"  achieved           : {metrics['achieved']:.3f}")
    print(f"  optimal (labels)   : {metrics['optimal']:.3f}")
    print(f"  floor  (random)    : {metrics['floor']:.3f}")
    print(f"  precision@k        : {metrics['precision_at_k']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
