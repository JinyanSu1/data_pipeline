"""Reference solution for budget-triage.

Competent, not optimal.  It exists to prove the task is solvable and to give a
sane comparison point; it deliberately does NOT define full marks.  The
provable optimum in the scoring rule is the ceiling, and this scores roughly
half way from random to it.

Method
------
Everything below is fitted offline on the *labelled dev split*
(``/app/dev_alerts.json``, 6 periods) with pure numpy -- no sklearn:

1. Standardise the 12 raw features with the dev mean/std (the raw columns are
   neither centred nor scaled, so this step matters).
2. L2-regularised logistic regression (Adam, 4000 steps) on ``is_incident``
   gives p_hat = P(incident | features).
3. Ridge regression on ``log(severity)`` gives sev_hat, de-biased for the
   lognormal transform with the +0.5*sigma^2 correction.
4. Rank by the *expected utility* of investigating, which is exactly the
   quantity the scoring rule sums:

       EV = p_hat * sev_hat * max(0, 1 - age_days / T_MAX) - (1 - p_hat) * COST

   Sorting by p_hat alone scores ~0.40; the full EV scores ~0.50.  The gap is
   the time-decay and severity terms earning their keep.

Why the coefficients are baked in
---------------------------------
The verifier container receives ``/app/solution.py`` and nothing else -- the
dev split is NOT present at scoring time.  A submission must therefore be
self-contained.  The fitting code that produced ``_MODEL`` is kept in this
file and can be re-run:

    python3 solution/solution.py --refit environment/app/dev_alerts.json

which re-prints the ``_MODEL`` literal below.
"""

from __future__ import annotations

import json
import math

T_MAX = 30.0
COST = 0.1
N_FEATURES = 12

# Fitted on environment/app/dev_alerts.json by fit() below.
_MODEL = {
    "mu": [
        0.43899,
        -0.3724,
        -0.0356,
        -0.17776,
        0.04002,
        0.23692,
        -0.52676,
        0.54236,
        -0.20727,
        -0.59854,
        0.29879,
        0.11298
    ],
    "sd": [
        1.30774,
        0.89347,
        1.1996,
        0.72623,
        1.42584,
        1.15551,
        1.49906,
        1.67819,
        1.47123,
        1.53656,
        0.75665,
        0.86859
    ],
    "w_inc": [
        -3.50454,
        -0.08027,
        0.70328,
        -0.6516,
        -0.07011,
        -0.13479,
        0.51336,
        0.04068,
        -0.1802,
        0.07939,
        -0.50165,
        -0.10803,
        1.08827
    ],
    "w_sev": [
        0.54435,
        -0.00444,
        -0.19537,
        -0.21377,
        -0.01141,
        -0.02036,
        0.01633,
        -4e-05,
        0.03509,
        -0.02023,
        0.2377,
        -0.14471,
        0.03562
    ],
    "sev_var": 0.0819
}


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _decay(age_days: float) -> float:
    return max(0.0, 1.0 - float(age_days) / T_MAX)


def _expected_utility(alert: dict) -> float:
    mu = _MODEL["mu"]
    sd = _MODEL["sd"]
    w_inc = _MODEL["w_inc"]
    w_sev = _MODEL["w_sev"]

    feats = alert["features"]
    z_inc = w_inc[0]
    z_sev = w_sev[0]
    for j in range(min(N_FEATURES, len(feats))):
        zj = (float(feats[j]) - mu[j]) / sd[j]
        z_inc += w_inc[j + 1] * zj
        z_sev += w_sev[j + 1] * zj

    p = _sigmoid(z_inc)
    sev = math.exp(z_sev + 0.5 * _MODEL["sev_var"])
    return p * sev * _decay(alert["age_days"]) - (1.0 - p) * COST


def rank(alerts) -> list[int]:
    """Rank alerts best-first by estimated expected utility.

    Returns a permutation of exactly the ids handed in.  Sorting globally is
    equivalent to sorting within each period, because the verifier restricts
    this ordering to one period at a time.
    """
    scored = [(-_expected_utility(a), int(a["id"])) for a in alerts]
    scored.sort()
    return [aid for _, aid in scored]


# ---------------------------------------------------------------------------
# Training.  Not used at scoring time; kept here so _MODEL is reproducible.
# ---------------------------------------------------------------------------
def fit(dev_alerts: list[dict], l2_inc: float = 1e-2, l2_sev: float = 1e-2) -> dict:
    import numpy as np

    x_raw = np.array([a["features"] for a in dev_alerts], dtype=float)
    mu = x_raw.mean(axis=0)
    sd = x_raw.std(axis=0) + 1e-9
    z = (x_raw - mu) / sd
    n = z.shape[0]
    zb = np.hstack([np.ones((n, 1)), z])

    y = np.array([1.0 if a["is_incident"] else 0.0 for a in dev_alerts])

    # --- logistic regression, Adam, L2 on the slopes only -------------------
    w = np.zeros(zb.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.15
    for t in range(1, 4001):
        p = 1.0 / (1.0 + np.exp(-np.clip(zb @ w, -30.0, 30.0)))
        g = zb.T @ (p - y) / n
        reg = l2_inc * w
        reg[0] = 0.0
        g = g + reg
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)

    # --- ridge on log severity ---------------------------------------------
    log_sev = np.log(np.array([a["severity"] for a in dev_alerts], dtype=float))
    a_mat = zb.T @ zb + l2_sev * n * np.eye(zb.shape[1])
    a_mat[0, 0] -= l2_sev * n
    w_sev = np.linalg.solve(a_mat, zb.T @ log_sev)
    resid = log_sev - zb @ w_sev

    return {
        "mu": [round(float(x), 5) for x in mu],
        "sd": [round(float(x), 5) for x in sd],
        "w_inc": [round(float(x), 5) for x in w],
        "w_sev": [round(float(x), 5) for x in w_sev],
        "sev_var": round(float(resid.var()), 5),
    }


def _refit(path: str) -> None:
    with open(path, "r") as fh:
        dev = json.load(fh)["alerts"]
    model = fit(dev)
    print("_MODEL = " + json.dumps(model, indent=4))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", metavar="DEV_ALERTS_JSON")
    args = ap.parse_args()
    if args.refit:
        _refit(args.refit)
    else:
        ap.error("nothing to do; pass --refit <dev_alerts.json>")
