"""Authoritative scoring for the budget-triage task.

Pure stdlib.  Imported by run_verifier.py (root, verifier-side) and by
degenerate.py.  It never imports agent code and never touches the filesystem.

Utility of investigating alert ``a``::

    u(a) = severity(a) * max(0, 1 - age_days(a) / T_MAX)   if is_incident(a)
         = -COST                                            otherwise

Per period the agent's ordering is restricted to that period and the top
K_BUDGET entries are "investigated".

    achieved = sum over periods of sum u(agent top-k)
    optimal  = sum over periods of sum u(top-k under the TRUE utility order)
    floor    = sum over periods of k * mean(u over every alert in the period)
               (the analytic expectation of a uniformly random order --
                sampling k without replacement from the pool has expected
                total k * mean(u), exactly)

    reward   = clamp((achieved - floor) / (optimal - floor), 0.0, 1.0)

Hard gate: the ordering must be a permutation of exactly the ids supplied --
no duplicates, nothing missing, nothing unknown.  A violation zeroes the
offending period and zeroes the overall reward.
"""

from __future__ import annotations

T_MAX = 30.0
COST = 0.1
K_BUDGET = 20


def decay(age_days: float) -> float:
    """Time-decay multiplier, clamped at zero for alerts past T_MAX days."""
    return max(0.0, 1.0 - float(age_days) / T_MAX)


def utility(alert: dict) -> float:
    """True utility of investigating one alert.  Needs the hidden labels."""
    if alert["is_incident"]:
        return float(alert["severity"]) * decay(alert["age_days"])
    return -COST


def group_by_period(alerts: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for a in alerts:
        out.setdefault(int(a["period"]), []).append(a)
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def check_permutation(order: list, ids: list[int]) -> str | None:
    """Return None if *order* is a permutation of *ids*, else the reason."""
    if not isinstance(order, list):
        return f"ordering is {type(order).__name__}, expected list"
    coerced: list[int] = []
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            # Accept an integral float (e.g. numpy tolist()) but nothing else.
            if isinstance(v, float) and float(v).is_integer():
                coerced.append(int(v))
                continue
            return f"ordering contains a non-integer id: {v!r}"
        coerced.append(int(v))
    want = set(ids)
    if len(coerced) != len(ids):
        return f"ordering has {len(coerced)} ids, expected {len(ids)}"
    seen = set()
    for v in coerced:
        if v in seen:
            return f"duplicate id {v} in ordering"
        seen.add(v)
    # Report unknown ids first: when an id has been swapped rather than
    # dropped, "you invented id X" is the more actionable message.
    unknown = seen - want
    if unknown:
        return f"{len(unknown)} unknown id(s) in ordering, e.g. {sorted(unknown)[:5]}"
    missing = want - seen
    if missing:
        return f"{len(missing)} id(s) missing from ordering, e.g. {sorted(missing)[:5]}"
    return None


def score(alerts: list[dict], order: list, k: int = K_BUDGET) -> dict:
    """Score a full ordering against the labelled alert pool.

    Returns the metrics dict written to metrics.json (all values untransformed).
    """
    all_ids = [int(a["id"]) for a in alerts]
    gate = check_permutation(order, all_ids)

    by_period = group_by_period(alerts)
    periods = sorted(by_period)

    if gate is not None:
        per_period = [
            {
                "period": p,
                "n_alerts": len(by_period[p]),
                "achieved": 0.0,
                "optimal": 0.0,
                "floor": 0.0,
                "normalised": 0.0,
                "valid": False,
                "error": gate,
            }
            for p in periods
        ]
        return {
            "achieved": 0.0,
            "optimal": 0.0,
            "floor": 0.0,
            "normalised": 0.0,
            "precision_at_k": 0.0,
            "mean_age_of_caught_incidents": None,
            "incidents_caught": 0,
            "incidents_available": sum(
                1 for a in alerts if a["is_incident"]
            ),
            "k": k,
            "t_max": T_MAX,
            "cost": COST,
            "valid_ordering": False,
            "error": gate,
            "per_period": per_period,
        }

    rank_of = {int(aid): i for i, aid in enumerate(int(v) for v in order)}

    tot_ach = tot_opt = tot_flr = 0.0
    tot_investigated = 0
    tot_caught = 0
    tot_available = 0
    caught_ages: list[float] = []
    per_period = []

    for p in periods:
        pool = by_period[p]
        pool_sorted = sorted(pool, key=lambda a: rank_of[int(a["id"])])
        chosen = pool_sorted[:k]

        utils = [utility(a) for a in pool]
        achieved = sum(utility(a) for a in chosen)
        optimal = sum(sorted(utils, reverse=True)[:k])
        floor = min(k, len(pool)) * (sum(utils) / len(pool))

        denom = optimal - floor
        norm = _clamp((achieved - floor) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0

        caught = [a for a in chosen if a["is_incident"]]
        available = sum(1 for a in pool if a["is_incident"])
        caught_ages.extend(float(a["age_days"]) for a in caught)

        tot_ach += achieved
        tot_opt += optimal
        tot_flr += floor
        tot_investigated += len(chosen)
        tot_caught += len(caught)
        tot_available += available

        per_period.append(
            {
                "period": p,
                "n_alerts": len(pool),
                "achieved": achieved,
                "optimal": optimal,
                "floor": floor,
                "normalised": norm,
                "precision_at_k": len(caught) / len(chosen) if chosen else 0.0,
                "incidents_caught": len(caught),
                "incidents_available": available,
                "valid": True,
            }
        )

    denom = tot_opt - tot_flr
    normalised = _clamp((tot_ach - tot_flr) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0

    return {
        "achieved": tot_ach,
        "optimal": tot_opt,
        "floor": tot_flr,
        "normalised": normalised,
        "precision_at_k": (tot_caught / tot_investigated) if tot_investigated else 0.0,
        "mean_age_of_caught_incidents": (
            sum(caught_ages) / len(caught_ages) if caught_ages else None
        ),
        "incidents_caught": tot_caught,
        "incidents_available": tot_available,
        "k": k,
        "t_max": T_MAX,
        "cost": COST,
        "valid_ordering": True,
        "error": None,
        "per_period": per_period,
    }
