"""Starter solution -- replace this.

It returns the alerts in the order they were handed over, which is the same as
not triaging at all.  It scores ~0.0: the reward is normalised so that "no
better than a random order" is exactly zero.

Contract (see /app/instruction.md):

    rank(alerts) -> list[int]

    alerts: list of dicts with keys
        'id'        int
        'features'  list[float], 12 entries
        'age_days'  float
        'period'    int

    Return every id exactly once, best first.  The verifier restricts your
    ordering to each period and investigates the top k of it.

Your file must be SELF-CONTAINED: at scoring time the container has
/app/solution.py and nothing else from this directory.  If you fit a model on
/app/dev_alerts.json, bake the coefficients into this file.
"""

from __future__ import annotations


def rank(alerts) -> list[int]:
    return [int(a["id"]) for a in alerts]
