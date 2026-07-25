"""Reward for `changepoint-restraint`.

reward = pooled F1, except that a detector which sprays predictions over the
null series is disqualified outright:

    null_fp_rate = (predictions made on null series) / (number of null series)

    null_fp_rate > 0.5  ->  reward = 0.0

The gate is a hard floor on restraint: more than one spurious change point per
two change-point-free series is categorically fatal rather than merely costly.
`null_fp_rate` is reported in metrics.json whether it passes or fails.

The ceiling of the reward is 1.0, the natural maximum of F1.  No reference
solution's score is used to normalise anything.
"""

from __future__ import annotations

NULL_FP_GATE = 0.5


def compute_reward(metrics):
    """Return (reward, gate_passed) from a metrics dict produced by grade.score."""
    null_fp_rate = float(metrics.get("null_fp_rate", 0.0))
    if null_fp_rate > NULL_FP_GATE:
        return 0.0, False
    f1 = float(metrics.get("f1", 0.0))
    if f1 != f1:  # NaN guard
        return 0.0, True
    return max(0.0, min(1.0, f1)), True
