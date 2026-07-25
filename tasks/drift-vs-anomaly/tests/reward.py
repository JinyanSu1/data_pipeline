"""Reward shaping for the drift-vs-anomaly benchmark.

One rule, no curve fitting, no reference-solution normalisation:

    reward = 0.0                if drift_fpr > 0.10
    reward = event_f1           otherwise

The ceiling of 1.0 is the natural maximum of event F1, not a constant lifted
from any particular solution's score.
"""

from __future__ import annotations

from grade import DRIFT_FPR_LIMIT


def compute_reward(summary):
    """Return (reward, explanation) for an aggregate metrics dict."""
    drift_fpr = float(summary["drift_fpr"])
    f1 = float(summary["event_f1"])
    if drift_fpr > DRIFT_FPR_LIMIT:
        return 0.0, (
            f"drift gate FAILED: drift_fpr={drift_fpr:.4f} > {DRIFT_FPR_LIMIT:.2f} "
            f"(event_f1 would have been {f1:.4f})"
        )
    return max(0.0, min(1.0, f1)), (
        f"drift gate passed: drift_fpr={drift_fpr:.4f} <= {DRIFT_FPR_LIMIT:.2f}; "
        f"reward = event_f1 = {f1:.4f}"
    )
