"""Anti-degenerate audit.

Scores a set of detectors that do no real work against the hidden test
episodes, using exactly the same grading path as the verifier.  Every one of
them must come out at (or very near) zero -- otherwise the benchmark can be
gamed without solving it.

    always      alarm on every timestep
    never       never alarm
    period-k    alarm every k-th timestep, for several k
    global-z    alarm when any channel exceeds a z-threshold computed from
                the WHOLE stream (an oracle a real detector could not have).
                This is the naive drift-blind detector; it is *supposed* to
                blow the drift gate, and showing that is the point.
    unseeded-random
                alarms with probability p using the `random` module without
                seeding it.  The harness seeds `random` and `numpy.random`
                in the prediction child, so this scores identically on every
                run -- that is what pins reproducibility.

Run:  python3 /tests/degenerate.py            (uses /tests/test_episodes.json)
      python3 degenerate.py path/to/episodes.json
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grade import score_all  # noqa: E402
from reward import compute_reward  # noqa: E402


def pred_always(x):
    return [1] * x.shape[0]


def pred_never(x):
    return [0] * x.shape[0]


def pred_period(k):
    def f(x):
        return [1 if (t % k == 0) else 0 for t in range(x.shape[0])]

    return f


def pred_global_z(thresh):
    """Whole-stream z-score -- the naive drift-blind detector."""

    def f(x):
        mu = x.mean(axis=0)
        sd = np.maximum(x.std(axis=0), 1e-9)
        z = np.abs((x - mu) / sd)
        return [int(v) for v in (z.max(axis=1) > thresh).astype(int)]

    return f


def pred_unseeded_random(p):
    """Uses `random` without seeding it -- pinned by the child's seeding."""

    def f(x):
        return [1 if random.random() < p else 0 for _ in range(x.shape[0])]

    return f


DETECTORS = [
    ("always-alarm", pred_always),
    ("never-alarm", pred_never),
    ("period-5", pred_period(5)),
    ("period-10", pred_period(10)),
    ("period-25", pred_period(25)),
    ("period-100", pred_period(100)),
    ("global-z-3.0 (naive drift-blind)", pred_global_z(3.0)),
    ("global-z-4.0 (naive drift-blind)", pred_global_z(4.0)),
    ("global-z-4.5 (oracle-tuned worst case)", pred_global_z(4.5)),
    ("unseeded-random-p0.02", pred_unseeded_random(0.02)),
]


def main(argv):
    path = argv[1] if len(argv) > 1 else "/tests/test_episodes.json"
    with open(path) as fh:
        suite = json.load(fh)
    episodes = suite["episodes"]
    arrays = [(ep["episode_id"], np.asarray(ep["X"], dtype=np.float64)) for ep in episodes]

    # Same seeding the prediction child applies before running agent code.
    random.seed(0)
    np.random.seed(0)

    print(f"degenerate audit on {path}  ({len(episodes)} episodes)")
    print(
        f"{'detector':36s} {'reward':>8s} {'f1':>7s} {'prec':>7s} "
        f"{'rec':>7s} {'drift_fpr':>10s} {'gate':>6s}"
    )
    worst = 0.0
    for name, fn in DETECTORS:
        preds = {eid: fn(x) for eid, x in arrays}
        summary, _ = score_all(preds, episodes)
        reward, _ = compute_reward(summary)
        worst = max(worst, reward)
        print(
            f"{name:36s} {reward:8.4f} {summary['event_f1']:7.4f} "
            f"{summary['event_precision']:7.4f} {summary['event_recall']:7.4f} "
            f"{summary['drift_fpr']:10.4f} "
            f"{'PASS' if summary['drift_fpr'] <= 0.10 else 'FAIL':>6s}"
        )
    print(f"\nhighest degenerate reward: {worst:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
