"""Self-scoring on the labelled dev episodes.

    python3 /app/run_dev.py                 # score /app/solution.py
    python3 /app/run_dev.py other.py        # score another file
    python3 /app/run_dev.py --episodes 0 3  # only these dev episodes

This applies exactly the rules the hidden verifier applies: event-wise F1
with NO point adjustment, plus the hard drift false-positive gate.  The dev
episodes are not the test episodes -- the hidden set is larger and contains
stream shapes that do not appear here -- so treat a good dev score as
necessary, not sufficient.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

DEV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_episodes.json")
TOL = 3
DRIFT_FPR_LIMIT = 0.10


# ---------------------------------------------------------------------------
# Scoring (kept deliberately explicit; identical rules to the verifier)
# ---------------------------------------------------------------------------
def credit_window(span, tol=TOL):
    s, e = int(span[0]), int(span[1])
    return s, max(e - 1, s + tol)


def alarm_runs(pred):
    runs, start = [], None
    for t, v in enumerate(pred):
        if v and start is None:
            start = t
        elif not v and start is not None:
            runs.append((start, t - 1))
            start = None
    if start is not None:
        runs.append((start, len(pred) - 1))
    return runs


def score_episode(pred, anomaly_spans, drift_spans, tol=TOL):
    n = len(pred)
    windows = [credit_window(s, tol) for s in anomaly_spans]

    tp, delays = 0, []
    for lo, hi in windows:
        for t in range(max(0, lo), min(n - 1, hi) + 1):
            if pred[t]:
                tp += 1
                delays.append(t - lo)
                break

    fp = 0
    for run in alarm_runs(pred):
        if not any(run[0] <= w[1] and w[0] <= run[1] for w in windows):
            fp += 1

    graced = [False] * n
    for lo, hi in windows:
        for t in range(max(0, lo), min(n - 1, hi) + 1):
            graced[t] = True
    d_tot = d_alarm = 0
    for s, e in drift_spans:
        for t in range(max(0, int(s)), min(n, int(e))):
            if not graced[t]:
                d_tot += 1
                d_alarm += pred[t]

    return {
        "n_events": len(anomaly_spans),
        "n_detected": tp,
        "n_fp_runs": fp,
        "drift_alarm_steps": d_alarm,
        "drift_steps": d_tot,
        "delays": delays,
        "n_alarm_points": sum(pred),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("solution", nargs="?", default="/app/solution.py")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)

    spec = importlib.util.spec_from_file_location("submitted", args.solution)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Detector = module.Detector

    with open(DEV_PATH) as fh:
        episodes = json.load(fh)["episodes"]
    if args.episodes:
        episodes = [episodes[i] for i in args.episodes]

    totals = {"n_events": 0, "n_detected": 0, "n_fp_runs": 0,
              "drift_alarm_steps": 0, "drift_steps": 0}
    delays = []

    print(f"scoring {args.solution} on {len(episodes)} dev episodes")
    print(f"{'episode':10s} {'C':>3s} {'T':>5s} {'events':>6s} {'det':>4s} "
          f"{'fp_runs':>7s} {'alarms':>7s} {'drift_fpr':>10s}")

    for ep in episodes:
        x = np.asarray(ep["X"], dtype=np.float64)
        det = Detector(ep["n_channels"])
        t0 = time.time()
        pred = [1 if det.step(np.array(x[t], copy=True)) else 0 for t in range(len(x))]
        elapsed = time.time() - t0
        r = score_episode(pred, ep["anomaly_spans"], ep["drift_spans"])
        for k in totals:
            totals[k] += r[k]
        delays.extend(r["delays"])
        fpr = r["drift_alarm_steps"] / r["drift_steps"] if r["drift_steps"] else 0.0
        print(f"{ep['episode_id']:10s} {ep['n_channels']:3d} {ep['n_steps']:5d} "
              f"{r['n_events']:6d} {r['n_detected']:4d} {r['n_fp_runs']:7d} "
              f"{r['n_alarm_points']:7d} {fpr:10.4f}   ({elapsed:.1f}s)")

    tp, fp = totals["n_detected"], totals["n_fp_runs"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / totals["n_events"] if totals["n_events"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    drift_fpr = (
        totals["drift_alarm_steps"] / totals["drift_steps"]
        if totals["drift_steps"]
        else 0.0
    )
    gate_ok = drift_fpr <= DRIFT_FPR_LIMIT
    reward = f1 if gate_ok else 0.0

    print("-" * 66)
    print(f"  event_f1             {f1:.4f}")
    print(f"  event_precision      {prec:.4f}   (TP={tp}, FP runs={fp})")
    print(f"  event_recall         {rec:.4f}   ({tp}/{totals['n_events']} events)")
    print(f"  mean_detection_delay {(sum(delays) / len(delays)) if delays else 0.0:.2f}")
    print(f"  drift_fpr            {drift_fpr:.4f}   "
          f"({totals['drift_alarm_steps']}/{totals['drift_steps']} drift steps)")
    print(f"  drift gate (<= {DRIFT_FPR_LIMIT:.2f})  "
          f"{'PASS' if gate_ok else 'FAIL -- reward is forced to 0.0'}")
    print(f"  reward               {reward:.4f}")
    print("-" * 66)
    print("  Reminder: point adjustment is NOT used.  One alarm inside an event"
          "\n  credits that one event and nothing else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
