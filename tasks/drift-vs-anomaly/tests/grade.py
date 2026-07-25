"""Event-wise scoring for the drift-vs-anomaly benchmark.

Deliberately plain stdlib Python: the verifier's root process imports this and
nothing else, so it stays small enough to audit by eye.

Scoring rules (these are the contract; instruction.md states them verbatim):

  * NO POINT ADJUSTMENT.  A ground-truth event is never expanded into "all of
    its points count as detected".  The only thing an alarm inside an event
    buys you is one credited event.

  * A ground-truth anomaly event [s, e) is DETECTED if the detector raises at
    least one alarm at a timestep t with  s <= t < e  or  s <= t <= s + TOL.
    Each event can be credited at most once.

  * A FALSE-POSITIVE EVENT is a maximal run of consecutive alarms that
    overlaps no ground-truth credit window.  Runs are counted, not points:
    one 500-step spurious alarm is one false positive, not 500.

  * event_precision = TP / (TP + FP)
    event_recall    = TP / n_events
    event_f1        = harmonic mean of the two

  * drift_fpr = (alarmed timesteps inside benign-drift windows) /
                (timesteps inside benign-drift windows),
    with timesteps that fall inside an anomaly credit window excluded from
    both numerator and denominator, so correctly firing on a fault that
    happens to occur during a drift window is never punished.
"""

from __future__ import annotations

TOL = 3
DRIFT_FPR_LIMIT = 0.10


def credit_window(span, tol=TOL):
    """Inclusive [lo, hi] window in which an alarm credits the event."""
    s, e = int(span[0]), int(span[1])
    return s, max(e - 1, s + tol)


def alarm_runs(pred):
    """Maximal runs of consecutive 1s as inclusive [start, end] pairs."""
    runs = []
    start = None
    for t, v in enumerate(pred):
        if v:
            if start is None:
                start = t
        else:
            if start is not None:
                runs.append((start, t - 1))
                start = None
    if start is not None:
        runs.append((start, len(pred) - 1))
    return runs


def _overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def score_episode(pred, anomaly_spans, drift_spans, tol=TOL):
    """Score one episode.  `pred` is a 0/1 sequence, one entry per timestep."""
    pred = [1 if int(v) else 0 for v in pred]
    n = len(pred)
    windows = [credit_window(s, tol) for s in anomaly_spans]

    # --- true positives: at most one credit per ground-truth event ---------
    delays = []
    tp = 0
    for lo, hi in windows:
        first = None
        for t in range(max(0, lo), min(n - 1, hi) + 1):
            if pred[t]:
                first = t
                break
        if first is not None:
            tp += 1
            delays.append(first - lo)

    # --- false positives: maximal alarm runs matching no event ------------
    runs = alarm_runs(pred)
    fp_runs = 0
    for run in runs:
        if not any(_overlaps(run, w) for w in windows):
            fp_runs += 1

    # --- drift false-positive rate ----------------------------------------
    graced = [False] * n
    for lo, hi in windows:
        for t in range(max(0, lo), min(n - 1, hi) + 1):
            graced[t] = True

    drift_total = 0
    drift_alarms = 0
    for s, e in drift_spans:
        for t in range(max(0, int(s)), min(n, int(e))):
            if graced[t]:
                continue
            drift_total += 1
            drift_alarms += pred[t]

    return {
        "n_events": len(anomaly_spans),
        "n_detected": tp,
        "n_fp_runs": fp_runs,
        "n_alarm_runs": len(runs),
        "n_alarm_points": sum(pred),
        "drift_alarm_steps": drift_alarms,
        "drift_steps": drift_total,
        "delays": delays,
    }


def _f1(precision, recall):
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def aggregate(per_episode):
    """Pool per-episode counters into the headline metrics."""
    tp = sum(e["n_detected"] for e in per_episode)
    fp = sum(e["n_fp_runs"] for e in per_episode)
    n_events = sum(e["n_events"] for e in per_episode)
    drift_alarms = sum(e["drift_alarm_steps"] for e in per_episode)
    drift_steps = sum(e["drift_steps"] for e in per_episode)
    delays = [d for e in per_episode for d in e["delays"]]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_events if n_events > 0 else 0.0
    f1 = _f1(precision, recall)
    drift_fpr = drift_alarms / drift_steps if drift_steps > 0 else 0.0

    return {
        "event_f1": f1,
        "event_precision": precision,
        "event_recall": recall,
        "drift_fpr": drift_fpr,
        "n_events": n_events,
        "n_detected": tp,
        "n_fp_runs": fp,
        "mean_detection_delay": (sum(delays) / len(delays)) if delays else 0.0,
        "drift_alarm_steps": drift_alarms,
        "drift_steps": drift_steps,
    }


def score_all(predictions, episodes, tol=TOL):
    """`predictions` maps episode_id -> 0/1 list.  Returns (summary, details).

    A missing or wrong-length prediction vector is padded/truncated with 0s
    (silence), never with a guess.
    """
    scored = []
    for ep in episodes:
        eid = ep["episode_id"]
        pred = list(predictions.get(eid) or [])
        if len(pred) != ep["n_steps"]:
            pred = (pred + [0] * ep["n_steps"])[: ep["n_steps"]]
        d = score_episode(pred, ep["anomaly_spans"], ep["drift_spans"], tol=tol)
        d["episode_id"] = eid
        scored.append(d)

    summary = aggregate(scored)

    details = []
    for d in scored:
        e = {k: v for k, v in d.items() if k != "delays"}
        e["mean_detection_delay"] = (
            sum(d["delays"]) / len(d["delays"]) if d["delays"] else 0.0
        )
        e["drift_fpr"] = (
            d["drift_alarm_steps"] / d["drift_steps"] if d["drift_steps"] else 0.0
        )
        details.append(e)

    return summary, details
