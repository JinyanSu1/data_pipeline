"""Scoring for `changepoint-restraint`.

Pure standard library.  Imported by `run_verifier.py`, which runs as root and
never imports anything the agent wrote.  The identical algorithm is duplicated
inside `environment/app/run_dev.py` so the agent can self-score on dev.

Matching rule (deterministic, one-to-one)
-----------------------------------------
Matching happens *within* a series; a prediction can never be credited against
a true change point of a different series.  For one series:

1. Build every candidate pair ``(p, t)`` with ``|p - t| <= TOL``.
2. Sort the candidates by ``(|p - t|, t, p)`` -- nearest first, ties broken by
   the true index and then the predicted index, so the order is total and
   independent of input ordering.
3. Walk the sorted list and accept a pair only if neither its prediction nor
   its true point has already been accepted in an earlier pair.

Step 3 is what makes the matching one-to-one: one prediction can be credited
for at most one true change point, and one true change point can absorb at
most one prediction.  A cluster of predictions around a single true point
yields one TP and the rest count as FP.

Counts are then pooled across all series:

    TP = accepted pairs          FP = unmatched predictions
    FN = unmatched true points

    precision = TP / (TP + FP)   recall = TP / (TP + FN)
    f1        = 2 * P * R / (P + R)

with the convention that a rate whose denominator is zero is 0.0, except that
f1 is 0.0 whenever TP is 0.
"""

from __future__ import annotations

TOL = 5


def sanitize(pred, length):
    """Coerce an untrusted prediction list into sorted unique in-range ints."""
    out = set()
    if not isinstance(pred, (list, tuple)):
        return []
    for item in pred:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            value = item
        elif isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                continue
            if float(item).is_integer():
                value = int(item)
            else:
                continue
        else:
            continue
        if 0 <= value < length:
            out.add(value)
    return sorted(out)


def match_one_to_one(preds, trues, tol=TOL):
    """Return (matched_pairs, unmatched_preds, unmatched_trues).

    `matched_pairs` is a list of (pred, true, distance) accepted by the
    nearest-first greedy rule documented in the module docstring.
    """
    candidates = []
    for pi, p in enumerate(preds):
        for ti, t in enumerate(trues):
            d = abs(p - t)
            if d <= tol:
                candidates.append((d, t, p, pi, ti))
    candidates.sort()

    used_p = set()
    used_t = set()
    pairs = []
    for d, t, p, pi, ti in candidates:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        pairs.append((p, t, d))

    left_p = [p for pi, p in enumerate(preds) if pi not in used_p]
    left_t = [t for ti, t in enumerate(trues) if ti not in used_t]
    return pairs, left_p, left_t


def score(series_meta, predictions, tol=TOL):
    """Pool per-series matches into the task metrics.

    `series_meta` is a list of dicts with keys id / length / change_points /
    is_null.  `predictions` maps series id -> raw predicted index list.
    """
    tp = fp = fn = 0
    offsets = []
    predicted_counts = {}
    true_counts = {}
    per_series = []
    null_predictions = 0
    n_null = 0

    for meta in series_meta:
        sid = meta["id"]
        length = int(meta["length"])
        trues = sorted(int(v) for v in meta["change_points"])
        preds = sanitize(predictions.get(sid, []), length)

        pairs, left_p, left_t = match_one_to_one(preds, trues, tol)
        tp += len(pairs)
        fp += len(left_p)
        fn += len(left_t)
        offsets.extend(d for _, _, d in pairs)

        predicted_counts[sid] = len(preds)
        true_counts[sid] = len(trues)
        is_null = bool(meta.get("is_null", len(trues) == 0))
        if is_null:
            n_null += 1
            null_predictions += len(preds)
        per_series.append(
            {
                "id": sid,
                "is_null": is_null,
                "n_pred": len(preds),
                "n_true": len(trues),
                "tp": len(pairs),
                "fp": len(left_p),
                "fn": len(left_t),
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if tp else 0.0
    null_fp_rate = (null_predictions / n_null) if n_null else 0.0
    mean_abs_offset = (sum(offsets) / len(offsets)) if offsets else 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "null_fp_rate": null_fp_rate,
        "null_predictions": null_predictions,
        "n_series": len(series_meta),
        "n_null_series": n_null,
        "mean_abs_offset": mean_abs_offset,
        "tolerance": tol,
        "predicted_counts": predicted_counts,
        "true_counts": true_counts,
        "per_series": per_series,
    }


def _demo():
    """`python3 grade.py` -- show that the matching really is one-to-one."""
    cases = [
        (
            "five predictions clustered on one true point",
            [98, 99, 100, 101, 102],
            [100],
        ),
        (
            "one prediction sitting between two true points",
            [100],
            [98, 102],
        ),
        (
            "nearest-first ordering decides who gets credited",
            [10, 15],
            [14, 19],
        ),
        (
            "the same pair set, predictions listed in reverse order",
            [15, 10],
            [19, 14],
        ),
        (
            "realistic spacing: true points are never within 2*TOL of each other",
            [103, 297, 300],
            [100, 300],
        ),
    ]
    print(f"TOL = {TOL}\n")
    for title, preds, trues in cases:
        pairs, left_p, left_t = match_one_to_one(preds, trues)
        print(title)
        print(f"  preds = {preds}   trues = {trues}")
        print(f"  matched  = {[(p, t) for p, t, _ in pairs]}")
        print(f"  TP = {len(pairs)}  FP = {len(left_p)} {left_p}  FN = {len(left_t)} {left_t}")
        assert len({p for p, _, _ in pairs}) == len(pairs), "a prediction matched twice"
        assert len({t for _, t, _ in pairs}) == len(pairs), "a true point matched twice"
        print("  one-to-one: OK\n")


if __name__ == "__main__":
    _demo()
