#!/usr/bin/env python3
"""Self-scoring on the labelled dev series.

    python3 /app/run_dev.py [--solution /app/solution.py] [--json]

Reports exactly what the hidden verifier reports: pooled F1 / precision /
recall, TP / FP / FN, mean absolute localisation offset, per-series predicted
vs true counts, and the null-series behaviour.

The matching and metric code below is a copy of the verifier's `grade.py`, so
the numbers here are computed the same way.  The dev split is NOT a preview of
the hidden split: it is small, it has no change-point-free series in it, and
its counts are not drawn to be representative.  A high dev score is necessary,
not sufficient.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

TOL = 5
DEV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_series.json")


# --------------------------------------------------------------------------
# scoring (identical algorithm to the verifier's grade.py)
# --------------------------------------------------------------------------
def sanitize(pred, length):
    out = set()
    if not isinstance(pred, (list, tuple)):
        return []
    for item in pred:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            value = item
        elif isinstance(item, float) and float(item).is_integer():
            value = int(item)
        else:
            continue
        if 0 <= value < length:
            out.add(value)
    return sorted(out)


def match_one_to_one(preds, trues, tol=TOL):
    """Nearest-first greedy, one-to-one: ties broken by (distance, true, pred)."""
    candidates = []
    for pi, p in enumerate(preds):
        for ti, t in enumerate(trues):
            d = abs(p - t)
            if d <= tol:
                candidates.append((d, t, p, pi, ti))
    candidates.sort()
    used_p, used_t, pairs = set(), set(), []
    for d, t, p, pi, ti in candidates:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        pairs.append((p, t, d))
    left_p = [p for pi, p in enumerate(preds) if pi not in used_p]
    left_t = [t for ti, t in enumerate(trues) if ti not in used_t]
    return pairs, left_p, left_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default="/app/solution.py")
    ap.add_argument("--data", default=DEV_PATH)
    ap.add_argument("--json", action="store_true", help="dump the metrics dict")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("candidate", args.solution)
    if spec is None or spec.loader is None:
        print(f"cannot load {args.solution}")
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    find_changepoints = getattr(module, "find_changepoints", None)
    if not callable(find_changepoints):
        print(f"{args.solution} does not define find_changepoints(series)")
        return 2

    with open(args.data) as fh:
        data = json.load(fh)
    tol = int(data.get("tolerance", TOL))

    tp = fp = fn = 0
    offsets = []
    rows = []
    null_predictions = 0
    n_null = 0
    started = time.time()

    for item in data["series"]:
        trues = sorted(int(c) for c in item["change_points"])
        t0 = time.time()
        try:
            raw = find_changepoints([list(row) for row in item["values"]])
        except Exception as exc:  # noqa: BLE001 - report, do not crash the harness
            print(f"{item['id']}: raised {type(exc).__name__}: {exc}")
            raw = []
        elapsed = time.time() - t0
        preds = sanitize(list(raw or []), int(item["length"]))

        pairs, left_p, left_t = match_one_to_one(preds, trues, tol)
        tp += len(pairs)
        fp += len(left_p)
        fn += len(left_t)
        offsets.extend(d for _, _, d in pairs)
        if not trues:
            n_null += 1
            null_predictions += len(preds)
        rows.append(
            {
                "id": item["id"],
                "dim": item["dim"],
                "length": item["length"],
                "n_pred": len(preds),
                "n_true": len(trues),
                "tp": len(pairs),
                "fp": len(left_p),
                "fn": len(left_t),
                "seconds": elapsed,
                "pred": preds,
                "true": trues,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if tp else 0.0
    mean_abs_offset = (sum(offsets) / len(offsets)) if offsets else 0.0
    null_fp_rate = (null_predictions / n_null) if n_null else 0.0

    print(f"solution : {args.solution}")
    print(f"series   : {len(rows)}  (tolerance +/-{tol} samples)")
    print("")
    print(f"{'id':<9}{'dim':>4}{'len':>6}{'#pred':>7}{'#true':>7}{'tp':>4}{'fp':>4}{'fn':>4}{'sec':>7}")
    for r in rows:
        print(
            f"{r['id']:<9}{r['dim']:>4}{r['length']:>6}{r['n_pred']:>7}{r['n_true']:>7}"
            f"{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}{r['seconds']:>7.2f}"
        )
    print("")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    print(f"mean_abs_offset={mean_abs_offset:.3f} samples over {len(offsets)} matches")
    print(f"total wall clock: {time.time() - started:.1f}s")
    print("")
    if n_null:
        print(f"null series in dev: {n_null}, null_fp_rate={null_fp_rate:.3f}")
    else:
        print(
            "null series in dev: 0.  The hidden split DOES contain series with "
            "no change points, and predictions on them are gated: "
            "null_fp_rate > 0.5 forces reward = 0."
        )
    print(
        "predicted counts: "
        + ", ".join(f"{r['id']}={r['n_pred']}(true {r['n_true']})" for r in rows)
    )
    if len({r["n_pred"] for r in rows}) == 1:
        print(
            "WARNING: this solution returned the same number of change points for "
            "every dev series. The hidden counts range from 0 to well above 5."
        )

    if args.json:
        print(
            json.dumps(
                {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "mean_abs_offset": mean_abs_offset,
                    "null_fp_rate": null_fp_rate,
                    "per_series": rows,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
