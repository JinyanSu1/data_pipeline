"""Starter solution for `changepoint-restraint` -- deliberately weak.

This is fixed-budget binary segmentation on the mean of the first channel: it
splits at whatever maximises the between-segment mean difference and stops
after exactly N_CHANGEPOINTS = 3 splits, every time, for every series.

That is the failure mode the task is built to punish:

  * on a series with no change points it still returns 3 predictions, which
    is pure false positives and drives `null_fp_rate` straight through the
    hard gate;
  * on a series whose regimes differ in variance, autocorrelation or noise
    shape but not in mean, the criterion is blind and the 3 points it does
    return land in arbitrary places;
  * on a series with 12 change points it can never return more than 3.

Replace it.  Nothing here is sacred except the signature: the verifier imports
this file and calls `find_changepoints(series)` once per series.

Run `python3 /app/run_dev.py` to score whatever is in this file against the
labelled dev series in /app/dev_series.json.
"""

from __future__ import annotations

import numpy as np

N_CHANGEPOINTS = 3   # a fixed budget -- this is the bug, not a tuning knob
MIN_SEG = 20


def _best_mean_split(x, start, end, min_seg):
    """Index in (start, end) maximising the CUSUM of the mean; None if too short."""
    n = end - start
    if n < 2 * min_seg:
        return None
    seg = x[start:end]
    csum = np.concatenate([[0.0], np.cumsum(seg)])
    total = csum[-1]
    ks = np.arange(min_seg, n - min_seg + 1)
    if ks.size == 0:
        return None
    # |mean(left) - mean(right)| weighted by sqrt(k*(n-k)/n)
    left = csum[ks] / ks
    right = (total - csum[ks]) / (n - ks)
    stat = np.abs(left - right) * np.sqrt(ks * (n - ks) / n)
    return start + int(ks[int(np.argmax(stat))])


def find_changepoints(series):
    """Return exactly N_CHANGEPOINTS indices, whatever the data says."""
    X = np.asarray(series, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < 2 * MIN_SEG:
        return []
    x = X[:, 0]
    n = x.shape[0]

    found = []
    for _ in range(N_CHANGEPOINTS):
        bounds = [0, *sorted(found), n]
        best = None
        for i in range(len(bounds) - 1):
            k = _best_mean_split(x, bounds[i], bounds[i + 1], MIN_SEG)
            if k is None:
                continue
            # crude: score the split by the raw mean gap it creates
            left = x[bounds[i] : k].mean()
            right = x[k : bounds[i + 1]].mean()
            score = abs(left - right)
            if best is None or score > best[0]:
                best = (score, k)
        if best is None:
            break
        found.append(best[1])
    return sorted(int(c) for c in found)
