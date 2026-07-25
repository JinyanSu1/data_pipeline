"""Reference solution for `changepoint-restraint`.

Competent, not optimal.  It exists to prove the task is solvable and to give a
sanity anchor for the grader; it does NOT define full marks.  Measured on the
hidden test split it reaches F1 ~= 0.65 with null_fp_rate 0.00, so there is a
lot of room above it.

Three of the four change types in this data are invisible to a detector that
watches only the mean, and the number of change points is not supplied, so the
solution has two jobs: make every change type visible, and decide *how many*
(possibly zero) changes there are.

1. Feature bank -- make every change type a mean shift
------------------------------------------------------
Per channel, after a robust standardisation ``z = (x - median) / (1.4826*MAD)``:

    z                     level       -> mean changes
    log(z^2 + eps)        log-energy  -> variance changes (a factor r in sigma
                                         shifts this by 2*log r; unlike z^2 it
                                         has light tails, so the test is not
                                         driven by a handful of outliers)
    z[t-1]*z[t]           lag product -> autocorrelation changes (its mean is
                                         the lag-1 autocorrelation)
    |z|                   mean abs    -> noise-shape changes (a standardised
                                         Laplace/t has E|z| ~ 0.71-0.75 versus
                                         0.798 for a Gaussian)
    log((z[t]-z[t-1])^2)  log-energy  -> autocorrelation changes again, via
                                         of the difference    Var(dz)=2(1-phi)

Each column is centred and scaled globally.

2. Model selection -- decide how many, including zero
-----------------------------------------------------
Binary segmentation driven by a *normalised* CUSUM statistic.  On a candidate
window of length L, for split k and feature f,

    D(k, f) = |S_k(f) - (k/L) * S_L(f)| / sqrt(L)
    T(k, f) = D(k, f) / sigma_lr(f)

where ``sigma_lr`` is the Bartlett long-run standard deviation of the feature
on that window.  The long-run variance is the crucial piece: the data are
AR(1), and under autocorrelation the plain sample variance badly understates
the variability of a partial-sum path, which is exactly how naive detectors
manufacture change points in pure noise.  Under the null, ``max_k T(k, f)``
converges to the supremum of a standard Brownian bridge, whose 1% quantile is
1.628; THRESHOLD = 1.65 adds a little slack for taking a maximum over features
and over recursion depth.  Nothing is fitted to the hidden split.

A window is split only when ``max_{k,f} T > THRESHOLD``; otherwise recursion
stops.  A null series therefore returns ``[]`` -- restraint is not a special
case bolted on, it is what the criterion does when there is no evidence.
Finally each surviving point is re-tested inside the window formed by its two
neighbours and dropped if it no longer clears the threshold.

Standard library + numpy, CPU, offline, deterministic.
"""

from __future__ import annotations

import math

import numpy as np

THRESHOLD = 1.65   # Brownian-bridge sup, 1% quantile is 1.628
MIN_SEG = 25       # shortest segment binary segmentation may create
CLIP = 8.0         # robust clip on standardised values
EPS = 1e-3         # keeps log(z^2) finite near zero


def _feature_matrix(series):
    """n x (5*d) feature matrix; see the module docstring."""
    X = np.asarray(series, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.size == 0:
        return np.zeros((0, 0))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n, d = X.shape

    columns = []
    for j in range(d):
        x = X[:, j]
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * mad
        if scale < 1e-12:
            scale = float(np.std(x)) or 1.0
        z = np.clip((x - med) / scale, -CLIP, CLIP)

        lag = np.zeros(n)
        dz = np.zeros(n)
        if n > 1:
            lag[1:] = z[:-1] * z[1:]
            dz[1:] = z[1:] - z[:-1]

        columns.append(z)
        columns.append(np.log(z * z + EPS))
        columns.append(np.clip(lag, -CLIP, CLIP))
        columns.append(np.abs(z))
        columns.append(np.log(dz * dz + EPS))

    F = np.column_stack(columns)
    F -= F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-9] = 1.0
    F /= sd
    return F


def _long_run_var(y):
    """Bartlett-kernel long-run variance of `y` (mean removed internally)."""
    L = len(y)
    r = y - y.mean()
    bw = max(1, int(round(4.0 * (L / 100.0) ** (2.0 / 9.0))))
    bw = min(bw, L - 1)
    g0 = float(r @ r) / L
    total = g0
    for h in range(1, bw + 1):
        total += 2.0 * (1.0 - h / (bw + 1.0)) * (float(r[h:] @ r[:-h]) / L)
    return max(total, 1e-12)


def _best_split(F, start, end, min_seg):
    """Strongest normalised CUSUM split in [start, end); None if none exists."""
    L = end - start
    if L < 2 * min_seg:
        return None
    window = F[start:end]
    csum = np.vstack([np.zeros(window.shape[1]), np.cumsum(window, axis=0)])
    total = csum[-1]
    ks = np.arange(min_seg, L - min_seg + 1)
    if ks.size == 0:
        return None
    bridge = np.abs(csum[ks] - (ks[:, None] / L) * total) / math.sqrt(L)

    best_stat, best_k = 0.0, None
    for f in range(window.shape[1]):
        sigma = math.sqrt(_long_run_var(window[:, f]))
        column = bridge[:, f] / sigma
        i = int(np.argmax(column))
        if column[i] > best_stat:
            best_stat, best_k = float(column[i]), int(ks[i])
    if best_k is None:
        return None
    return best_stat, start + best_k


def find_changepoints(series):
    """Return the sorted indices at which the generating distribution changes.

    `series` is a list of rows; each row is a list of floats (1-3 channels).
    An index i means sample i is the first sample of the new regime.  The
    number of change points is decided by the threshold, not supplied.
    """
    F = _feature_matrix(series)
    n = F.shape[0]
    if n < 2 * MIN_SEG or F.shape[1] == 0:
        return []

    found = []
    stack = [(0, n)]
    while stack:
        start, end = stack.pop()
        best = _best_split(F, start, end, MIN_SEG)
        if best is None:
            continue
        stat, k = best
        if stat > THRESHOLD:
            found.append(k)
            stack.append((start, k))
            stack.append((k, end))
    found.sort()

    # Prune: re-test every point against the window spanned by its neighbours.
    dropped = True
    while dropped and found:
        dropped = False
        bounds = [0, *found, n]
        for i in range(len(found)):
            best = _best_split(F, bounds[i], bounds[i + 2], MIN_SEG)
            if best is None or best[0] <= THRESHOLD:
                found = found[:i] + found[i + 1 :]
                dropped = True
                break

    return [int(c) for c in found]
