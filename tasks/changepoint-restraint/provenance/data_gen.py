#!/usr/bin/env python3
"""Deterministic generator for the `changepoint-restraint` task.

Produces two files:

    environment/app/dev_series.json   8 labelled development series
    tests/test_series.json           20 hidden test series (labels + metadata)

Design notes
------------
Every series is a piecewise-stationary AR(1) process, generated per channel:

    x_t = mu + phi * (x_{t-1} - mu) + sigma * sqrt(1 - phi^2) * e_t

with ``e_t`` drawn from a unit-variance noise distribution (normal, laplace,
student-t(3) or uniform).  Scaling the innovation by ``sqrt(1 - phi^2)`` keeps
the *marginal* standard deviation equal to ``sigma`` regardless of ``phi``, so
an autocorrelation change is a pure autocorrelation change and not a disguised
variance change.  The process state is carried across a boundary (the new
regime starts from the previous sample), so a change point is never marked by
a trivial discontinuity artefact.

At each boundary one of four things changes, on a randomly chosen non-empty
subset of the channels:

    mean   mu   shifts by delta * sigma,  delta ~ U(0.7, 2.6)
    var    sigma is multiplied by r or 1/r,  r ~ U(1.7, 3.2)
    ar     phi jumps to a value at least 0.55 away, marginal variance held
    dist   the innovation distribution switches to one of contrasting shape
           (normal/uniform <-> laplace/t3)
           (mean, variance and autocorrelation all held fixed)

Magnitudes are sampled across the whole range, so some boundaries are blatant
and some are genuinely subtle.

Anti-leakage
------------
The number of change points is sampled from a fixed multiset that is assigned
to series *independently* of series length; the assignment permutation is
re-drawn until the Pearson and Spearman correlations between length and count
are both below 0.15 in absolute value.  Nothing else in the payload
(dimensionality, id, ordering) carries count information, and the payload
handed to the agent's function contains the samples and nothing else.

Index convention: a change point at index ``i`` means sample ``i`` is the first
sample of the new regime (the boundary lies between ``i-1`` and ``i``).

Run:  python3 provenance/data_gen.py [--out-root <task dir>]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

MASTER_SEED = 20260725
MIN_SEG = 35
LEN_LO, LEN_HI = 520, 1000
TOL = 5

KINDS = ("mean", "var", "ar", "dist")
DISTS = ("normal", "laplace", "t3", "uniform")
# Kurtosis-contrasting partners: a "dist" change must cross this partition.
LIGHT_TAILED = ("normal", "uniform")
HEAVY_TAILED = ("laplace", "t3")


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------
def draw_noise(rng: np.random.Generator, dist: str, n: int) -> np.ndarray:
    """Unit-variance, zero-mean innovations of the requested shape."""
    if dist == "normal":
        return rng.standard_normal(n)
    if dist == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)
    if dist == "t3":
        return rng.standard_t(3, n) / np.sqrt(3.0)
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)
    raise ValueError(f"unknown dist {dist!r}")


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def sample_change_points(
    rng: np.random.Generator, length: int, n_cp: int, min_seg: int = MIN_SEG
) -> list[int]:
    """`n_cp` boundaries with every segment at least `min_seg` long."""
    if n_cp == 0:
        return []
    n_seg = n_cp + 1
    slack = length - n_seg * min_seg
    if slack < 0:
        raise ValueError(f"cannot fit {n_cp} change points in {length} samples")
    weights = rng.dirichlet(np.ones(n_seg))
    extra = np.floor(weights * slack).astype(int)
    # hand the rounding remainder to random segments so lengths stay integral
    remainder = int(slack - extra.sum())
    for idx in rng.choice(n_seg, size=remainder, replace=True):
        extra[idx] += 1
    seg_lens = extra + min_seg
    return [int(v) for v in np.cumsum(seg_lens)[:-1]]


def pick_channels(rng: np.random.Generator, dim: int) -> np.ndarray:
    """A non-empty subset of channels affected by one boundary."""
    k = int(rng.integers(1, dim + 1))
    return rng.choice(dim, size=k, replace=False)


def apply_change(
    rng: np.random.Generator,
    kind: str,
    channels: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    phi: np.ndarray,
    dist: list[str],
) -> dict[str, Any]:
    """Mutate regime parameters in place; return a description of the change."""
    info: dict[str, Any] = {"kind": kind, "channels": [int(c) for c in channels]}
    if kind == "mean":
        delta = float(rng.uniform(0.7, 2.6))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        for c in channels:
            mu[c] += sign * delta * sigma[c]
        info["magnitude"] = delta
    elif kind == "var":
        ratio = float(rng.uniform(1.7, 3.2))
        if rng.random() < 0.5:
            ratio = 1.0 / ratio
        for c in channels:
            sigma[c] *= ratio
        info["magnitude"] = abs(float(np.log(ratio)))
    elif kind == "ar":
        for c in channels:
            for _ in range(50):
                new_phi = float(rng.uniform(-0.35, 0.88))
                if abs(new_phi - phi[c]) >= 0.55:
                    break
            else:  # pragma: no cover - the loop essentially always succeeds
                new_phi = -0.35 if phi[c] > 0.3 else 0.88
            info.setdefault("magnitude", abs(new_phi - phi[c]))
            phi[c] = new_phi
    elif kind == "dist":
        for c in channels:
            pool = HEAVY_TAILED if dist[c] in LIGHT_TAILED else LIGHT_TAILED
            dist[c] = str(rng.choice(pool))
        info["magnitude"] = 1.0
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return info


def make_series(
    rng: np.random.Generator,
    length: int,
    dim: int,
    n_cp: int,
    kind_weights: dict[str, float],
    forced_kind: str | None = None,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    cps = sample_change_points(rng, length, n_cp)
    bounds = [0, *cps, length]

    mu = rng.normal(0.0, 1.0, dim)
    sigma = np.exp(rng.uniform(np.log(0.6), np.log(1.8), dim))
    phi = rng.uniform(-0.2, 0.6, dim)
    dist = [str(rng.choice(DISTS, p=[0.55, 0.2, 0.15, 0.10])) for _ in range(dim)]

    kinds = list(kind_weights)
    probs = np.array([kind_weights[k] for k in kinds], dtype=float)
    probs = probs / probs.sum()

    values = np.zeros((length, dim))
    prev = mu + sigma * rng.standard_normal(dim)  # stationary start
    changes: list[dict[str, Any]] = []

    for seg_idx in range(len(bounds) - 1):
        if seg_idx > 0:
            kind = forced_kind if forced_kind else str(rng.choice(kinds, p=probs))
            info = apply_change(rng, kind, pick_channels(rng, dim), mu, sigma, phi, dist)
            info["index"] = int(bounds[seg_idx])
            changes.append(info)
        s, e = bounds[seg_idx], bounds[seg_idx + 1]
        n = e - s
        for c in range(dim):
            inn = sigma[c] * np.sqrt(max(1e-9, 1.0 - phi[c] ** 2)) * draw_noise(rng, dist[c], n)
            x = np.empty(n)
            last = prev[c]
            for t in range(n):
                last = mu[c] + phi[c] * (last - mu[c]) + inn[t]
                x[t] = last
            values[s:e, c] = x
            prev[c] = last
    return values, cps, changes


# --------------------------------------------------------------------------
# assignment of counts to lengths without correlation
# --------------------------------------------------------------------------
def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    am, bm = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - am) * (y - bm) for x, y in zip(a, b))
    da = sum((x - am) ** 2 for x in a) ** 0.5
    db = sum((y - bm) ** 2 for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def decorrelated_assignment(
    rng: np.random.Generator, lengths: list[int], counts: list[int], limit: float = 0.15
) -> list[int]:
    """Permute `counts` over `lengths` until length and count look independent."""
    counts = list(counts)
    for _ in range(20000):
        perm = rng.permutation(len(counts))
        trial = [counts[i] for i in perm]
        fl = [float(v) for v in lengths]
        ft = [float(v) for v in trial]
        if abs(_pearson(fl, ft)) < limit and abs(_pearson(_rank(fl), _rank(ft))) < limit:
            return trial
    raise RuntimeError("failed to find a decorrelated assignment")


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
DEV_WEIGHTS = {"mean": 0.45, "var": 0.25, "ar": 0.20, "dist": 0.10}
TEST_WEIGHTS = {"mean": 0.40, "var": 0.27, "ar": 0.21, "dist": 0.12}


def build_dev() -> list[dict[str, Any]]:
    """8 dev series: 1-5 change points each, never zero, never variance-only."""
    rng = np.random.default_rng([MASTER_SEED, 1])
    counts = [2, 4, 1, 3, 5, 2, 3, 1]
    lengths = [int(rng.integers(LEN_LO, LEN_HI + 1)) for _ in counts]
    counts = decorrelated_assignment(rng, lengths, counts)

    out = []
    for i, (length, n_cp) in enumerate(zip(lengths, counts)):
        srng = np.random.default_rng([MASTER_SEED, 1, i])
        dim = int(srng.integers(1, 4))
        for _ in range(40):
            values, cps, changes = make_series(srng, length, dim, n_cp, DEV_WEIGHTS)
            kinds = {c["kind"] for c in changes}
            if kinds != {"var"}:  # keep variance-only series out of dev
                break
        out.append(
            {
                "id": f"dev_{i:02d}",
                "length": length,
                "dim": dim,
                "values": [[round(float(v), 6) for v in row] for row in values],
                "change_points": [int(c) for c in cps],
                "changes": changes,
            }
        )
    return out


def build_test() -> list[dict[str, Any]]:
    """20 hidden test series, including regimes never seen in dev."""
    rng = np.random.default_rng([MASTER_SEED, 2])
    # 4 nulls, a very long change list, and a broad spread in between.
    counts = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 12, 2, 1, 3]
    lengths = [int(rng.integers(LEN_LO, LEN_HI + 1)) for _ in counts]
    counts = decorrelated_assignment(rng, lengths, counts)

    # Two series are forced to a single change kind: one variance-only (a
    # mean-based detector is blind to it) and one autocorrelation-only.
    non_null = [i for i, c in enumerate(counts) if c >= 2]
    forced_var = int(non_null[int(rng.integers(0, len(non_null)))])
    remaining = [i for i in non_null if i != forced_var]
    forced_ar = int(remaining[int(rng.integers(0, len(remaining)))])

    out = []
    for i, (length, n_cp) in enumerate(zip(lengths, counts)):
        srng = np.random.default_rng([MASTER_SEED, 2, i])
        forced = None
        dim = int(srng.integers(1, 4))
        if i == forced_var:
            forced, dim = "var", 1
        elif i == forced_ar:
            forced, dim = "ar", 1
        values, cps, changes = make_series(
            srng, length, dim, n_cp, TEST_WEIGHTS, forced_kind=forced
        )
        out.append(
            {
                "id": f"test_{i:02d}",
                "length": length,
                "dim": dim,
                "values": [[round(float(v), 6) for v in row] for row in values],
                "change_points": [int(c) for c in cps],
                "changes": changes,
                "is_null": bool(n_cp == 0),
                "forced_kind": forced,
            }
        )
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=default_root)
    args = ap.parse_args()

    dev = build_dev()
    test = build_test()

    dev_path = os.path.join(args.out_root, "environment", "app", "dev_series.json")
    test_path = os.path.join(args.out_root, "tests", "test_series.json")
    os.makedirs(os.path.dirname(dev_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    with open(dev_path, "w") as fh:
        json.dump({"tolerance": TOL, "split": "dev", "series": dev}, fh)
    with open(test_path, "w") as fh:
        json.dump({"tolerance": TOL, "split": "test", "series": test}, fh)

    lengths = [s["length"] for s in test]
    counts = [len(s["change_points"]) for s in test]
    print(f"dev  -> {dev_path}")
    print(f"      counts {[len(s['change_points']) for s in dev]}")
    print(f"      lengths {[s['length'] for s in dev]}")
    print(f"test -> {test_path}")
    print(f"      counts {counts}")
    print(f"      lengths {lengths}")
    print(f"      nulls {sum(1 for s in test if s['is_null'])}")
    print(
        "      pearson(length,count) = "
        f"{_pearson([float(v) for v in lengths], [float(v) for v in counts]):.4f}"
    )
    print(
        "      spearman(length,count) = "
        f"{_pearson(_rank([float(v) for v in lengths]), _rank([float(v) for v in counts])):.4f}"
    )


if __name__ == "__main__":
    main()
