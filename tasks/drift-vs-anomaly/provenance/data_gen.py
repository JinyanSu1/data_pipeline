#!/usr/bin/env python3
"""Deterministic generator for the drift-vs-anomaly streaming benchmark.

Every episode is a multivariate "industrial sensor" stream that contains two
kinds of unusual-looking behaviour:

  * benign concept drift -- the operating point of the process moves smoothly
    over a long window (a mean shift, a scale change, or a rotation of the
    channel space).  Nothing is broken.  Alarming here is the expensive
    mistake and is punished by a hard gate in the verifier.

  * true anomalies -- short injected faults (spikes, abrupt level steps,
    stuck-at sensors, inter-channel correlation breaks).  These are the
    events that must be flagged.

Everything that could plausibly be hard-coded is randomised per episode:
channel count, stream length, latent factor count, noise floor, the number /
length / direction / magnitude / kind of the drift segments, and the number /
type / duration / amplitude of the anomalies.

Usage:
    python3 data_gen.py            # regenerate both episode files in-place
    python3 data_gen.py --check    # regenerate to a temp dir and diff hashes

Numpy + stdlib only.  Deterministic given the seeds below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# Global constants of the benchmark.
# --------------------------------------------------------------------------

TOL = 3  # detection tolerance, in steps, after an event's start

DEV_SEED_BASE = 20240517
TEST_SEED_BASE = 918273645

N_DEV = 6
N_TEST = 12

ROUND_DECIMALS = 5

DRIFT_KINDS = ("mean", "scale", "rotation")
ANOM_KINDS = ("spike", "step", "stuck", "corr_break")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """C1-continuous 0->1 ramp on u in [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _ar1(rng: np.random.Generator, n: int, rho: float, size: int) -> np.ndarray:
    """`size` independent unit-variance AR(1) paths of length n."""
    out = np.empty((n, size))
    innov = rng.normal(size=(n, size)) * np.sqrt(max(1e-12, 1.0 - rho * rho))
    out[0] = rng.normal(size=size)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + innov[t]
    return out


def _robust_sd(a: np.ndarray) -> np.ndarray:
    """Column-wise MAD-based standard deviation estimate."""
    med = np.median(a, axis=0)
    mad = np.median(np.abs(a - med), axis=0)
    sd = 1.4826 * mad
    # Fall back to the plain sd where the MAD degenerates.
    plain = a.std(axis=0)
    sd = np.where(sd < 1e-9, plain, sd)
    return np.maximum(sd, 1e-6)


def _non_overlapping(
    rng: np.random.Generator,
    n: int,
    lengths: list[int],
    lo: int,
    hi: int,
    gap: int,
    forbidden: list[tuple[int, int]] | None = None,
    required_inside: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Place `n` intervals of the given lengths inside [lo, hi).

    Intervals are separated by at least `gap` steps, avoid `forbidden`
    ranges (inflated by `gap`), and -- when `required_inside` is given --
    the first interval is forced to start inside one of those ranges.
    """
    placed: list[tuple[int, int]] = []
    forbidden = forbidden or []
    for i in range(n):
        length = lengths[i]
        for _attempt in range(400):
            if i == 0 and required_inside:
                a, b = required_inside[rng.integers(0, len(required_inside))]
                top = min(b - 1, hi - length)
                if top <= a:
                    continue
                start = int(rng.integers(a, top))
            else:
                if hi - length <= lo:
                    break
                start = int(rng.integers(lo, hi - length))
            end = start + length
            ok = True
            for ps, pe in placed:
                if start < pe + gap and ps - gap < end:
                    ok = False
                    break
            if ok:
                for fs, fe in forbidden:
                    if start < fe + gap and fs - gap < end:
                        ok = False
                        break
            if ok:
                placed.append((start, end))
                break
    placed.sort()
    return placed


# --------------------------------------------------------------------------
# Episode generation
# --------------------------------------------------------------------------


def generate_episode(seed: int, spec: dict | None = None) -> dict:
    """Build one fully-labelled episode.

    `spec` optionally forces structural choices (used to guarantee that the
    hidden test set contains shapes the dev set never shows).
    """
    spec = dict(spec or {})
    rng = np.random.default_rng(seed)

    # ---- structure -------------------------------------------------------
    c_lo, c_hi = spec.get("channel_range", (8, 17))
    n_ch = int(spec.get("n_channels", rng.integers(c_lo, c_hi)))
    n_steps = int(spec.get("n_steps", rng.integers(2200, 3201)))
    n_factors = int(rng.integers(2, min(5, n_ch)))
    warmup = int(spec.get("warmup", 260))

    # ---- noise floor / loadings -----------------------------------------
    base_scale = float(rng.uniform(0.5, 3.0))  # overall signal amplitude
    noise_floor = float(rng.uniform(0.25, 1.10))  # idiosyncratic noise level
    loadings = rng.normal(0.0, 1.0, size=(n_ch, n_factors))
    loadings *= rng.uniform(0.5, 1.6, size=(n_ch, 1))
    idio = rng.uniform(0.5, 1.5, size=n_ch) * noise_floor
    rho = rng.uniform(0.85, 0.97, size=n_factors)
    rho_bar = float(rho.mean())

    # latent factors, unit stationary variance each
    z = np.empty((n_steps, n_factors))
    innov = rng.normal(size=(n_steps, n_factors)) * np.sqrt(1.0 - rho**2)
    z[0] = rng.normal(size=n_factors)
    for t in range(1, n_steps):
        z[t] = rho * z[t - 1] + innov[t]

    eps = rng.normal(size=(n_steps, n_ch)) * idio
    y = z @ loadings.T + eps  # correlated, marginal var = ||w_c||^2 + idio_c^2

    mu0 = rng.normal(0.0, 2.0, size=n_ch) * base_scale

    # ---- drift segments --------------------------------------------------
    n_drift = int(spec.get("n_drift", rng.integers(1, 3)))
    drift_lengths = [int(rng.integers(180, 520)) for _ in range(n_drift)]
    drift_slots = _non_overlapping(
        rng,
        n_drift,
        drift_lengths,
        lo=warmup,
        hi=max(warmup + 1, n_steps - 260),
        gap=110,
    )

    allowed_kinds = list(spec.get("drift_kinds_pool", DRIFT_KINDS))
    forced_kinds = spec.get("drift_kinds")

    mu_t = np.tile(mu0, (n_steps, 1))
    scale_t = np.ones((n_steps, n_ch))
    rotations: list[tuple[int, int, np.ndarray]] = []  # (a, b, angle_per_t)
    drift_records = []

    for i, (ds, de) in enumerate(drift_slots):
        if forced_kinds is not None:
            kind = forced_kinds[i % len(forced_kinds)]
        else:
            kind = str(allowed_kinds[rng.integers(0, len(allowed_kinds))])
        # Two benign shapes, both smooth: a persistent move to a new
        # operating point, or a long excursion that ramps out and back.  The
        # excursion is the case that destroys any detector using a single
        # whole-stream yardstick, while remaining perfectly benign.
        profile = spec.get("drift_profile")
        if profile is None:
            profile = "excursion" if rng.random() < 0.6 else "persistent"
        n_win = de - ds
        ramp = np.zeros(n_steps)
        u = (np.arange(n_win) + 1.0) / n_win
        if profile == "excursion":
            up = _smoothstep(u / 0.42)
            down = 1.0 - _smoothstep((u - 0.58) / 0.42)
            ramp[ds:de] = np.minimum(up, down)
        else:
            ramp[ds:de] = _smoothstep(u)
            ramp[de:] = 1.0  # the new operating point persists

        detail: dict = {"kind": kind, "profile": profile}
        if kind == "mean":
            # A genuine operating-point change moves the whole process, not a
            # handful of channels: every channel shifts, by its own amount.
            chans = np.arange(n_ch)
            marg_sd = base_scale * np.sqrt((loadings**2).sum(1) + idio**2)
            lo_m, hi_m = (5.0, 12.0) if profile == "excursion" else (2.5, 7.0)
            mag = rng.uniform(lo_m, hi_m, size=n_ch) * rng.choice(
                [-1.0, 1.0], size=n_ch
            )
            delta = mag * marg_sd
            mu_t += ramp[:, None] * delta[None, :]
            detail["channels"] = [int(c) for c in chans]
            detail["magnitude_sd"] = [float(abs(m)) for m in mag]
        elif kind == "scale":
            chans = np.arange(n_ch)
            lo_f, hi_f = (0.22, 4.5) if profile == "excursion" else (0.33, 3.0)
            f = np.exp(rng.uniform(np.log(lo_f), np.log(hi_f), size=n_ch))
            scale_t *= np.exp(ramp[:, None] * np.log(f)[None, :])
            detail["channels"] = [int(c) for c in chans]
            detail["factors"] = [float(v) for v in f]
        else:  # rotation
            n_pairs = int(rng.integers(1, min(4, n_ch // 2) + 1))
            perm = rng.permutation(n_ch)
            angles = []
            for p in range(n_pairs):
                a, b = int(perm[2 * p]), int(perm[2 * p + 1])
                theta = float(rng.uniform(0.35, 1.25) * rng.choice([-1.0, 1.0]))
                rotations.append((a, b, ramp * theta))
                angles.append({"pair": [a, b], "theta": theta})
            detail["pairs"] = angles

        # The benign-drift window we hold the detector to: the ramp plus a
        # short settling tail at the new operating point.  An excursion has
        # already returned home by the end of its window, so it gets no tail.
        tail = 0 if profile == "excursion" else int(rng.integers(40, 130))
        drift_records.append(
            {
                "span": [int(ds), int(min(n_steps, de + tail))],
                "ramp": [int(ds), int(de)],
                "detail": detail,
            }
        )

    # ---- assemble the clean stream --------------------------------------
    v = scale_t * y
    for a, b, ang in rotations:
        ca, sa = np.cos(ang), np.sin(ang)
        va, vb = v[:, a].copy(), v[:, b].copy()
        v[:, a] = ca * va - sa * vb
        v[:, b] = sa * va + ca * vb
    x = mu_t + base_scale * v

    # An uncorrelated surrogate with matched marginals and matched temporal
    # smoothness -- used to inject correlation-break faults that are
    # invisible to any per-channel marginal test.
    u = _ar1(rng, n_steps, rho_bar, n_ch)

    def local_sd(t0: int) -> np.ndarray:
        lo = max(0, t0 - 160)
        hi = max(lo + 20, t0)
        return _robust_sd(x[lo:hi])

    # ---- anomalies -------------------------------------------------------
    if "n_anomalies" in spec:
        n_anom = int(spec["n_anomalies"])
    else:
        n_anom = int(rng.integers(2, 6))

    kinds_pool = list(spec.get("anom_kinds_pool", ANOM_KINDS))
    kinds = [str(kinds_pool[rng.integers(0, len(kinds_pool))]) for _ in range(n_anom)]
    forced_kinds_a = spec.get("anom_kinds")
    if forced_kinds_a is not None:
        for i, k in enumerate(forced_kinds_a[:n_anom]):
            kinds[i] = k

    durations = []
    for k in kinds:
        if k == "spike":
            durations.append(int(rng.integers(1, 5)))
        elif k == "step":
            durations.append(int(rng.integers(6, 22)))
        elif k == "stuck":
            durations.append(int(rng.integers(12, 41)))
        else:  # corr_break
            durations.append(int(rng.integers(10, 31)))

    drift_spans = [tuple(d["span"]) for d in drift_records]
    inside_drift = bool(spec.get("anomaly_inside_drift", False))
    avoid_drift = bool(spec.get("anomalies_avoid_drift", False))
    anom_slots = _non_overlapping(
        rng,
        n_anom,
        durations,
        lo=warmup + 40,
        hi=max(warmup + 60, n_steps - 40),
        gap=25,
        forbidden=[(s, e) for s, e in drift_spans] if avoid_drift else None,
        required_inside=drift_spans if (inside_drift and drift_spans) else None,
    )

    anomaly_records = []
    for (as_, ae), kind in zip(anom_slots, kinds):
        sd = local_sd(as_)
        detail = {"kind": kind}
        if kind == "spike":
            k_aff = int(rng.integers(1, min(3, n_ch) + 1))
            chans = rng.permutation(n_ch)[:k_aff]
            amp = rng.uniform(4.0, 7.0, size=k_aff) * rng.choice(
                [-1.0, 1.0], size=k_aff
            )
            for j, c in enumerate(chans):
                x[as_:ae, c] += amp[j] * sd[c]
            detail["channels"] = sorted(int(c) for c in chans)
            detail["amplitude_sd"] = [float(abs(a)) for a in amp]
        elif kind == "step":
            k_aff = int(rng.integers(1, min(4, n_ch) + 1))
            chans = rng.permutation(n_ch)[:k_aff]
            amp = rng.uniform(3.0, 5.5, size=k_aff) * rng.choice(
                [-1.0, 1.0], size=k_aff
            )
            for j, c in enumerate(chans):
                x[as_:ae, c] += amp[j] * sd[c]
            detail["channels"] = sorted(int(c) for c in chans)
            detail["amplitude_sd"] = [float(abs(a)) for a in amp]
        elif kind == "stuck":
            k_aff = int(rng.integers(1, min(3, n_ch) + 1))
            chans = rng.permutation(n_ch)[:k_aff]
            for c in chans:
                held = x[as_ - 1, c]
                quant = rng.normal(0.0, 0.01 * sd[c], size=ae - as_)
                x[as_:ae, c] = held + quant
            detail["channels"] = sorted(int(c) for c in chans)
        else:  # corr_break
            k_aff = int(rng.integers(2, min(6, n_ch) + 1))
            chans = rng.permutation(n_ch)[:k_aff]
            win_sd = local_sd(as_)
            centre = np.median(x[max(0, as_ - 80) : as_], axis=0)
            for c in chans:
                x[as_:ae, c] = centre[c] + win_sd[c] * u[as_:ae, c]
            detail["channels"] = sorted(int(c) for c in chans)
        anomaly_records.append({"span": [int(as_), int(ae)], "detail": detail})

    x = np.round(x, ROUND_DECIMALS)

    return {
        "n_channels": n_ch,
        "n_steps": n_steps,
        "X": x.tolist(),
        "anomaly_spans": [r["span"] for r in anomaly_records],
        "drift_spans": [r["span"] for r in drift_records],
        "anomaly_types": [r["detail"]["kind"] for r in anomaly_records],
        "drift_types": [r["detail"]["kind"] for r in drift_records],
        "meta": {
            "seed": int(seed),
            "n_factors": n_factors,
            "base_scale": round(base_scale, 6),
            "noise_floor": round(noise_floor, 6),
            "warmup": warmup,
            "anomaly_detail": [r["detail"] for r in anomaly_records],
            "drift_detail": [r["detail"] for r in drift_records],
        },
    }


# --------------------------------------------------------------------------
# Dev / test suites
# --------------------------------------------------------------------------

# Dev is deliberately the *narrow* distribution: 8-13 channels, 1-2 drift
# segments, at least two anomalies, and anomalies never overlap a drift
# window.  The hidden test set widens all of that.
DEV_SPECS: list[dict] = [
    {
        "channel_range": (8, 12),
        "drift_kinds": ["mean"],
        "n_drift": 1,
        "anom_kinds": ["spike", "spike", "step"],
        "n_anomalies": 3,
        "anomalies_avoid_drift": True,
    },
    {
        "channel_range": (9, 13),
        "drift_kinds": ["scale"],
        "n_drift": 1,
        "anom_kinds": ["stuck", "spike"],
        "n_anomalies": 3,
        "anomalies_avoid_drift": True,
    },
    {
        "channel_range": (8, 13),
        "drift_kinds": ["rotation"],
        "n_drift": 1,
        "anom_kinds": ["corr_break", "spike"],
        "n_anomalies": 3,
        "anomalies_avoid_drift": True,
    },
    {
        "channel_range": (10, 13),
        "drift_kinds": ["mean", "scale"],
        "n_drift": 2,
        "anom_kinds": ["step", "stuck", "spike"],
        "n_anomalies": 4,
        "anomalies_avoid_drift": True,
    },
    {
        "channel_range": (8, 12),
        "drift_kinds": ["scale", "rotation"],
        "n_drift": 2,
        "anom_kinds": ["corr_break", "stuck", "step"],
        "n_anomalies": 4,
        "anomalies_avoid_drift": True,
    },
    {
        "channel_range": (9, 13),
        "drift_kinds": ["mean"],
        "n_drift": 2,
        "anom_kinds": ["spike", "corr_break", "step", "stuck"],
        "n_anomalies": 5,
        "anomalies_avoid_drift": True,
    },
]

# Test widens the distribution and adds shapes dev never contains:
#   * an episode with drift and ZERO anomalies (index 3)
#   * episodes where an anomaly fires *inside* a benign-drift window (5, 9)
#   * three drift segments covering all three drift kinds (7, 9)
#   * the maximum channel count (11)
TEST_SPECS: list[dict] = [
    {"channel_range": (8, 12), "drift_kinds": ["mean"], "n_drift": 1,
     "anom_kinds": ["spike", "step"], "n_anomalies": 3,
     "anomalies_avoid_drift": True},
    {"channel_range": (10, 15), "drift_kinds": ["scale"], "n_drift": 2,
     "anom_kinds": ["stuck", "corr_break", "spike"], "n_anomalies": 4,
     "anomalies_avoid_drift": True},
    {"channel_range": (12, 16), "drift_kinds": ["rotation"], "n_drift": 2,
     "anom_kinds": ["corr_break", "step"], "n_anomalies": 4,
     "anomalies_avoid_drift": True},
    # ZERO anomalies, drift present -- a detector that always finds something
    # scores nothing here and picks up false-positive runs.
    {"channel_range": (9, 14), "drift_kinds": ["mean", "rotation"], "n_drift": 2,
     "n_anomalies": 0},
    {"channel_range": (8, 11), "drift_kinds": ["scale"], "n_drift": 1,
     "anom_kinds": ["stuck", "stuck", "spike"], "n_anomalies": 3,
     "anomalies_avoid_drift": True},
    # anomaly *inside* a drift window
    {"channel_range": (11, 16), "drift_kinds": ["mean"], "n_drift": 2,
     "anom_kinds": ["spike", "step", "corr_break"], "n_anomalies": 4,
     "anomaly_inside_drift": True},
    {"channel_range": (8, 13), "drift_kinds": ["rotation", "scale"], "n_drift": 2,
     "anom_kinds": ["corr_break", "corr_break", "spike"], "n_anomalies": 4,
     "anomalies_avoid_drift": True},
    # all three drift kinds at once
    {"channel_range": (13, 17), "drift_kinds": ["mean", "scale", "rotation"],
     "n_drift": 3, "anom_kinds": ["spike", "stuck", "step"], "n_anomalies": 5,
     "anomalies_avoid_drift": True},
    {"channel_range": (8, 12), "drift_kinds": ["mean", "scale"], "n_drift": 2,
     "anom_kinds": ["step", "spike"], "n_anomalies": 6,
     "anomalies_avoid_drift": True},
    # anomaly inside a drift window *and* three drift kinds
    {"channel_range": (12, 17), "drift_kinds": ["scale", "rotation", "mean"],
     "n_drift": 3, "anom_kinds": ["step", "corr_break", "stuck"],
     "n_anomalies": 4, "anomaly_inside_drift": True},
    {"channel_range": (9, 14), "drift_kinds": ["rotation"], "n_drift": 1,
     "anom_kinds": ["stuck", "corr_break"], "n_anomalies": 3,
     "anomalies_avoid_drift": True},
    # widest stream in the suite
    {"n_channels": 16, "drift_kinds": ["mean", "rotation"], "n_drift": 2,
     "anom_kinds": ["spike", "step", "stuck", "corr_break"], "n_anomalies": 7,
     "anomalies_avoid_drift": True},
]


def build_suite(prefix: str, seed_base: int, specs: list[dict]) -> dict:
    episodes = []
    for i, spec in enumerate(specs):
        ep = generate_episode(seed_base + 1000 * i, spec)
        ep["episode_id"] = f"{prefix}_{i:02d}"
        episodes.append(ep)
    return {
        "tol": TOL,
        "drift_fpr_limit": 0.10,
        "episodes": episodes,
    }


def _write(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    with open(path, "w") as fh:
        fh.write(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help="root of the task directory")
    ap.add_argument("--check", action="store_true", help="print hashes only")
    args = ap.parse_args()

    root = args.outdir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev = build_suite("dev", DEV_SEED_BASE, DEV_SPECS)
    test = build_suite("test", TEST_SEED_BASE, TEST_SPECS)

    if args.check:
        for name, payload in (("dev", dev), ("test", test)):
            text = json.dumps(payload, separators=(",", ":"))
            print(name, hashlib.sha256(text.encode()).hexdigest())
        return 0

    h1 = _write(os.path.join(root, "environment", "app", "dev_episodes.json"), dev)
    h2 = _write(os.path.join(root, "tests", "test_episodes.json"), test)
    print(f"dev  sha256={h1}  episodes={len(dev['episodes'])}")
    print(f"test sha256={h2}  episodes={len(test['episodes'])}")
    for tag, payload in (("dev", dev), ("test", test)):
        for ep in payload["episodes"]:
            print(
                f"  {ep['episode_id']}: T={ep['n_steps']} C={ep['n_channels']} "
                f"drift={ep['drift_types']} anoms={ep['anomaly_types']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
