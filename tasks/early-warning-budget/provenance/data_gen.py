#!/usr/bin/env python3
"""Deterministic episode generator for the `early-warning-budget` task.

Signal model
------------
`n_channels = 24` observed channels, partitioned into `n_groups = 6` contiguous
groups of 4.  Every episode is a latent-factor model:

    x[t, i] = wg[i] * A_glob(t) * g[t]                  (system-wide factor)
            + w[i]  * A_grp[k(i)](t) * f[k(i), t]       (group factor)
            + sigma * e[t, i]                           (idiosyncratic noise)

with `g`, `f`, `e` iid standard normal and `k(i)` the group of channel `i`.
All the structure lives in the time-varying loadings:

  * calm prefix      `t < onset`              : every loading is 0, the channels
                                                are independent white noise.
  * build-up         `[onset, onset+build)`   : ONE group (`origin_group`) ramps
                                                its group loading 0 -> A_grp.
                                                Only 6 of the 276 channel pairs
                                                are affected, so the whole-system
                                                mean pairwise correlation barely
                                                moves.  This is the earliest
                                                moment at which the break is
                                                observable at all.
  * spread           `[onset+build, spread_end)` : `n_affected-1` further groups
                                                ramp in, staggered.
  * obvious          `[global_start, T)`      : the system-wide factor ramps in
                                                over `glob_len` steps until the
                                                population mean pairwise
                                                correlation reaches `rho_target`
                                                (0.80-0.90), which is what drives
                                                the published `t_obvious` marker.

Some episodes never enter the obvious phase (`reaches_obvious = False`): the
group structure appears and persists but the system-wide factor stays at 0, so
`t_obvious` does not exist for them.

Published reference point (also stated in instruction.md)
---------------------------------------------------------
`t_obvious` = the first timestep `t >= 19` at which the mean of the 276
off-diagonal entries of `numpy.corrcoef(x[t-19 : t+1], rowvar=False)` exceeds
0.7.  It is computed here, on the *rounded* values that are actually shipped in
the JSON files, so the agent sees exactly the numbers the label was derived
from.

L_MAX
-----
`L_MAX` is an information bound, not a score anybody achieved: it is the mean
over the scored (i.e. `t_obvious is not None`) TEST episodes of

    t_obvious - onset

i.e. the number of timesteps between the earliest instant at which any
build-up signal exists in the data and the instant at which the break becomes
obvious under the published definition.  A detector that alarmed on the very
first sample of the build-up -- with zero estimation delay, which is physically
impossible -- would score exactly 1.0.  Run this file to print the value; it is
hard-coded in `tests/reward.py`.

Usage
-----
    python3 provenance/data_gen.py --out-root <task-dir>

Writes `environment/app/dev_episodes.json` (6 episodes) and
`tests/test_episodes.json` (12 episodes) and prints the derived constants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Fixed structural constants (mirrored in instruction.md / the verifier).
# --------------------------------------------------------------------------
N_CHANNELS = 24
GROUP_SIZE = 4
N_GROUPS = N_CHANNELS // GROUP_SIZE
GROUPS = [list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)) for g in range(N_GROUPS)]

OBVIOUS_WINDOW = 20
OBVIOUS_THRESHOLD = 0.7

DECIMALS = 4  # values are rounded before labelling, so labels match the data

MASTER_SEED = 20260724
DEV_SEED = MASTER_SEED + 1
TEST_SEED = MASTER_SEED + 2

_TRIU = np.triu_indices(N_CHANNELS, 1)


# --------------------------------------------------------------------------
# The published statistic.
# --------------------------------------------------------------------------
def mean_pairwise_corr(window: np.ndarray) -> float:
    """Mean of the off-diagonal entries of the channel correlation matrix."""
    c = np.corrcoef(window, rowvar=False)
    v = c[_TRIU]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(v.mean())


def find_t_obvious(x: np.ndarray) -> int | None:
    """First t with a trailing-20 whole-system mean pairwise corr above 0.7."""
    for t in range(OBVIOUS_WINDOW - 1, x.shape[0]):
        if mean_pairwise_corr(x[t - OBVIOUS_WINDOW + 1 : t + 1]) > OBVIOUS_THRESHOLD:
            return t
    return None


# --------------------------------------------------------------------------
# Loading schedules.
# --------------------------------------------------------------------------
def ramp(n_steps: int, start: int, length: int, amplitude: float) -> np.ndarray:
    """0 before `start`, amplitude/length at `start`, amplitude from start+length-1."""
    t = np.arange(n_steps, dtype=float)
    frac = (t - start + 1.0) / max(length, 1)
    return amplitude * np.clip(frac, 0.0, 1.0)


def _population_mean_corr(a_glob: float, wg: np.ndarray, b: np.ndarray,
                          same_group: np.ndarray, sigma: float) -> float:
    """Population mean pairwise correlation at fully-loaded amplitudes."""
    gl = a_glob * wg
    var = gl * gl + b * b + sigma * sigma
    sd = np.sqrt(var)
    cov = np.outer(gl, gl) + np.where(same_group, np.outer(b, b), 0.0)
    corr = cov / np.outer(sd, sd)
    return float(corr[_TRIU].mean())


def solve_global_amplitude(target: float, wg: np.ndarray, b: np.ndarray,
                           same_group: np.ndarray, sigma: float) -> float:
    """Bisect the system-wide amplitude that yields `target` population corr."""
    lo, hi = 0.0, 40.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if _population_mean_corr(mid, wg, b, same_group, sigma) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Episode sampling.
# --------------------------------------------------------------------------
ONSET_GRID_START = 260
ONSET_GRID_SPACING = 130   # > the largest possible (t_obvious - onset), see below
ONSET_GRID_JITTER = 40


def gridded_onsets(rng: np.random.Generator, n: int) -> list[int]:
    """One onset per episode, on a widely spaced jittered grid, then shuffled.

    Spacing (130) comfortably exceeds the largest attainable
    `t_obvious - onset` (~60), so the per-episode intervals
    `[onset, t_obvious)` are pairwise disjoint across the split.  A detector
    that hard-codes a single timestep can therefore be inside at most ONE
    episode's detection window, which caps that strategy at
    `max_i(D_i) / (n_scored * L_MAX)` -- the arithmetic floor for "gets exactly
    one episode right", and nothing more.
    """
    slots = [
        ONSET_GRID_START + i * ONSET_GRID_SPACING
        + int(rng.integers(0, ONSET_GRID_JITTER + 1))
        for i in range(n)
    ]
    rng.shuffle(slots)
    return [int(v) for v in slots]


def sample_spec(rng: np.random.Generator, *, build_lo: int, build_hi: int,
                reaches_obvious: bool, onset: int) -> dict:
    # The calm prefix (length `onset`) and the tail after the break are both
    # randomised, so the series length is not a usable cue either.
    n_steps = onset + int(rng.integers(240, 421))
    build_len = int(rng.integers(build_lo, build_hi + 1))
    spread_len = int(rng.integers(8, 19))
    stagger_ramp = max(3, spread_len // 2)
    glob_len = int(rng.integers(4, 10))
    n_affected = int(rng.integers(3, N_GROUPS + 1))
    sigma = float(rng.uniform(0.7, 1.4))
    a_grp = sigma * float(rng.uniform(1.4, 2.6))
    rho_target = float(rng.uniform(0.80, 0.90))
    origin = int(rng.integers(0, N_GROUPS))
    others = [g for g in range(N_GROUPS) if g != origin]
    rng.shuffle(others)
    spread_groups = others[: n_affected - 1]
    return dict(
        n_steps=n_steps,
        build_len=build_len,
        spread_len=spread_len,
        stagger_ramp=stagger_ramp,
        glob_len=glob_len,
        n_affected=n_affected,
        onset=onset,
        sigma=sigma,
        a_grp=a_grp,
        rho_target=rho_target,
        origin=origin,
        spread_groups=spread_groups,
        reaches_obvious=reaches_obvious,
    )


def build_episode(rng: np.random.Generator, spec: dict, episode_id: str) -> dict:
    n_steps = spec["n_steps"]
    onset = spec["onset"]
    origin = spec["origin"]
    spread_groups = spec["spread_groups"]
    sigma = spec["sigma"]

    # Per-channel loading jitter so groups are not perfectly homogeneous.
    w = rng.uniform(0.80, 1.20, size=N_CHANNELS)
    wg = rng.uniform(0.90, 1.10, size=N_CHANNELS)

    # --- group loading schedules -----------------------------------------
    grp_load = np.zeros((N_GROUPS, n_steps))
    grp_load[origin] = ramp(n_steps, onset, spec["build_len"], spec["a_grp"])

    build_end = onset + spec["build_len"]
    spread_end = build_end
    n_spread = len(spread_groups)
    for j, k in enumerate(spread_groups):
        offset = int(round(j * spec["spread_len"] / max(1, n_spread)))
        start = build_end + offset
        grp_load[k] = ramp(n_steps, start, spec["stagger_ramp"], spec["a_grp"])
        spread_end = max(spread_end, start + spec["stagger_ramp"])

    affected = [origin] + list(spread_groups)

    # --- system-wide loading schedule ------------------------------------
    b = np.zeros(N_CHANNELS)
    for k in affected:
        for i in GROUPS[k]:
            b[i] = w[i] * spec["a_grp"]
    same_group = np.zeros((N_CHANNELS, N_CHANNELS), dtype=bool)
    for grp in GROUPS:
        for i in grp:
            for j in grp:
                same_group[i, j] = True

    global_start = spread_end
    if spec["reaches_obvious"]:
        a_glob = solve_global_amplitude(spec["rho_target"], wg, b, same_group, sigma)
        glob_sched = ramp(n_steps, global_start, spec["glob_len"], a_glob)
    else:
        a_glob = 0.0
        glob_sched = np.zeros(n_steps)
        global_start = None

    # --- realise the series ----------------------------------------------
    g = rng.standard_normal(n_steps)
    f = rng.standard_normal((N_GROUPS, n_steps))
    e = rng.standard_normal((n_steps, N_CHANNELS))

    x = sigma * e
    for k in range(N_GROUPS):
        load = grp_load[k]
        if not np.any(load):
            continue
        contrib = np.outer(load * f[k], np.ones(GROUP_SIZE))
        for col, i in enumerate(GROUPS[k]):
            x[:, i] += w[i] * contrib[:, col]
    x += np.outer(glob_sched * g, wg)

    x = np.round(x, DECIMALS)
    t_obvious = find_t_obvious(x)

    return dict(
        episode_id=episode_id,
        n_channels=N_CHANNELS,
        groups=[list(gr) for gr in GROUPS],
        T=n_steps,
        onset=onset,
        build_end=build_end,
        spread_end=int(spread_end),
        global_start=(int(global_start) if global_start is not None else None),
        origin_group=origin,
        affected_groups=sorted(int(k) for k in affected),
        n_affected=spec["n_affected"],
        build_len=spec["build_len"],
        spread_len=spec["spread_len"],
        glob_len=spec["glob_len"],
        sigma=round(sigma, 6),
        a_group=round(spec["a_grp"], 6),
        a_global=round(float(a_glob), 6),
        rho_target=round(spec["rho_target"], 6),
        reaches_obvious=bool(spec["reaches_obvious"]),
        t_obvious=t_obvious,
        x=x.tolist(),
    )


# --------------------------------------------------------------------------
# Splits.
# --------------------------------------------------------------------------
def make_dev() -> list[dict]:
    """6 'ordinary' episodes.  Every one of them reaches the obvious phase and
    every build-up is long-ish (16-30) -- the test split deliberately contains
    shapes this split never shows."""
    rng = np.random.default_rng(DEV_SEED)
    onsets = gridded_onsets(rng, 6)
    episodes = []
    for i in range(6):
        spec = sample_spec(rng, build_lo=16, build_hi=30, reaches_obvious=True,
                           onset=onsets[i])
        episodes.append(build_episode(rng, spec, f"dev-{i:02d}"))
    return episodes


def make_test() -> list[dict]:
    """12 hidden episodes.  Includes shapes absent from dev:
      * index 3  : a much shorter build-up (6 steps)
      * index 7  : another short build-up (9 steps)
      * index 5  : break never reaches the obvious phase (t_obvious is None)
      * index 10 : ditto
    """
    rng = np.random.default_rng(TEST_SEED)
    onsets = gridded_onsets(rng, 12)
    short_build = {3: 6, 7: 9}
    never_obvious = {5, 10}
    episodes = []
    for i in range(12):
        spec = sample_spec(
            rng,
            build_lo=12,
            build_hi=32,
            reaches_obvious=i not in never_obvious,
            onset=onsets[i],
        )
        if i in short_build:
            spec["build_len"] = short_build[i]
        episodes.append(build_episode(rng, spec, f"test-{i:02d}"))
    return episodes


def dump(path: Path, episodes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(episodes, fh, separators=(",", ":"))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB, {len(episodes)} episodes)")


def summarise(name: str, episodes: list[dict]) -> float:
    print(f"\n--- {name} ---")
    print(f"{'id':<10}{'T':>6}{'onset':>7}{'build':>7}{'spread':>7}"
          f"{'naff':>6}{'t_obv':>7}{'D':>6}")
    ds = []
    for ep in episodes:
        d = "" if ep["t_obvious"] is None else ep["t_obvious"] - ep["onset"]
        if d != "":
            ds.append(d)
        print(f"{ep['episode_id']:<10}{ep['T']:>6}{ep['onset']:>7}{ep['build_len']:>7}"
              f"{ep['spread_len']:>7}{ep['n_affected']:>6}"
              f"{str(ep['t_obvious']):>7}{str(d):>6}")
    lmax = sum(ds) / len(ds) if ds else 0.0
    print(f"scored episodes: {len(ds)}/{len(episodes)}   "
          f"mean(t_obvious - onset) = {lmax:.6f}")
    return lmax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root = Path(args.out_root)

    dev = make_dev()
    test = make_test()

    dump(root / "environment" / "app" / "dev_episodes.json", dev)
    dump(root / "tests" / "test_episodes.json", test)

    summarise("dev", dev)
    lmax = summarise("test", test)

    print("\n============================================================")
    print("L_MAX (hard-code this in tests/reward.py):")
    print(f"    L_MAX = {lmax:.6f}")
    print("  = mean over the 10 scored TEST episodes of (t_obvious - onset)")
    print("  = mean gap between the first timestep at which any build-up")
    print("    signal exists and the timestep at which the break becomes")
    print("    obvious under the published trailing-20 / rho>0.7 rule.")
    print("  Information bound, independent of any detector.")
    print("============================================================")


if __name__ == "__main__":
    main()
