#!/usr/bin/env python3
"""Anti-degenerate check for `early-warning-budget`.

Runs a set of detectors that do no detection at all through the EXACT scoring
path the verifier uses (`run_verifier.run_online` -> sandboxed `predict.py`
child -> `grade.grade` -> `reward.compute_reward`) and prints their rewards.
All of them must come out at ~0.

    always alarm            -> fpr 1.0                  -> hard gate     -> 0
    never alarm             -> every lead 0                              -> 0
    fixed hard-coded step   -> wrong on ~every episode                   -> ~0
    random p=0.02 per step  -> first alarm lands in the calm prefix       -> ~0
    unseeded random         -> same as above AND provably reproducible

Paths can be overridden for offline use:
    EWB_TESTS_DIR      (default /tests)
    EWB_EPISODES       (default $EWB_TESTS_DIR/test_episodes.json)
    EWB_DEV_EPISODES   (default /app/dev_episodes.json)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(os.environ.get("EWB_TESTS_DIR", "/tests"))
sys.path.insert(0, str(TESTS_DIR))

import run_verifier as rv  # noqa: E402
from grade import grade  # noqa: E402
from reward import FPR_LIMIT, L_MAX, compute_reward  # noqa: E402

EPISODES_PATH = Path(os.environ.get(
    "EWB_EPISODES", str(TESTS_DIR / "test_episodes.json")))
DEV_EPISODES_PATH = Path(os.environ.get(
    "EWB_DEV_EPISODES", "/app/dev_episodes.json"))

# Drive the real child from wherever this checkout lives.
rv.TESTS_DIR = TESTS_DIR
rv.PREDICT_PATH = TESTS_DIR / "predict.py"

# Anything above this counts as "meaningfully above 0" and fails the check.
TOLERANCE = 0.15


# --------------------------------------------------------------------------
# The degenerate strategies, as source code (they are imported by the child
# exactly like a real submission).
# --------------------------------------------------------------------------
ALWAYS = """
class Detector:
    def __init__(self, n_channels, groups):
        pass

    def step(self, x_t):
        return 1
"""

NEVER = """
class Detector:
    def __init__(self, n_channels, groups):
        pass

    def step(self, x_t):
        return 0
"""

FIXED = """
FIRE_AT = {t}


class Detector:
    def __init__(self, n_channels, groups):
        self.t = -1

    def step(self, x_t):
        self.t += 1
        return 1 if self.t == FIRE_AT else 0
"""

RANDOM_SEEDED = """
import random

_rng = random.Random(1234)


class Detector:
    def __init__(self, n_channels, groups):
        pass

    def step(self, x_t):
        return 1 if _rng.random() < 0.02 else 0
"""

# Deliberately uses the *global* random module with no seed of its own.  The
# harness seeds `random` once in the child before importing this, so the reward
# is nevertheless reproducible run to run -- that is what this entry proves.
RANDOM_UNSEEDED = """
import random


class Detector:
    def __init__(self, n_channels, groups):
        pass

    def step(self, x_t):
        return 1 if random.random() < 0.02 else 0
"""


def evaluate(name: str, source: str, episodes: list[dict]) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="ewb_deg_", dir="/tmp"))
    os.chmod(workdir, 0o755)
    try:
        solution = workdir / "agent_solution.py"
        solution.write_text(source)
        os.chmod(solution, 0o444)

        notes: list[str] = []
        payload = rv.label_free_payload(episodes)
        decisions = rv.run_online(payload, solution, workdir, notes)
        result = grade(episodes, decisions)
        reward = compute_reward(result["mean_lead_steps"], result["fpr"])
        return {
            "name": name,
            "reward": reward,
            "mean_lead_steps": result["mean_lead_steps"],
            "fpr": result["fpr"],
            "n_never_alarmed": result["n_never_alarmed"],
            "per_episode_lead": result["per_episode_lead"],
            "gate_passed": result["fpr"] <= FPR_LIMIT,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------
# Analytic ceiling for the hard-coded-timestep family.
# --------------------------------------------------------------------------
def fixed_t_curve(episodes: list[dict]) -> dict[int, float]:
    """reward(t) for "alarm exactly at timestep t", over the whole timeline."""
    scored = [ep for ep in episodes if ep["t_obvious"] is not None]
    if not scored:
        return {}
    horizon = max(ep["T"] for ep in episodes)
    curve = {}
    for t in range(horizon):
        total = 0
        for ep in scored:
            if ep["onset"] <= t < ep["t_obvious"]:
                total += ep["t_obvious"] - t
        curve[t] = compute_reward(total / len(scored), 0.0)
    return curve


def best_fixed_t(episodes: list[dict]) -> tuple[int, float]:
    curve = fixed_t_curve(episodes)
    if not curve:
        return (0, 0.0)
    t = max(curve, key=lambda k: curve[k])
    return t, curve[t]


def main() -> int:
    episodes = json.loads(EPISODES_PATH.read_text())

    # A hard-coded timestep an agent could actually pick: the one that is
    # optimal on the DEV episodes it can see.  Falls back to a plain round
    # number when the dev file is not reachable.
    if DEV_EPISODES_PATH.is_file():
        dev_episodes = json.loads(DEV_EPISODES_PATH.read_text())
        dev_t, dev_t_reward = best_fixed_t(dev_episodes)
        dev_label = f"dev-optimal t={dev_t} (scores {dev_t_reward:.3f} on dev)"
    else:
        dev_t, dev_label = 500, "fallback t=500 (dev episodes not reachable)"

    cases = [
        ("always alarm (return 1)", ALWAYS),
        ("never alarm (return 0)", NEVER),
        (f"fixed timestep: {dev_label}", FIXED.format(t=dev_t)),
        ("fixed timestep: t=300", FIXED.format(t=300)),
        ("fixed timestep: t=600", FIXED.format(t=600)),
        ("fixed timestep: t=900", FIXED.format(t=900)),
        ("random p=0.02/step (seeded)", RANDOM_SEEDED),
        ("random p=0.02/step (UNSEEDED module rng)", RANDOM_UNSEEDED),
    ]

    print(f"episodes : {EPISODES_PATH}  ({len(episodes)} episodes)")
    print(f"L_MAX    : {L_MAX}    FPR gate: {FPR_LIMIT}")
    print()
    header = (f"{'strategy':<48}{'reward':>9}{'mean_lead':>11}"
              f"{'fpr':>10}{'gate':>7}{'never':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for name, source in cases:
        r = evaluate(name, source, episodes)
        results.append(r)
        print(f"{name:<48}{r['reward']:>9.4f}{r['mean_lead_steps']:>11.3f}"
              f"{r['fpr']:>10.5f}{'pass' if r['gate_passed'] else 'GATED':>7}"
              f"{r['n_never_alarmed']:>7d}")

    # Reproducibility: the unseeded strategy must score identically twice.
    repeat = evaluate("repeat", RANDOM_UNSEEDED, episodes)
    unseeded = next(r for r in results
                    if r["name"].startswith("random p=0.02/step (UNSEEDED"))
    reproducible = (repeat["reward"] == unseeded["reward"]
                    and repeat["per_episode_lead"] == unseeded["per_episode_lead"])
    print()
    print(f"unseeded strategy re-run: reward {repeat['reward']:.6f} "
          f"({'identical' if reproducible else 'DIFFERENT -- seeding is broken'})")

    t_star, r_star = best_fixed_t(episodes)
    print()
    print("diagnostic (NOT attainable by an agent -- needs the hidden onsets):")
    print(f"  oracle-best hard-coded timestep on this very test set: "
          f"t={t_star} -> {r_star:.4f}")
    print("  Onsets sit on a grid whose spacing (130) exceeds the largest")
    print("  (t_obvious - onset), so such a t can be inside at most ONE")
    print("  episode's detection window; this number is just")
    print("  max_i(t_obvious_i - onset_i) / (n_scored * L_MAX) = "
          f"{max(ep['t_obvious'] - ep['onset'] for ep in episodes if ep['t_obvious'] is not None)}"
          f" / (10 * {L_MAX}), the arithmetic floor for 'gets exactly one")
    print("  episode perfectly right', which any strategy whatsoever enjoys.")

    worst = max(results, key=lambda r: r["reward"])
    print()
    if worst["reward"] > TOLERANCE:
        print(f"FAIL: '{worst['name']}' scored {worst['reward']:.4f} "
              f"> tolerance {TOLERANCE}")
        return 1
    if not reproducible:
        print("FAIL: scoring is not reproducible for an unseeded solution")
        return 1
    print(f"OK: every degenerate strategy scored <= {worst['reward']:.4f} "
          f"(tolerance {TOLERANCE}); scoring is reproducible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
