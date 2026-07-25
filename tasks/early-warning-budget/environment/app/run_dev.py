#!/usr/bin/env python3
"""Score your detector on the 6 labelled dev episodes.

    python3 /app/run_dev.py                 # scores /app/solution.py
    python3 /app/run_dev.py other_file.py

This is a convenience harness, not the grader.  It drives your `Detector`
exactly the way the real verifier does -- constructed once per episode, then
`step(x_t)` once per timestep in order -- and applies the same scoring rules.

Differences from the real run, so you are not surprised:
  * it uses the 6 DEV episodes; you are graded on 12 hidden ones that include
    shapes the dev split does not show;
  * `fpr` here is pooled over the dev calm prefixes only;
  * the reward printed against L_MAX is the real published constant, but the
    dev split's own information bound is printed too, since the two differ.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV_EPISODES = HERE / "dev_episodes.json"

FPR_LIMIT = 0.05
L_MAX = 46.7  # published generator constant used by the real verifier


def load_detector(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Detector


def episode_lead(onset, t_obvious, t_alarm):
    if t_obvious is None:
        return None
    if t_alarm is None or t_alarm < onset or t_alarm >= t_obvious:
        return 0
    return t_obvious - t_alarm


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "solution.py"
    detector_cls = load_detector(target)
    episodes = json.loads(DEV_EPISODES.read_text())

    leads: list[int] = []
    calm_steps = 0
    calm_alarms = 0
    n_never = 0
    rows = []

    for ep in episodes:
        detector = detector_cls(ep["n_channels"], [list(g) for g in ep["groups"]])
        t_alarm = None
        alarms_in_calm = 0
        onset = ep["onset"]
        for t, row in enumerate(ep["x"]):
            decision = 1 if int(detector.step(list(row))) == 1 else 0
            if decision:
                if t_alarm is None:
                    t_alarm = t
                if t < onset:
                    alarms_in_calm += 1

        if t_alarm is None:
            n_never += 1
        calm_steps += onset
        calm_alarms += alarms_in_calm

        lead = episode_lead(onset, ep["t_obvious"], t_alarm)
        if lead is not None:
            leads.append(lead)
        rows.append((ep["episode_id"], ep["T"], onset, ep["t_obvious"],
                     t_alarm, lead, ep["t_obvious"] - onset, alarms_in_calm))

    mean_lead = sum(leads) / len(leads) if leads else 0.0
    fpr = calm_alarms / calm_steps if calm_steps else 0.0
    gate_ok = fpr <= FPR_LIMIT
    reward = 0.0 if not gate_ok else min(1.0, max(0.0, mean_lead / L_MAX))
    dev_bound = sum(r[6] for r in rows) / len(rows)

    print(f"detector: {target}")
    print()
    header = (f"{'episode':<10}{'T':>7}{'onset':>7}{'t_obvious':>11}"
              f"{'t_alarm':>9}{'lead':>7}{'max_lead':>10}{'calm_alarms':>13}")
    print(header)
    print("-" * len(header))
    for eid, n, onset, tobv, ta, lead, mx, ca in rows:
        print(f"{eid:<10}{n:>7}{onset:>7}{str(tobv):>11}{str(ta):>9}"
              f"{str(lead):>7}{mx:>10}{ca:>13}")

    print()
    print(f"mean_lead_steps  : {mean_lead:.3f}")
    print(f"fpr              : {fpr:.5f}   (limit {FPR_LIMIT}) "
          f"-> {'PASS' if gate_ok else 'GATED TO 0.0'}")
    print(f"n_never_alarmed  : {n_never}")
    print(f"reward (vs published L_MAX={L_MAX}) : {reward:.4f}")
    print(f"dev-split information bound        : "
          f"mean(t_obvious - onset) = {dev_bound:.2f}")
    print()
    if not gate_ok:
        print("Your false-positive rate is over the gate. The reward is 0.0 "
              "no matter how early you alarm.")
    else:
        print("Reminder: an alarm raised before a break has begun scores a "
              "lead of 0 for that episode, so a scattergun detector loses the "
              "episode even when the pooled fpr stays under the gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
