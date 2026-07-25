"""Pure-stdlib scoring for the `early-warning-budget` task.

This module never imports, executes or otherwise touches the agent's code.  It
is handed (a) the ground-truth episode labels and (b) a per-episode vector of
0/1 alarm decisions that came back from the sandboxed child process, and it
turns them into the metrics defined in instruction.md.

Definitions (identical to instruction.md):

  t_alarm    first timestep at which the detector returned 1, else None.
  t_obvious  published reference point: first t at which the whole-system mean
             pairwise correlation over the trailing 20-step window exceeds 0.7.
             Precomputed by the generator and stored with each episode.
  lead       t_obvious - t_alarm, with lead = 0 when
               * the detector never alarmed, or
               * it alarmed at or after t_obvious, or
               * it alarmed strictly before `onset`.
             The last clause is what makes "lead" mean *detection*: an alarm
             raised while the series is still pure noise has not detected
             anything, it is a false alarm, and it is also counted in `fpr`.
             It also guarantees 0 <= lead <= t_obvious - onset, so no episode
             can contribute more than the information bound.
  fpr        alarms on calm-prefix timesteps (t < onset), pooled over *all*
             episodes -- including the episodes that never reach the obvious
             phase and are therefore excluded from the lead average.
"""

from __future__ import annotations


def first_alarm(decisions: list[int]) -> int | None:
    for t, d in enumerate(decisions):
        if d == 1:
            return t
    return None


def episode_lead(onset: int, t_obvious: int | None,
                 t_alarm: int | None) -> int | None:
    """Lead in timesteps, or None when the episode is not lead-scored."""
    if t_obvious is None:
        return None  # break never became obvious -> excluded from mean_lead
    if t_alarm is None:
        return 0
    if t_alarm < onset:
        return 0  # false alarm during the calm prefix, not a detection
    if t_alarm >= t_obvious:
        return 0  # not earlier than obvious
    return t_obvious - t_alarm


def normalise_decisions(raw: object, n_steps: int) -> list[int]:
    """Coerce whatever came back from the child into exactly n_steps 0/1 ints."""
    out = [0] * n_steps
    if not isinstance(raw, (list, tuple)):
        return out
    for t in range(min(n_steps, len(raw))):
        try:
            out[t] = 1 if int(raw[t]) == 1 else 0
        except (TypeError, ValueError):
            out[t] = 0
    return out


def grade(episodes: list[dict], decisions_by_episode: dict[str, object]) -> dict:
    """Score one run.  `episodes` carries the labels, `decisions_by_episode`
    maps episode_id -> raw decision vector produced by the agent's Detector."""
    per_episode = []
    leads: list[int] = []
    calm_steps = 0
    calm_alarms = 0
    n_never = 0

    for ep in episodes:
        eid = ep["episode_id"]
        n_steps = int(ep["T"])
        onset = int(ep["onset"])
        t_obvious = ep["t_obvious"]
        decisions = normalise_decisions(decisions_by_episode.get(eid), n_steps)

        t_alarm = first_alarm(decisions)
        if t_alarm is None:
            n_never += 1

        calm_steps += onset
        calm_alarms += sum(decisions[:onset])

        lead = episode_lead(onset, t_obvious, t_alarm)
        if lead is not None:
            leads.append(lead)

        per_episode.append({
            "episode_id": eid,
            "T": n_steps,
            "onset": onset,
            "t_obvious": t_obvious,
            "t_alarm": t_alarm,
            "lead": lead,
            "max_lead": (None if t_obvious is None else t_obvious - onset),
            "calm_alarms": int(sum(decisions[:onset])),
            "calm_steps": onset,
            "scored_for_lead": t_obvious is not None,
        })

    mean_lead = (sum(leads) / len(leads)) if leads else 0.0
    fpr = (calm_alarms / calm_steps) if calm_steps else 0.0

    return {
        "mean_lead_steps": mean_lead,
        "fpr": fpr,
        "n_never_alarmed": n_never,
        "per_episode_lead": [p["lead"] for p in per_episode],
        "n_scored_episodes": len(leads),
        "calm_steps_total": calm_steps,
        "calm_alarms_total": calm_alarms,
        "per_episode": per_episode,
    }
