# Drift vs. anomaly: online detection on a drifting sensor stream

You are given multivariate streams from a simulated industrial process. Two
different things in these streams look "unusual" to a naive detector, and they
mean opposite things:

* **Benign concept drift.** The process slowly moves to a different operating
  point. Over a long window (hundreds of steps) the channel means shift, or
  the channel scales grow/shrink, or the channel space rotates. Sometimes the
  process settles at the new operating point and stays there; sometimes it
  ramps out and comes back. **Nothing is wrong.** Alarming here is the
  expensive mistake, and it is the one this task punishes hardest.

* **True anomalies.** Short injected faults: transient **spikes**, abrupt
  **level steps**, **stuck-at** sensors that freeze, and **inter-channel
  correlation breaks** where a group of channels keeps its usual marginal
  distribution but stops moving with the rest of the process. These are
  genuine faults and must be flagged.

Telling these two apart *online* is the entire task.

## What you must deliver

`/app/solution.py`, defining exactly this interface:

```python
class Detector:
    def __init__(self, n_channels): ...
    def step(self, x_t) -> int:   # 0 = normal, 1 = alarm
```

* `n_channels` is an `int`. It varies from episode to episode (8-16).
* `step` is called once per timestep, **in stream order**, with `x_t` a
  1-D `numpy` array of `n_channels` float64 values (a fresh copy each call;
  you may mutate it). It must return `0` or `1` for that timestep.
* A fresh `Detector` is constructed for every episode. Nothing carries over.
* `/app/solution.py` is the **only** file collected. It must be self-contained:
  it is copied elsewhere and imported by path, so a helper module you leave
  next to it will not exist at grading time.
* The verifier never hands your code the whole array and never tells you the
  stream length in advance, so lookahead is structurally impossible. Anything
  you know at time `t` you had to compute from `x_0 .. x_t`.

Available: Python 3.12 standard library and `numpy`. Nothing else, no network,
CPU only.

The harness seeds `random` and `numpy.random` (both to 0) once, before your
module is imported, so that the same `solution.py` always produces the same
reward -- do not rely on entropy for behaviour.

Keep the run cheap: all 12 hidden episodes together must finish inside a few
minutes (the reference solution takes about one second in total). The whole
prediction phase is capped at 600 seconds; anything after the cap is scored as
silence.

## What is provided

* `/app/solution.py` -- a deliberately weak starter: a global z-score on
  running mean/std. It fires all the way through the drift windows, blows the
  drift gate, and therefore scores **0.0**. Beating it is not a matter of
  nudging its threshold; it requires actually distinguishing a moving
  operating point from a fault.
* `/app/dev_episodes.json` -- 6 fully labelled development episodes:
  `X` (`n_steps` x `n_channels`), `anomaly_spans`, `drift_spans`,
  `anomaly_types`, `drift_types`, plus a `meta` block describing how each
  episode was built.
* `/app/run_dev.py` -- self-scoring on dev with exactly the rules below:
  `python3 /app/run_dev.py` (add a path to score a different file).

The 12 hidden test episodes are drawn from a **wider** distribution than dev.
They contain stream shapes dev does not show you: more channels, more drift
segments, all three drift kinds in one episode, at least one episode with
drift and **zero** anomalies, and episodes where a fault happens **during** a
benign-drift window. Do not tune to the six dev episodes.

## Scoring

Spans are half-open, `[start, end)`. `TOL = 3`.

**Point adjustment is banned.** The standard trick in this literature -- if
any single point inside a ground-truth event is flagged, mark the entire event
as correctly detected point-by-point -- inflates scores enormously and is
*not* used here. An alarm inside an event credits **that one event**, nothing
more.

* **Detected event (TP).** A ground-truth anomaly event `[s, e)` counts as
  detected if you alarm at least once at a timestep `t` with `s <= t < e`, or
  with `s <= t <= s + TOL`. Each ground-truth event is credited **at most
  once**.
* **False-positive event (FP).** A maximal run of consecutive alarms that
  overlaps no ground-truth event credit window. **Runs are counted, not
  points**: one 500-step spurious alarm is one FP, not 500. Conversely,
  spraying alarms every other step turns one mistake into many.
* `event_precision = TP / (TP + FP)`
* `event_recall = TP / n_events`
* `event_f1 = 2 * precision * recall / (precision + recall)`

All counters are pooled across the 12 hidden episodes before the ratios are
formed.

### The drift gate (hard)

```
drift_fpr = (alarmed timesteps inside benign-drift windows)
            / (timesteps inside benign-drift windows)
```

Timesteps that fall inside an anomaly's credit window are excluded from both
numerator and denominator, so correctly catching a fault that happens during a
drift window is never punished.

```
reward = 0.0        if drift_fpr > 0.10
reward = event_f1   otherwise
```

There is no partial credit past the gate: a detector that finds every fault
but chatters through the drift windows scores exactly zero. The measured
`drift_fpr` is reported either way.

`metrics.json` records, untransformed: `event_f1`, `event_precision`,
`event_recall`, `drift_fpr`, `n_events`, `n_detected`, `n_fp_runs`,
`mean_detection_delay`, plus per-episode breakdowns.

The maximum reward is `1.0`, which is the natural maximum of event F1. It is
not normalised against any reference solution's score.

## Hints, if you want them

* The failure mode of the starter is that its notion of "normal" is fixed (or
  nearly so) while the process's is not. Anything with a *slowly adapting*
  notion of normal -- an EWMA level, a Holt level-plus-trend predictor, a
  short-horizon linear forecast -- absorbs a smooth ramp and leaves the fast
  deviations sticking out.
* A fixed alarm threshold on an adaptive statistic still fails when the noise
  level itself drifts. Consider making the threshold adaptive too.
* Correlation breaks are invisible to any per-channel test: the marginals are
  unchanged by construction. Something joint is required.
* Stuck-at faults are invisible to any test that looks for *large* residuals:
  they are anomalously **quiet**.
* Do not let a fault teach your model that it is normal, and do not let a
  genuine regime change lock you into alarming forever.

## Time budget

You have **60 minutes** (3600 seconds) of wall-clock time for this task. There
is no warning before the deadline. When it expires, whatever is at
`/app/solution.py` is collected and graded exactly as it stands, so keep that
file in a working, submittable state at every point while you iterate --
`python3 /app/run_dev.py` after each change is the cheap way to be sure.
