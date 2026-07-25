# Early warning on a grouped multivariate stream

You are monitoring 24 sensor channels that arrive one timestep at a time. The
channels are partitioned into 6 groups of 4 (`groups` is given to you).

For a long stretch nothing is happening: the channels are independent noise.
Then, at a time you are not told, a regime break starts. It arrives in three
stages:

1. **Build-up.** The four channels of *one* group start co-moving. Only that
   group. Across the whole system this is nearly invisible — 6 of the 276
   channel pairs move, and the system-wide average correlation barely twitches.
2. **Spread.** Further groups get pulled in, one after another.
3. **Obvious.** A system-wide factor takes over and everything moves together.

By stage 3 anybody can see it. **Your job is to raise the alarm in stage 1**,
as early as you can, without crying wolf during the calm prefix — which can be
as long as 1700 timesteps.

## What you deliver

Edit `/app/solution.py`. It must define:

```python
class Detector:
    def __init__(self, n_channels, groups):
        """n_channels: int. groups: list of lists of channel indices."""

    def step(self, x_t) -> int:
        """x_t: list of n_channels floats. Return 1 to alarm, 0 otherwise."""
```

* A fresh `Detector` is constructed for each episode.
* `step` is called exactly once per timestep, in order, and is handed only that
  timestep's observation. You never receive the array, and there is no way to
  reach a future sample — the observations are streamed into your process one
  row at a time from outside it. Design accordingly.
* Return value must be `0` or `1`. Anything else is read as `0`.
* Python standard library and **numpy** only. No network, no GPU. Keep it fast
  enough to run ~16,000 timesteps in a few minutes.
* The harness seeds `random` and `numpy.random` before your code is imported,
  so runs are reproducible — do not rely on entropy for behaviour.

## How you are scored

### `t_obvious` — the published reference point

`t_obvious` is a fixed, deterministic property of each episode's data. You are
allowed to know exactly how it is computed:

> `t_obvious` is the first timestep `t >= 19` at which the mean of the 276
> off-diagonal entries of the correlation matrix of the trailing 20-step window
> exceeds `0.7`.

```python
import numpy as np

def mean_pairwise_corr(window):          # window: (20, 24) array
    c = np.corrcoef(window, rowvar=False)
    v = c[np.triu_indices(24, 1)]
    return float(v[np.isfinite(v)].mean())

def find_t_obvious(x):                   # x: (T, 24) array
    for t in range(19, len(x)):
        if mean_pairwise_corr(x[t - 19 : t + 1]) > 0.7:
            return t
    return None                          # some episodes never get there
```

It is computed once by the episode generator, on exactly the numbers you are
given, and stored as a label. It is *not* a target to reproduce — a detector
that fires precisely at `t_obvious` scores zero. It is the baseline you have to
beat, and the starter code in `/app/solution.py` is exactly that baseline.

### Per episode

* `t_alarm` — the first timestep on which `step` returned `1`; `None` if it
  never did.
* `lead = t_obvious - t_alarm`, **except** that the lead is `0` when any of
  these hold:
  * the detector never alarmed;
  * it alarmed at or after `t_obvious`;
  * it alarmed **strictly before the break began**. An alarm raised while the
    series is still pure noise has not detected anything. It is a false alarm,
    it costs you the whole episode, and it also counts toward `fpr` below.
* Episodes whose break never reaches the obvious phase have no `t_obvious`.
  They are excluded from the lead average, but their calm timesteps still count
  toward `fpr`.

### Across episodes

* `mean_lead_steps` — mean of `lead` over the episodes that have a `t_obvious`.
* `fpr` — the fraction of *calm-prefix* timesteps (strictly before the break
  begins) on which you alarmed, pooled over all episodes.

```
if fpr > 0.05:
    reward = 0.0                                   # hard gate, no partial credit
else:
    reward = clamp(mean_lead_steps / L_MAX, 0.0, 1.0)
```

`L_MAX = 46.7`. It is a generator constant, not a score anybody achieved: the
mean over the scored test episodes of `t_obvious - onset`, i.e. the gap between
the first timestep on which any build-up signal exists at all and the timestep
at which the break becomes obvious. A reward of 1.0 would require detecting a
correlation change from its very first sample, with zero estimation delay,
which is not possible. Treat it as the ceiling of the ruler, not as a target.

Note the two pressures pull against each other. The gate gives you a 5% budget
of calm-prefix alarms, which sounds generous — but a single premature alarm
zeroes that episode's lead, so in practice you want to be *far* below the gate
and only commit when you are confident.

## What you have to work with

* `/app/dev_episodes.json` — 6 fully labelled development episodes (`onset`,
  the phase boundaries, `t_obvious`, and the observations `x`). The hidden test
  set has 12 episodes and deliberately contains shapes the dev split does not
  show, so do not tune to these six.
* `/app/run_dev.py` — scores a detector on the dev episodes and prints
  `mean_lead_steps`, `fpr`, whether the gate passes, and the reward:

  ```
  python3 /app/run_dev.py            # scores /app/solution.py
  ```

Everything varies from episode to episode: when the break starts, which group
starts it, how long the build-up and the spread last, how many groups end up
involved, the noise scale, the magnitude of the co-movement, and the length of
the series. Nothing is constant enough to hard-code.

## Time budget

**You have 60 minutes.** There is no warning before the clock runs out.

When it expires, `/app/solution.py` is collected exactly as it is on disk at
that moment and scored. There is no grace period and no final save step. So
keep the file in a submittable state the whole time you are working: make one
improvement, confirm it still imports and still runs under
`python3 /app/run_dev.py`, and only then start the next one. A half-finished
edit at minute 60 scores 0.
