"""Starter detector.  THIS IS THE FILE YOU EDIT.

It currently implements the most obvious possible rule: watch the whole-system
mean pairwise correlation over a trailing 20-step window and alarm the moment
it goes above 0.7.  That is *exactly* the published definition of `t_obvious`,
so this detector alarms at `t_obvious` on every episode and earns a lead of 0
-- a reward of 0.0.  It never cries wolf, so it does pass the false-positive
gate; it is simply useless as an early warning.

Your job is to alarm much earlier, during the build-up, while keeping the
false-positive rate over the (long) calm prefixes under 5%.

Contract, enforced by the harness:
  * `Detector(n_channels, groups)` is constructed once per episode.
  * `step(x_t)` is called once per timestep, in order, with a plain list of
    `n_channels` floats, and must return 0 or 1.
  * You never receive the full series.  There is no way to look ahead.

Only the Python standard library and numpy are available.  No network, no GPU.
"""

import numpy as np

WINDOW = 20
THRESHOLD = 0.7


class Detector:
    def __init__(self, n_channels, groups):
        self.n_channels = int(n_channels)
        self.groups = [list(g) for g in groups]
        self.window = []
        self._triu = np.triu_indices(self.n_channels, 1)

    def step(self, x_t):
        self.window.append(list(x_t))
        if len(self.window) > WINDOW:
            self.window.pop(0)
        if len(self.window) < WINDOW:
            return 0

        block = np.asarray(self.window, dtype=float)
        corr = np.corrcoef(block, rowvar=False)
        vals = corr[self._triu]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0
        return 1 if float(vals.mean()) > THRESHOLD else 0
