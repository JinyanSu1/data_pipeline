"""Starter detector -- deliberately naive.  Replace this.

It keeps a running mean/std over everything it has seen so far and alarms
whenever any channel's z-score exceeds a fixed threshold.  That is exactly
the "drift-blind" detector the benchmark is built to expose: when the
process drifts to a new operating point the running statistics lag behind,
every channel looks extreme for hundreds of steps in a row, and the drift
false-positive gate zeroes the reward.

Run `python3 /app/run_dev.py` to see it fail on the dev episodes.

The API the verifier uses is fixed:
    Detector(n_channels)  ->  .step(x_t) -> 0 or 1
`step` is called once per timestep, in order.  You never see the future.
"""

import numpy as np


class Detector:
    def __init__(self, n_channels):
        self.n_channels = int(n_channels)
        self.count = 0
        self.sum = np.zeros(self.n_channels, dtype=np.float64)
        self.sumsq = np.zeros(self.n_channels, dtype=np.float64)
        self.threshold = 3.0

    def step(self, x_t):
        x = np.asarray(x_t, dtype=np.float64)
        self.count += 1
        self.sum += x
        self.sumsq += x * x

        if self.count < 50:
            return 0

        mean = self.sum / self.count
        var = self.sumsq / self.count - mean * mean
        sd = np.sqrt(np.maximum(var, 1e-12))
        z = np.abs(x - mean) / sd
        return int(np.any(z > self.threshold))
