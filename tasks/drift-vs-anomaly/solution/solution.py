"""Reference detector for the drift-vs-anomaly benchmark.

Idea in one line: build a *self-updating model of "normal right now"* so that
smooth changes in the operating point are absorbed into the model, and alarm
only on deviations that are too fast for the model to have absorbed.

Four pieces:

1.  Holt (double-exponential) smoothing per channel gives a level *and* a
    trend.  A linear ramp is tracked with no asymptotic bias, so a benign
    drift produces near-zero residuals once the trend term catches up --
    this is what a plain EWMA (or a global mean) gets wrong.

2.  A slowly adapting robust scale per channel absorbs benign *scale* drift,
    so residuals stay standardised as the process gets noisier or quieter.

3.  An EWMA correlation matrix of the standardised residuals absorbs benign
    *rotation* drift.  A Mahalanobis distance against it stays flat while the
    channel space slowly rotates, but jumps when the inter-channel
    relationship breaks abruptly.

4.  Every alarm statistic is compared against its own *adaptive* threshold
    (running robust centre + k * running mean-absolute-deviation).  A
    sustained mild elevation -- the signature of drift -- lifts the
    threshold with it; a short violent excursion -- the signature of a fault
    -- does not have time to.

Plus one explicitly non-statistical rule for stuck-at sensors, which are
invisible to residual-magnitude tests precisely because they are too quiet.

Model updates freeze while an alarm is live, so a fault cannot teach the
model that it is normal; the freeze is released after a while so a genuine
regime change is eventually absorbed instead of alarming forever.
"""

import numpy as np


class Detector:
    # --- adaptation rates (all in "per step" units) -----------------------
    ALPHA_LEVEL = 0.030   # Holt level
    BETA_TREND = 0.0025   # Holt trend
    ALPHA_SCALE = 0.010   # robust per-channel scale
    ALPHA_CORR = 0.006    # correlation matrix
    ALPHA_THR = 0.010     # threshold statistics
    ALPHA_DIFF = 0.010    # typical |x_t - x_{t-1}| per channel

    WARMUP = 200          # stay silent while the model calibrates
    WIN = 6               # boxcar length for the "sustained deviation" stats
    RIDGE = 0.08          # correlation-matrix regularisation

    K_MAX = 6.2           # threshold multipliers (tuned on the dev episodes)
    K_MAH = 5.6
    K_WIN = 5.6
    K_WMAH = 5.6

    STUCK_RATIO = 0.06    # |diff| below this fraction of typical => frozen
    STUCK_RUN = 5         # consecutive frozen steps before alarming
    MAX_FREEZE = 60       # give up freezing the model after this many alarms

    def __init__(self, n_channels):
        c = int(n_channels)
        self.c = c
        self.t = 0

        self.level = np.zeros(c)
        self.trend = np.zeros(c)
        self.scale = np.ones(c)
        self.prev_x = None

        self.corr = np.eye(c)
        self.diff_scale = np.ones(c)
        self.stuck_run = np.zeros(c, dtype=np.int64)

        self.buf = np.zeros((self.WIN, c))
        self.buf_n = 0
        self.mah_buf = np.zeros(self.WIN)

        # adaptive thresholds: per-statistic centre and mean absolute deviation
        self.stat_mu = {}
        self.stat_ad = {}

        self.alarm_run = 0

    # ------------------------------------------------------------------
    def _rate(self, alpha):
        """Adaptation rate, accelerated while the estimator is still cold."""
        return max(alpha, 1.0 / max(1.0, float(self.t)))

    def _thr(self, name, value, k, update):
        """Adaptive threshold: robust centre + k * mean absolute deviation."""
        if name not in self.stat_mu:
            self.stat_mu[name] = value
            self.stat_ad[name] = abs(value) * 0.25 + 1e-6
            return float("inf")
        mu = self.stat_mu[name]
        ad = self.stat_ad[name]
        thr = mu + k * max(ad, 1e-9)
        if update:
            a = self._rate(self.ALPHA_THR)
            # clip the update so a single huge excursion cannot move the bar
            v = min(value, mu + 6.0 * max(ad, 1e-9))
            self.stat_mu[name] = (1.0 - a) * mu + a * v
            self.stat_ad[name] = (1.0 - a) * ad + a * abs(v - mu)
        return thr

    # ------------------------------------------------------------------
    def step(self, x_t):
        x = np.asarray(x_t, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.c:
            x = np.resize(x, self.c)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.t += 1
        if self.t == 1:
            self.level = x.copy()
            self.prev_x = x.copy()
            return 0

        # ---- 1. prediction from level + trend ---------------------------
        pred = self.level + self.trend
        resid = x - pred

        # ---- 2. standardise by the slowly adapting robust scale ---------
        r = resid / np.maximum(self.scale, 1e-9)
        r = np.clip(r, -25.0, 25.0)

        # ---- 3. Mahalanobis against the adapting correlation ------------
        reg = self.corr + self.RIDGE * np.eye(self.c)
        try:
            sol = np.linalg.solve(reg, r)
            d2 = float(max(0.0, r @ sol))
        except np.linalg.LinAlgError:
            d2 = float(r @ r)
        mah = np.sqrt(d2 / self.c)

        # ---- 4. sustained-deviation statistics --------------------------
        self.buf[(self.t - 2) % self.WIN] = r
        self.mah_buf[(self.t - 2) % self.WIN] = d2 / self.c
        self.buf_n = min(self.buf_n + 1, self.WIN)
        win_mean = np.abs(self.buf.sum(axis=0) / self.buf_n)
        s_win = float(win_mean.max())
        s_wmah = float(np.sqrt(self.mah_buf.sum() / self.buf_n))

        s_max = float(np.abs(r).max())

        # ---- 5. stuck-at rule -------------------------------------------
        diff = np.abs(x - self.prev_x)
        frozen = diff < self.STUCK_RATIO * np.maximum(self.diff_scale, 1e-12)
        self.stuck_run = np.where(frozen, self.stuck_run + 1, 0)
        stuck_alarm = bool(np.any(self.stuck_run >= self.STUCK_RUN))

        warm = self.t > self.WARMUP
        # thresholds always keep learning; only the *model* freezes on alarm
        thr_max = self._thr("max", s_max, self.K_MAX, True)
        thr_mah = self._thr("mah", mah, self.K_MAH, True)
        thr_win = self._thr("win", s_win, self.K_WIN, True)
        thr_wmah = self._thr("wmah", s_wmah, self.K_WMAH, True)

        alarm = warm and (
            stuck_alarm
            or s_max > thr_max
            or mah > thr_mah
            or s_win > thr_win
            or s_wmah > thr_wmah
        )

        # ---- 6. update the model (frozen while alarming) ----------------
        self.alarm_run = self.alarm_run + 1 if alarm else 0
        update_model = (not alarm) or self.alarm_run > self.MAX_FREEZE

        if update_model:
            a, b = self.ALPHA_LEVEL, self.BETA_TREND
            new_level = (1.0 - a) * (self.level + self.trend) + a * x
            self.trend = (1.0 - b) * self.trend + b * (new_level - self.level)
            self.level = new_level

            ascale = self._rate(self.ALPHA_SCALE)
            self.scale = (1.0 - ascale) * self.scale + ascale * (
                1.2533 * np.abs(resid)
            )
            self.scale = np.maximum(self.scale, 1e-9)

            ac = self._rate(self.ALPHA_CORR)
            rc = np.clip(r, -6.0, 6.0)
            self.corr = (1.0 - ac) * self.corr + ac * np.outer(rc, rc)

            ad = self._rate(self.ALPHA_DIFF)
            self.diff_scale = (1.0 - ad) * self.diff_scale + ad * diff

        self.prev_x = x.copy()
        return int(alarm)
