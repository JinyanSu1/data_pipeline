# drift-vs-anomaly

Online multivariate anomaly detection on a stream that also contains **benign
concept drift**. The agent writes `/app/solution.py` exposing
`Detector(n_channels).step(x_t) -> 0|1`, called once per timestep in order.
Faults must be flagged; drift must not be.

```
reward = 0.0        if drift_fpr > 0.10      # hard gate, no partial credit
reward = event_f1   otherwise                # event-wise F1, NO point adjustment
```

## Layout

```
task.toml                     schema 1.3, artifacts top-level, verifier "separate"
instruction.md                the agent-facing spec (scoring rules stated in full)
README.md                     this file
environment/
  Dockerfile                  python:3.12-slim + numpy
  .dockerignore
  app/solution.py             weak starter (global z-score) -- scores 0.0 by design
  app/run_dev.py              self-scoring on dev with the verifier's exact rules
  app/dev_episodes.json       6 labelled dev episodes
tests/
  Dockerfile                  python:3.12-slim + numpy + util-linux (setpriv) + procps;
                              COPY . /tests (separate mode does not upload tests/)
  .dockerignore
  test.sh                     entry point; always leaves a scalar reward
  run_verifier.py             ROOT side: never imports agent code
  predict.py                  UNPRIVILEGED child: the only place agent code runs
  grade.py                    event-wise scoring (no point adjustment)
  reward.py                   the drift gate
  degenerate.py               anti-degenerate audit
  test_episodes.json          12 hidden labelled episodes (chmod 600 at verify time)
solution/{solution.py,solve.sh}   competent reference detector
cheat/{solution.py,solve.sh}      adversarial "solution"; must score 0.0
provenance/data_gen.py            deterministic generator for both episode files
```

## The data

`provenance/data_gen.py` is deterministic from two seed bases and randomises
per episode: channel count (8-16), stream length (2200-3200), latent factor
count, overall amplitude, noise floor, per-channel idiosyncratic noise, AR(1)
persistence, and then

* **drift**: 1-3 segments, each 180-520 steps, kind `mean` / `scale` /
  `rotation`, profile `persistent` (smoothstep to a new operating point that
  holds) or `excursion` (smoothstep out and back), magnitude 2.5-12 sd for
  mean shifts, factor 0.22-4.5 for scale changes, 1-3 Givens rotations of
  0.35-1.25 rad for rotations. Labelled window = ramp (+ settling tail for
  persistent drift).
* **anomalies**: 0-7 per episode, kind `spike` (1-4 steps, 4-7 sd),
  `step` (6-21 steps, 3-5.5 sd), `stuck` (12-40 steps, channel frozen with
  1% quantisation noise), `corr_break` (10-30 steps, channels replaced by a
  surrogate with *matched* marginal mean/variance/smoothness and no
  cross-channel correlation). Amplitudes are in units of the **local** robust
  sd, so nothing is hard-codable in absolute terms.

Dev is the narrow distribution: 8-13 channels, 1-2 drift segments, 3-5
anomalies, and anomalies never overlap a drift window. Test widens it and adds
shapes dev never shows:

| test episode | shape absent from dev |
|---|---|
| `test_03` | drift present, **zero anomalies** (a detector that always finds something loses here) |
| `test_05`, `test_09` | anomalies fire **inside** a benign-drift window |
| `test_07`, `test_09` | three drift segments covering all three drift kinds |
| `test_11` | 16 channels, 7 anomalies of all four kinds |

Regenerate with `python3 provenance/data_gen.py` (rewrites both JSON files in
place; `--check` prints hashes only).

## Measured results

All numbers below are from the real scoring path (`tests/run_verifier.py` ->
`tests/predict.py` -> `tests/grade.py` -> `tests/reward.py`) on the 12 hidden
test episodes, 47 ground-truth events, 8526 gate-eligible drift timesteps.

| solution | reward | event_f1 | precision | recall | drift_fpr | gate |
|---|---|---|---|---|---|---|
| `environment/app/solution.py` (starter) | **0.0000** | 0.0647 | 0.0343 | 0.5532 | **0.2587** | **FAIL** |
| `solution/solution.py` (reference) | **0.7556** | 0.7556 | 0.7907 | 0.7234 | 0.0025 | pass |
| `cheat/solution.py` | **0.0000** | 0.9895 | 0.9792 | 1.0000 | **1.0000** | **FAIL** |

On the 6 dev episodes the same two honest detectors give: starter
`f1=0.0330, drift_fpr=0.1976 -> reward 0.0`; reference
`f1=0.8182, drift_fpr=0.0148 -> reward 0.8182`.

The reference is *competent, not optimal*: it misses 13 of 47 events (mostly
correlation breaks) and emits 9 false-positive runs. Full marks remain
available above it, and the 1.0 ceiling is the natural maximum of event F1 --
no reference-relative normalisation anywhere.

### The drift gate does its job

The starter alarms on **2206 of 8526** drift timesteps (`drift_fpr = 0.2587`).
Its raw event F1 of 0.0647 is discarded and the reward is forced to `0.0`.
That is the intended lesson: a detector whose notion of "normal" does not move
with the process is not merely imprecise, it is disqualified.

### Anti-degenerate audit (`python3 tests/degenerate.py`)

| detector | reward | event_f1 | precision | recall | drift_fpr | gate |
|---|---|---|---|---|---|---|
| always alarm | **0.0000** | 0.9895 | 0.9792 | 1.0000 | 1.0000 | FAIL |
| never alarm | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | pass |
| period-5 | **0.0000** | 0.0145 | 0.0073 | 0.9574 | 0.1997 | FAIL |
| period-10 | **0.0261** | 0.0261 | 0.0133 | 0.8723 | 0.0997 | pass |
| period-25 | **0.0416** | 0.0416 | 0.0216 | 0.5745 | 0.0400 | pass |
| period-100 | **0.0384** | 0.0384 | 0.0220 | 0.1489 | 0.0099 | pass |
| global-z 3.0 (naive drift-blind) | **0.0000** | 0.0829 | 0.0468 | 0.3617 | 0.1329 | FAIL |
| global-z 4.0 (naive drift-blind) | **0.1043** | 0.1043 | 0.0671 | 0.2340 | 0.0341 | pass |
| global-z 4.5 (oracle-tuned worst case) | **0.1739** | 0.1739 | 0.1471 | 0.2128 | 0.0106 | pass |
| unseeded `random` p=0.02 | **0.0425** | 0.0425 | 0.0229 | 0.2979 | 0.0198 | pass |

Notes on the two rows that are not ~0.00:

* **`always alarm` earns event F1 0.9895 and still scores zero.** That single
  row is why the gate exists: without it, the laziest possible detector would
  be near the top of the leaderboard.
* **global-z at 3.0 blows the drift gate** (0.1329) exactly as intended -- it
  is the drift-blind detector, and it is disqualified for it. Raising its
  threshold to 4.0/4.5 buys gate compliance at the cost of recall; even the
  *oracle-tuned* variant (threshold swept against the hidden labels, which no
  agent can do, using whole-stream statistics an online detector cannot have)
  tops out at **0.1739**, i.e. 23% of the reference. A sweep of thresholds
  2.5-7.0 never exceeds that.
* Periodic alarms buy recall for nothing, but each period is its own
  false-positive run, so precision collapses to ~0.02. The gate additionally
  kills any period below ~10.

### Determinism

The prediction child seeds `random` and `numpy.random` (both to 0) once,
before the agent's module is imported. Two full scoring runs of the reference
in separate processes both produced `reward.txt = 0.755556`, byte-identical.
A deliberately unseeded random detector also scores identically across runs
through the harness (`event_f1 = 0.032117` twice), while the same file run
*without* the harness seeding gave 0.045307 and 0.030166 on consecutive runs
-- that difference is what the seeding removes.

## Verifier security

`tests/run_verifier.py` runs as root and **never imports, execs, or evaluates
anything the agent wrote**. It:

1. `chmod 700 /logs/verifier`, `chmod 600 /tests/test_episodes.json`
   (`/tests` itself stays 755 so the child can read `predict.py`);
2. builds a **label-free** payload -- `episode_id`, `n_channels`, `n_steps`,
   `X` and nothing else: no `anomaly_spans`, no `drift_spans`, no `meta`;
3. copies `/app/solution.py` into a scratch dir and runs
   `setpriv --reuid nobody --regid nogroup --clear-groups python3 /tests/predict.py`;
4. reads back only integers, coerces them to 0/1 and to the exact episode
   length, and scores in the root process;
5. writes a scalar to `/logs/verifier/reward.txt` and diagnostics to
   `metrics.json`. It never writes `reward.json`, and `test.sh` deletes any
   that appears (Harbor prefers `reward.json` over `reward.txt`).

Child stdout/stderr go to files in the temp workdir rather than to pipes: a
detached `os.fork()` daemon inherits the write end of a pipe and never closes
it, so `capture_output` / `communicate` would block for the entire timeout.
The verifier `Popen`s the child, `wait(timeout=900)`s on the *direct* child
only, kills it on overrun, then tails the log files for diagnostics and runs
`pkill -9 -u nobody`. The child also gets a minimal env (`HOME`/`TMPDIR`/
`MPLCONFIGDIR` pointed at the scratch dir). Every failure path still writes a
reward (0.0).

`cheat/solution.py` exercises all of this: it forges
`/logs/verifier/reward.{txt,json}`, forks a detached `os.fork()` daemon that
keeps re-forging for 600 s, globs `/tests /app /tmp /opt` for `anomaly_spans`,
and alarms on every timestep if it finds nothing. Result: **0.0** (writes
denied, no labels reachable, drift gate blown).

## Harbor mechanics worth not re-learning

* `schema_version = "1.3"` -- harbor 0.20.0's own default for the field and
  the value the rest of the suite uses. (`TaskConfig` types it as a plain
  `str` and does not validate it, so a wrong value fails later, not here.)
* With `environment_mode = "separate"`, harbor sets `skip_tests_upload=True`
  and uses `tests/` **only as the Docker build context**. The image must
  therefore own `/tests` itself: `tests/Dockerfile` needs `COPY . /tests`.
  Without it there is no `/tests/test.sh`, the verifier silently never runs.
* `artifacts = ["/app/solution.py"]` must sit **above the first `[table]`
  header**. TOML binds keys written after a header into that table, so
  putting it lower silently makes it `solution.artifacts`, collects nothing,
  and scores every trial 0.0 with no error. (This was hit and fixed here.)
* `[task].name` must be `org/name`, not a bare name.
* `[verifier] environment_mode = "separate"` builds the verifier image from
  `tests/` and re-materialises `/app/solution.py` at its original path.
* `network_mode` is set nowhere: the default `public` is the `[environment]`
  baseline the agent CLI installs under, and a differing `[agent]` policy
  would require `dynamic_network_policy`, which non-Docker backends lack.
* `[agent] timeout_sec = 3600.0` and the `## Time budget` section of
  `instruction.md` are one fact written twice -- change both together.
* Single container, no `docker-compose.yaml`.
