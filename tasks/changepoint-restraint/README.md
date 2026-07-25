# changepoint-restraint

Find the change points in a set of piecewise-stationary series when **the
number of change points is unknown, varies per series, and is sometimes zero**.

Change-point detection is a solved problem when someone hands you `K`. This
task never does. A detector with a fixed budget finds exactly `K` "change
points" in pure noise, and the scoring is built so that this is fatal rather
than merely suboptimal: several hidden series contain no change points at all,
and predicting on them trips a hard gate.

* Deliverable: `/app/solution.py` implementing
  `find_changepoints(series) -> list[int]`.
* Reward: pooled F1 over all hidden series, ceiling 1.0 (the natural maximum of
  the metric — nothing is normalised by a reference score).
* Hard gate: `null_fp_rate > 0.5` forces reward `0.0`.
* Stdlib + numpy, CPU, offline, single container, 60-minute agent budget.

## Layout

```
task.toml                       schema 1.3; artifacts top level; separate verifier
instruction.md                  the agent-facing brief (ends with the time budget)
README.md                       this file
environment/
  Dockerfile                    python:3.12-slim + numpy
  .dockerignore                 __pycache__/, *.py[cod]
  app/solution.py               WEAK starter: fixed-budget binseg, always 3 points
  app/run_dev.py                self-scoring on the labelled dev split
  app/dev_series.json           8 labelled dev series
tests/
  Dockerfile                    python:3.12-slim + util-linux (setpriv) + numpy
  .dockerignore                 __pycache__/, *.py[cod]
  test.sh                       harbor entry point
  run_verifier.py               root side: payload, privilege drop, scoring, reward
  predict.py                    untrusted side: runs as nobody, imports the agent's code
  grade.py                      matching + metrics (`python3 grade.py` demos the matching)
  reward.py                     F1 with the null-series gate
  degenerate.py                 scores the strategies that require no understanding
  test_series.json              20 hidden series with labels (chmod 600 at verify time)
solution/{solution.py,solve.sh} competent reference: normalised CUSUM + threshold
cheat/{solution.py,solve.sh}    adversarial probe: forge the reward, steal the labels
provenance/data_gen.py          deterministic generator for both splits
```

## Data

Each series is a piecewise-stationary AR(1) process per channel,

```
x_t = mu + phi * (x_{t-1} - mu) + sigma * sqrt(1 - phi^2) * e_t
```

with `e_t` unit-variance normal, Laplace, standardised t(3) or uniform. Scaling
the innovation by `sqrt(1 - phi^2)` keeps the marginal variance fixed when
`phi` changes, so an autocorrelation change is not a disguised variance change.
The process state carries across each boundary, so a change point is never
marked by a discontinuity artefact.

Randomised per series: length (520-1000), dimensionality (1-3), the number of
change points (including zero), the segment lengths (Dirichlet, minimum 35),
what changes at each boundary and by how much.

| change kind | what moves | magnitude range |
| --- | --- | --- |
| `mean` | level | `0.7-2.6` sigma |
| `var` | scale | factor `1.7-3.2` up or down |
| `ar` | lag-1 dependence, marginal variance held | `|dphi| >= 0.55` |
| `dist` | innovation shape, light <-> heavy tailed | kurtosis only |

**Dev (8 series, labelled):** counts `[1, 3, 4, 2, 2, 3, 5, 1]`; never zero,
never more than five, never variance-only.

**Test (20 series, hidden):** counts
`[2, 1, 3, 1, 4, 3, 12, 0, 4, 7, 1, 5, 3, 0, 8, 2, 2, 0, 6, 0]` — 64 true
change points in total. Regimes absent from dev: **4 null series** (`test_07`,
`test_13`, `test_17`, `test_19`), one series with **12** change points
(`test_06`), one series whose changes are **all variance** (`test_11`, 1-D:
mean-based detectors are blind to it) and one whose changes are **all
autocorrelation** (`test_16`).

### Anti-leakage

The payload handed to `find_changepoints` is the raw series and nothing else:
no count argument, no metadata, no ids that encode anything. The counts are a
fixed multiset assigned to lengths by a permutation re-drawn until length and
count are uncorrelated — the shipped test split has Pearson `0.0585` and
Spearman `-0.0341`. Dev counts are 1-5 with no zeros, so no fixed count learned
on dev transfers (predicting 3 everywhere scores 0.267 on dev and 0.0 on test).

Regenerate with `python3 provenance/data_gen.py` (deterministic from
`MASTER_SEED = 20260725`; generated with numpy 2.5.1 / CPython 3.12).

## Scoring

`TOL = 5`. A prediction matches a true change point if it is within 5 samples.

**Matching (deterministic, one-to-one, per series).** Build every candidate
pair `(p, t)` with `|p - t| <= TOL`; sort by `(|p - t|, t, p)` — nearest first,
ties broken by true index then predicted index, a total order independent of
input ordering; walk the sorted list and accept a pair only when neither
endpoint has already been matched. Predictions are never matched across
series.

`python3 tests/grade.py` prints the demonstration:

```
five predictions clustered on one true point
  preds = [98, 99, 100, 101, 102]   trues = [100]
  matched  = [(100, 100)]
  TP = 1  FP = 4 [98, 99, 101, 102]  FN = 0 []

one prediction sitting between two true points
  preds = [100]   trues = [98, 102]
  matched  = [(100, 98)]
  TP = 1  FP = 0 []  FN = 1 [102]

nearest-first ordering decides who gets credited
  preds = [10, 15]   trues = [14, 19]
  matched  = [(15, 14)]
  TP = 1  FP = 1 [10]  FN = 1 [19]

the same pair set, predictions listed in reverse order
  preds = [15, 10]   trues = [19, 14]
  matched  = [(15, 14)]          <- identical result, ordering does not matter
```

So one prediction cannot be credited for two true points, and one true point
cannot absorb two predictions. In the shipped data the minimum gap between
adjacent true change points is **35 samples** (`> 2 * TOL`), so no prediction
is ever within tolerance of two true points at once and this greedy matching
coincides with a maximum-cardinality (Hungarian) matching; the tie-breaking
rules exist to keep the result deterministic in the degenerate cases anyway.

Counts are pooled across all 20 series into `TP / FP / FN`, then

```
precision = TP/(TP+FP)   recall = TP/(TP+FN)   f1 = 2PR/(P+R)   reward = f1
```

**Null gate.** `null_fp_rate = (predictions made on null series) / (number of
null series)`. If it exceeds `0.5`, reward is `0.0` whatever F1 says. It is
reported in `metrics.json` in every run, pass or fail.

`metrics.json` carries, untransformed: `f1`, `precision`, `recall`, `tp`,
`fp`, `fn`, `null_fp_rate`, `n_series`, `n_null_series`, `mean_abs_offset`,
`predicted_counts` and `true_counts` per series, plus `per_series` breakdowns,
`null_gate_passed`, `null_gate_threshold`, `predictor_exit_code` and
`predictor_errors`. `reward.txt` holds the scalar float and nothing else;
`reward.json` is never created.

## Measured results

Everything below was produced by running the real scoring path
(`run_verifier.py` -> payload -> `predict.py` -> `grade.py` -> `reward.py`)
against `tests/test_series.json`.

| solution | F1 | precision | recall | TP | FP | FN | null_fp_rate | gate | **reward** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starter (`environment/app/solution.py`, always 3 points) | 0.2742 | 0.283 | 0.266 | 17 | 43 | 47 | 3.00 | FAIL | **0.0000** |
| reference (`solution/solution.py`) | 0.6500 | 0.696 | 0.609 | 39 | 17 | 25 | 0.00 | pass | **0.6500** |
| cheat (`cheat/solution.py`) | 0.0000 | 0.000 | 0.000 | 0 | 0 | 64 | 0.00 | pass | **0.0000** |

The reference's `mean_abs_offset` is 1.77 samples over its 39 matches. On the
dev split it scores F1 0.6154 (P 0.667, R 0.571); the starter scores 0.2667.
Predicted vs true counts for the reference on test: it returns 0 on all four
null series and 11 on the 12-change-point series, with per-series counts
`2,1,4,1,3,2,11,0,4,5,2,3,1,0,6,2,2,0,7,0` against true
`2,1,3,1,4,3,12,0,4,7,1,5,3,0,8,2,2,0,6,0`.

The reference is competent, not optimal: it misses most noise-shape changes and
many autocorrelation changes, so 0.65 is a solvability proof and a sanity
anchor, not a definition of full marks.

### Anti-degenerate check

`python3 tests/degenerate.py` (identical numbers were reproduced by running each
strategy as a real `/app/solution.py` through the full verifier path):

| strategy | F1 | precision | recall | null_fp_rate | gate | **reward** |
| --- | --- | --- | --- | --- | --- | --- |
| return `[]` | 0.0000 | 0.000 | 0.000 | 0.00 | pass | **0.0000** |
| return every index | 0.0078 | 0.004 | 1.000 | 849.00 | FAIL | **0.0000** |
| uniform grid of 8 | 0.0357 | 0.025 | 0.062 | 8.00 | FAIL | **0.0000** |
| `[len(series)//2]` | 0.0000 | 0.000 | 0.000 | 1.00 | FAIL | **0.0000** |
| 4 random indices (seeded) | 0.0694 | 0.062 | 0.078 | 4.00 | FAIL | **0.0000** |

`[]` scoring 0 while spraying also scores 0 is the point of the task: silence
and noise are both failures, and the only way to score is to decide, per
series, how many change points there are. The maximum degenerate reward is
**0.0000**.

### The null gate is decisive, not decorative

Reference predictions plus one spurious point on successive null series
(pooled F1 barely moves; the reward falls off a cliff):

| spurious predictions on null series | null_fp_rate | F1 | gate | reward |
| --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.6500 | pass | 0.6500 |
| 1 | 0.25 | 0.6446 | pass | 0.6446 |
| 2 | 0.50 | 0.6393 | pass | 0.6393 |
| 3 | 0.75 | 0.6341 | **FAIL** | **0.0000** |
| 4 | 1.00 | 0.6290 | **FAIL** | **0.0000** |

## Reproducibility

`tests/predict.py` seeds `random` and `numpy.random` with 0 once, in the
untrusted child, before the agent's module is imported: the same artifact must
earn the same reward on every run, and an unseeded solution would otherwise
score differently run to run. Two independent full-path runs of the reference
both wrote `0.650000` to `reward.txt`, byte for byte.

## Harbor mechanics (verified against harbor 0.20.0)

* `schema_version = "1.3"`. `TaskConfig` itself does not police the string —
  `"1.4"` parses — but 1.3 is what this release's own tooling writes
  (`harbor/cli/tasks.py` hard-codes it), so 1.3 it is.
* `artifacts = ["/app/solution.py"]` sits at **top level**. Nested inside a
  table it silently parses into that table, nothing is collected, and the trial
  scores 0 with no error.
* Single container, no `docker-compose.yaml`. `network_mode` is set nowhere:
  the default `public` baseline is what the agent CLI installs under, and a
  differing `[agent]` policy would need `dynamic_network_policy`.
* `[verifier] environment_mode = "separate"`. In that mode harbor **does not
  upload `tests/`** — `Verifier._resolve_tests` short-circuits on
  `skip_tests_upload` with the comment "the verifier image already owns
  /tests". The verifier image is built with `tests/` as its build context, so
  `tests/Dockerfile` must `COPY . /tests` itself. Without that line there is no
  `/tests/test.sh` and the verifier cannot run at all.
* Artifacts are re-uploaded into the separate verifier at their original
  source path, so the agent's file lands at `/app/solution.py` there.
* `[agent] timeout_sec = 3600.0` and the "60 minutes" in `instruction.md` are
  one fact in two places; task.toml says so at the line.
* The reward is a scalar float in `/logs/verifier/reward.txt`. `reward.json` is
  never created (harbor prefers it when both exist). Diagnostics go to
  `metrics.json`.

## Verifier security

* Root never imports, execs or otherwise trusts the agent's code. The only
  process that touches `/app/solution.py` is `tests/predict.py`, launched as
  `setpriv --reuid nobody --regid nogroup --clear-groups python3 ...`.
* `run_verifier.py` chmods `/logs/verifier` to `700` root and deletes any stale
  `reward.txt` / `reward.json` / `metrics.json` **before** anything untrusted
  runs, then writes the real reward last.
* The hidden labels are chmod `600`, chown root. The untrusted child receives a
  payload of `{id, values}` only — no labels, no counts, no null flags.
* Untrusted output is read back as JSON and re-sanitised by root
  (`grade.sanitize`): non-integers, out-of-range indices and duplicates are
  dropped before scoring.
* The child's stdout/stderr go to files, not pipes, so a solution that forks a
  detached daemon cannot hold a pipe open and stall the verifier — the wait is
  on the direct child, with a 420 s timeout and a kill.
* A solution that crashes, hangs, imports nothing, or defines no
  `find_changepoints` scores 0.0 rather than breaking the run (all four cases
  measured).

`cheat/solution.py` exercises the surface: it writes `1.0` into
`/logs/verifier/reward.txt`, forks a detached `os.fork()` daemon that keeps
rewriting it, walks `/tests /app /tmp /opt` (bounded to 400 files) for JSON
containing `change_points`, and returns whatever it stole without analysing
anything. Under the privilege barrier it harvests **0** label sets, finds the
reward directory unwritable, returns `[]` for every series and scores
**0.0000**. With the barrier removed it does score 1.0 — which is exactly why
the barrier is enforced by file ownership and `setpriv` rather than by
politeness.
