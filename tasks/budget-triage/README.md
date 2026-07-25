# budget-triage — task notes

Alert triage under a hard capacity constraint. The agent ranks a pool of
anomaly-detection alerts; the verifier investigates the top `k = 20` of each
period and scores the result against a **provable optimum**.

```
u(a) = severity(a) * max(0, 1 - age_days(a) / T_MAX)   if is_incident(a)
     = -COST                                            otherwise
T_MAX = 30.0   COST = 0.1   k = 20

achieved = Σ periods Σ u(agent's top-k)
optimal  = Σ periods Σ u(top-k under the TRUE utility order)
floor    = Σ periods (k × mean u over the period)      # analytic, not sampled
reward   = clamp((achieved - floor) / (optimal - floor), 0, 1)
```

Nothing in the reward is anchored on code we wrote. `optimal` is the best any
ordering could achieve, and `floor` is computed in closed form.

## Layout

```
task.toml                     schema 1.3; artifacts top-level (above all tables)
instruction.md                the agent's brief; ends with "## Time budget"
README.md                     this file
environment/
  Dockerfile                  python:3.12-slim + numpy 2.3.4
  .dockerignore               __pycache__/, *.py[cod]
  app/solution.py             weak starter: returns the input order
  app/run_dev.py              self-scoring on the dev split
  app/dev_alerts.json          6 periods, labels VISIBLE (3015 alerts)
tests/
  Dockerfile                  + util-linux (setpriv); labels chmod 0600 root
  .dockerignore               __pycache__/, *.py[cod]
  test.sh                     entrypoint; always (re)writes reward.txt
  run_verifier.py             root: payload, privilege drop, scoring
  predict.py                  the ONLY thing that imports agent code, as nobody
  grade.py                    reward.txt + metrics.json emission
  reward.py                   the scoring rule (pure stdlib, no I/O)
  test_alerts.json           12 periods, labels HIDDEN (6206 alerts)
  degenerate.py               anti-degeneracy + optimality harness
solution/{solution.py,solve.sh}   competent reference (~0.50)
cheat/{solution.py,solve.sh}      adversarial no-op (0.00)
provenance/data_gen.py            deterministic generator for both splits
```

## Measured numbers

All on the hidden 12-period test split (6206 alerts), via the real scoring path.

| submission | reward | precision@k | achieved |
|---|---|---|---|
| **starter** (`environment/app/solution.py`, identity order) | **0.0000** | 0.054 | −13.66 |
| **reference** (`solution/solution.py`) | **0.4955** | 0.433 | 121.28 |
| **cheat** (`cheat/solution.py`) | **0.0000** | 0.054 | −13.66 |

For scale: `optimal = 256.15`, `floor = −11.19`.

### Anti-degeneracy — every label-blind heuristic

| strategy | reward | precision@k |
|---|---|---|
| identity (order as given) | 0.0000 | 0.054 |
| reverse | 0.0000 | 0.050 |
| random (10 seeds) | min 0.0000 / mean 0.0059 / max 0.0254 | ~0.07 |
| sort `age_days` ascending | 0.1145 | 0.121 |
| sort `age_days` descending | 0.0000 | 0.067 |
| sort `features[0]` descending | 0.0094 | 0.087 |
| sort `features[0]` ascending | 0.0249 | 0.087 |

Reproduce with `python3 tests/degenerate.py --solution solution/solution.py
--verify-optimal --random-seeds 10`.

**Random is ≈0 by construction, and empirically so.** `floor` is the analytic
expectation of a uniformly random order (period mean × k), so a random ordering
scores 0 in expectation *by definition of the reward*. Measured over 10 seeds it
lands between 0.0000 and 0.0254 — the residual is ordinary sampling noise around
a floor the formula puts at exactly zero, not a bias.

**`sort age_days ascending` is the strongest degenerate at 0.115, and that is
structural rather than a leak.** The freshest alerts genuinely carry the largest
decay multiplier, so surfacing them is not *literally* worthless — the utility
function says so. It is still five times worse than a real ranker (0.115 vs
0.496) and nowhere near passing. Across ten different dataset seeds it ranges
0.037–0.130 (mean ≈ 0.075); the shipped seed sits at the high end of that range,
and this is disclosed rather than seed-shopped away.

Decomposing it on the shipped dataset: regenerating with the identical RNG
stream but the age↔incident coupling forced to zero still leaves age-ascending
at **0.0687** (precision@k 0.079, i.e. the base rate). So 0.069 of the 0.115 is
the irreducible decay effect — fresh alerts really are worth more — and only
0.046 comes from the coupling. That coupling has a **random sign** per period,
so age is partly informative within a period and carries no consistent direction
across them.

### `optimal` really is optimal

A period's score depends only on *which* 20 alerts the agent surfaces, so the
achievable maximum is the max over all C(n, 20) subsets. `tests/degenerate.py
--verify-optimal` brute-forces that directly on tractable pools:

```
brute-forced 200 random pools of n=14, k=5 (2002 subsets each): ALL MATCH
-> `optimal` is a provable optimum
```

The greedy "sort by `u`, take k" construction in `reward.py` matched the
exhaustive maximum on every pool. (It must: the objective is a sum of
independent per-item values, so the top-k by value *is* the argmax subset. The
brute force confirms the implementation, not the theorem.) An oracle ranker
scores exactly 1.0 on the full split.

### Time decay is doing real work

A ranker that identifies incidents **perfectly** but ignores `age_days`:

| ranker (uses hidden labels; diagnostics only) | reward |
|---|---|
| oracle — true utility order | 1.0000 |
| perfect classifier + true severity, **ignores age** | **0.7798** |
| perfect classifier only (ignores severity *and* age) | 0.6435 |

Perfect detection with no sense of timing forfeits 22% of the attainable range.
If that number had come out near 1.0 the time decay would be decorative and the
task would need more age dispersion — it does not. The generator was tuned to
this: severity dispersion was deliberately narrowed (log-severity σ = 0.57
overall, of which only 0.22 is noise; severity p10/p50/p90 = 0.82 / 1.73 / 3.65)
and the age distribution widened until the decay factor spans the full [0, 1]
range with a large point mass at zero — 27.8% of test alerts are fully aged out
(60% in the deliberately-old period 9), mean decay 0.42. Severity varies by
roughly 4× across its bulk while decay varies by ∞, so neither factor swamps the
other.

Complementarily, the reference ranking by `p_hat` alone scores 0.3973; adding
severity and decay to form the expected utility lifts it to 0.4955. Neither
"sort by predicted probability" nor "sort by age" is close to optimal — the
optimum requires all three factors.

### Reproducibility

The same `solution.py` must always earn the same reward. `tests/predict.py`
seeds `random` and `numpy.random` to 0 once, in the unprivileged child, before
any agent code is imported.

* A deliberately **unseeded** `random.shuffle` submission produced byte-identical
  orderings and the identical reward `0.005227272302255577` across two separate
  child processes.
* Control: with the seeding lines removed, the same submission produced
  **different** orderings run to run — so the seeding is load-bearing.
* The reference solution, scored twice in separate processes, is bit-identical:
  `0.49550330433506007 / 121.28040227299999 / 256.153819627 / -11.188699190161524`.

This is distinct from the `floor` term, which is the **analytic** expected
utility of a random order and is never sampled.

## The data

`provenance/data_gen.py` is deterministic from a fixed hyper-seed (20260724) plus
disjoint split seeds (dev 101, test 90210). Regenerate both files with
`python3 provenance/data_gen.py --stats`.

Both splits are drawn from one dataset-level hyper-distribution, so the mapping
is learnable across periods; every period re-draws its own parameters from it, so
nothing is memorisable period-by-period.

* **12 features.** `is_incident` is a Bernoulli draw from a logistic model over a
  *sparse* weighted combination (5 of 12 features carry linear signal) plus two
  pairwise interactions and one quadratic term. Learnable but not trivial, and
  Bernoulli sampling means an irreducible Bayes error — no ranker can classify
  perfectly from the features. Raw feature columns are shifted and scaled per
  index, so a solver that skips standardisation is handicapped.
* **Incident rate** 5–10% per period (one test period is forced to 1.9%).
* **`severity`** is log-linear in a *different*, partially overlapping sparse
  weight vector with a fat noise term: correlated with the features, not readable
  off them.
* **`age_days`** is a per-period mixture (exponential + uniform, both with
  per-period parameters) plus a coupling to the incident logit whose **sign is
  random per period**. Partly informative, partly not, by design.
* **Per period the generator re-draws**: the number of alerts (350–700), the
  incident rate, the incident and severity weight vectors (jittered around the
  hyper-mean), and the whole age distribution.

Split composition:

| split | periods | alerts | notes |
|---|---|---|---|
| dev (`environment/app/dev_alerts.json`) | 6 | 3015 | labels visible |
| test (`tests/test_alerts.json`) | 12 | 6206 | labels hidden |

The test split carries two deliberate stresses: **period 4** has an unusually low
incident rate (1.9%, 10 incidents in 520 alerts — fewer than the budget of 20),
and **period 9** has unusually old alerts (mean age 37.4 days, 60% already past
`T_MAX`).

Typical periods hold 24–49 incidents against a budget of 20, which is what makes
the *choice* of which incidents to investigate matter. Were incidents scarcer
than the budget everywhere, any perfect classifier would trivially be optimal.

## Verifier security

Root never imports the agent's code.

1. `chmod 700 /logs/verifier` and `chmod 600 /tests/test_alerts.json` (root) run
   before anything else.
2. Root builds a payload containing **only** `id`, `features`, `age_days`,
   `period`. Two assertions fire before it is written: structural (each row's key
   set is exactly those four) and textual (the serialised bytes contain no
   `"is_incident"`, `"severity"`, `"label"` or `"utility"`).
3. `tests/predict.py` — the only code that ever imports `/app/solution.py` — runs
   under `setpriv --reuid nobody --regid nogroup --clear-groups`, with a scrubbed
   environment and a 300 s timeout. If `setpriv` is missing, the verifier refuses
   to run agent code at all rather than falling back to root.
   The child's stdout/stderr go to **files**, and the verifier waits on the
   **direct child only** (killing it if it overruns) — never
   `subprocess.run(capture_output=True)`. See "The pipe-EOF defect" below.
4. The child writes its result through a **file descriptor inherited from root**
   whose directory is `0700` root — it cannot open, replace or symlink-swap the
   output path, only write bytes into the descriptor it was handed. Root also
   refuses to evaluate an `/app/solution.py` that is a symlink or not a regular
   file.
5. Root scores the returned ordering itself, from the labels, and writes a scalar
   float to `/logs/verifier/reward.txt` **last**. Diagnostics go to
   `metrics.json`. A `reward.json` is never created.
6. `/tests` is `0711`: the child can exec `/tests/predict.py` by exact path but
   cannot enumerate the directory. `reward.py`, `grade.py`, `run_verifier.py`
   and `degenerate.py` are `0600` root.
7. All of those modes are set in `tests/Dockerfile` **and re-applied at runtime**
   by `run_verifier.py` before any agent code runs, because some Harbor
   providers re-upload `tests/` into the container at verifier start and would
   otherwise reset them. `run_verifier.py` asserts the labels really are `0600`
   after doing so.

`test.sh` writes `0.000000` before doing anything and re-checks the file
afterwards, so a reward the agent left behind is never trusted and no code path
exits without a reward on disk.

### The pipe-EOF defect

`run_verifier.py` originally collected the child's output with
`subprocess.run(..., capture_output=True, timeout=...)`. That waits for **EOF on
the child's stdout/stderr pipes**, not for the child to exit. Any solution that
forks a background process inherits the write ends of those pipes and holds them
open, so EOF never arrives and the call blocks for the **entire** timeout — even
though the direct child finished in a fraction of a second. The submission is
then misreported as a timeout failure instead of being scored.

The fix (mirroring `tasks/changepoint-restraint/tests/run_verifier.py`, so both
verifiers behave identically): redirect the child's stdout/stderr to files under
the payload dir, `Popen` + `proc.wait(timeout=...)` on the direct child only,
`kill()` it if it overruns, and `_tail()` the log files afterwards for
diagnostics.

Measured with a deliberately short 20 s demonstration timeout (production is
300 s):

| submission | old `capture_output=True` | new files + direct-child wait |
|---|---|---|
| starter | 0.08 s ok | 0.11 s ok |
| reference | 0.09 s ok | 0.11 s ok |
| **cheat** | **20.02 s TIMEOUT** (the full timeout) | **0.11 s, scored normally** |

Under the old code the cheat was never actually *scored* — it was written off as
a timeout, which happened to also give 0.0. In production it would have burnt
the full 300 s doing so. The whole three-case end-to-end run now takes 0.47 s.

The rewards are unchanged: starter 0.0000, reference 0.4955, cheat 0.0000, and
every degenerate identical to the table above.

Note that `cheat/solution.py`'s daemon **deliberately leaves stdout and stderr
inherited** (it redirects only stdin) precisely so the shipped cheat exercises
this failure mode. If someone reintroduces pipe-based capture, the cheat run
will visibly stall for the full timeout again. The earlier version of the cheat
`dup2`'d `/dev/null` over fds 0/1/2, which released the pipes and masked the
bug — it was found in a sibling task and confirmed here.

### The cheat, and why it scores 0.0

`cheat/solution.py` does no analysis. It writes `1.0` into five plausible reward
paths, forks a detached double-`fork()`/`setsid()` daemon that keeps rewriting
them for 600 s hoping to race the verifier's final write (and which holds the
inherited stdout/stderr open — see "The pipe-EOF defect"), globs
`/tests /app /tmp /opt` for anything containing `is_incident`/`severity`, and
only then falls back to returning the input order.

It scores **0.0000**. Every avenue is closed: `/logs/verifier` is `0700` root and
the cheat is `nobody`, so neither the direct write nor the daemon lands;
`/tests/test_alerts.json` is `0600` root, so the glob finds nothing; and the
fallback is the identity order, which the normalisation puts at zero.

**Positive control**: pointed at a world-readable copy of the same labels, the
identical cheat code scores **1.0000** and reports `[cheat] leaked 6206 labels`.
The 0.0 is bought by the file permissions and the privilege drop, not by a cheat
that does not work.

## Harbor mechanics worth remembering

* `schema_version = "1.3"`. The docs show 1.4; the 0.20.0 release rejects it.
* `artifacts = ["/app/solution.py"]` must be **top level** — and therefore
  **above every `[table]` header**. TOML binds a bare key to the table that
  precedes it, so writing it at the bottom of the file silently makes it
  `[environment].artifacts`; nothing is collected and the trial scores 0 with no
  error. This was caught during validation of this very file.
* Single container, no `docker-compose.yaml`.
* No `network_mode` anywhere. The default `public` baseline is required: the
  agent CLI installs under the `[environment]` baseline, and a differing
  `[agent]` policy would need `dynamic_network_policy`, which non-Docker backends
  lack. The task itself is fully offline — numpy is baked into both images.
* `[verifier] environment_mode = "separate"`, so the hidden labels and `setpriv`
  live in an image the agent never sees. Artifacts re-materialise at their
  original path, so `/app/solution.py` lands at `/app/solution.py`.
* `[agent] timeout_sec = 3600.0` and the `## Time budget` section of
  `instruction.md` are one fact in two places — change them together.

## Notes for the next person

* **A submission must be self-contained.** The verifier container has
  `/app/solution.py` and nothing else from the agent's working directory —
  `dev_alerts.json` is not there. This is stated in `instruction.md`. The
  reference demonstrates the pattern: its coefficients are baked in as literals,
  regenerable with `python3 solution/solution.py --refit
  environment/app/dev_alerts.json`. Its `rank()` is pure stdlib and needs no
  numpy at scoring time.
* `environment/app/run_dev.py` deliberately re-implements the scoring rule rather
  than importing `tests/reward.py` (the agent must not see `tests/`). The two
  were checked bit-for-bit equal on both splits across five random orderings. If
  you change one, change both, and re-run that check.
* The reference scores 0.5638 in-sample on dev and 0.4955 out-of-sample on test.
  It is a plain linear logistic regression; a full quadratic feature expansion
  reaches 0.582 on the same pipeline. The reference is deliberately *not* the
  ceiling — there is real headroom between it and 1.0.

## What has not been verified

No Docker daemon was available in this environment, so the following are
reasoned-through but **not executed**:

* Both image builds (`environment/Dockerfile`, `tests/Dockerfile`), including the
  `apt-get install util-linux` and `pip install numpy==2.3.4` steps.
* The actual `setpriv --reuid nobody --regid nogroup --clear-groups` privilege
  drop, and hence that `nobody` genuinely cannot read a `0600` root
  `test_alerts.json` or write a `0700` root `/logs/verifier`. The scoring logic
  either side of that boundary was exercised in full; the enforcement was not.
  Everything else about the cheat is verified, including — via the positive
  control — that the labels are the only thing standing between it and 1.0.
* Harbor's own artifact collection: that `/app/solution.py` is picked up from the
  agent container and re-materialised at `/app/solution.py` in the verifier. The
  `task.toml` was validated against `harbor.models.task.config.TaskConfig`, but no
  trial was run.
* The agent-phase network baseline, and that the agent CLI installs cleanly under
  it.
* `numpy` importing successfully inside the unprivileged child. The reference
  sidesteps this (stdlib-only at scoring time), but an agent solution that uses
  numpy in `rank()` depends on it.
