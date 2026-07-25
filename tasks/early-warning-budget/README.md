# early-warning-budget

Online change-point detection under an explicit earliness/false-alarm budget.
The agent writes a causal streaming detector; the reward is how many timesteps
of warning it buys before the break becomes obvious, gated on not crying wolf.

* Harbor `schema_version = "1.3"`, single container, `[verifier]
  environment_mode = "separate"`.
* CPU only, offline at scoring time, Python stdlib + numpy.
* Agent budget 60 minutes (`[agent] timeout_sec = 3600.0`).
* Collected artifact: `/app/solution.py` (top-level `artifacts` in task.toml).

## The problem

24 channels in 6 groups of 4. Long calm prefix of independent noise, then a
regime break in three stages: a correlation **build-up** inside one group, a
**spread** to more groups, and finally an **obvious** phase where a system-wide
factor makes everything move together. The agent must alarm during the
build-up.

`t_obvious` is published and deterministic — first `t >= 19` at which the mean
of the 276 off-diagonal entries of the trailing-20-step correlation matrix
exceeds 0.7 — and is computed by the generator on exactly the rounded values
that ship in the JSON files. The starter solution implements that rule
verbatim, and it alarms at exactly `t_obvious` on all 12 test episodes, which
is the cleanest possible confirmation that the label and the data agree.

### Scoring

```
lead_i  = t_obvious_i - t_alarm_i
        = 0 if never alarmed, or alarmed at/after t_obvious,
          or alarmed strictly before onset      (a false alarm is not a detection)
        = excluded if the episode has no t_obvious

fpr     = calm-prefix alarms / calm-prefix timesteps, pooled over ALL episodes
          (including the ones with no t_obvious)

reward  = 0.0                                    if fpr > 0.05
        = clamp(mean_lead_steps / L_MAX, 0, 1)   otherwise
```

`metrics.json` reports `mean_lead_steps`, `fpr`, `n_never_alarmed`,
`per_episode_lead` and `reward` untransformed, plus per-episode detail.

### L_MAX = 46.7

An information bound, derived in `provenance/data_gen.py` and hard-coded in
`tests/reward.py` with the full derivation table in its docstring:

> mean, over the 10 hidden test episodes that have a `t_obvious`, of
> `t_obvious - onset` — the gap between the first timestep on which any
> build-up signal exists in the data and the timestep at which the break
> becomes obvious.

`(44+56+60+31+47+50+39+42+37+61)/10 = 46.7`. Reaching 1.0 would mean detecting
a correlation change from its first sample with zero estimation delay. Nothing
about any reference implementation enters this number; re-run the generator and
it prints it.

## Measured results

All numbers below are from the real scoring path (`tests/predict.py` child +
`tests/grade.py` + `tests/reward.py`) on the 12 hidden test episodes.

| submission | reward | mean_lead | fpr | never alarmed |
|---|---|---|---|---|
| `environment/app/solution.py` (starter) | **0.0000** | 0.0 | 0.00000 | 2 |
| `solution/solution.py` (reference) | **0.4818** | 22.5 | 0.00000 | 0 |
| `cheat/solution.py` | **0.0000** | 0.0 | 1.00000 | 0 |

The reference detects every episode, 10–41 steps ahead of `t_obvious`, with
zero calm-prefix alarms. It leaves ~52% of the bound on the table, which is
the point: it proves solvability and acts as a regression baseline, it does not
define full marks.

The starter's 2 "never alarmed" episodes are exactly the two test episodes
whose break never reaches the obvious phase — the whole-system correlation
never crosses 0.7 there, so a detector built on that rule correctly stays
silent.

### Degenerate strategies (`tests/degenerate.py`)

| strategy | reward | mean_lead | fpr | gate |
|---|---|---|---|---|
| always alarm (`return 1`) | **0.0000** | 0.000 | 1.00000 | **GATED** |
| never alarm (`return 0`) | **0.0000** | 0.000 | 0.00000 | pass |
| fixed timestep, dev-optimal `t=666` | **0.0000** | 0.000 | 0.00075 | pass |
| fixed timestep `t=300` | **0.0128** | 0.600 | 0.00092 | pass |
| fixed timestep `t=600` | **0.0000** | 0.000 | 0.00075 | pass |
| fixed timestep `t=900` | **0.0000** | 0.000 | 0.00059 | pass |
| random, p=0.02/step (seeded) | **0.0000** | 0.000 | 0.01717 | pass |
| random, p=0.02/step (unseeded module RNG) | **0.0000** | 0.000 | 0.02001 | pass |

Worst degenerate: **0.0128**.

Two design decisions are what make these collapse, and both were forced by
measurement rather than assumed:

* **`p = 0.02` random passes the FPR gate** (0.017–0.020 < 0.05), so the gate
  alone does not kill it. What kills it is that an alarm raised before `onset`
  scores a lead of 0. A 2%-per-step detector fires within the first ~50 steps
  of a calm prefix that is 267–1707 steps long, so essentially every episode is
  a false alarm and contributes nothing. Without that clause this strategy
  would have scored *above* the reference, since its alarms land far to the
  left of `t_obvious`.
* **Hard-coded timesteps** are defeated by the onset distribution. Onsets are
  placed on a jittered grid with spacing 130, which exceeds the largest
  attainable `t_obvious - onset` (61), so the per-episode detection windows
  `[onset, t_obvious)` are pairwise disjoint and a single constant can be
  inside at most one of them.

The residual `0.0128` is one episode partially caught by `t=300`. For
completeness, the **oracle-best** constant on this very test set is `t=1198` →
`0.1306`. That is not attainable by an agent — it requires reading the hidden
onsets — and it is exactly the arithmetic floor `max_i(t_obvious_i - onset_i) /
(n_scored × L_MAX) = 61 / (10 × 46.7)`, i.e. the value *any* strategy gets for
happening to nail one episode out of ten. With 12 fixed episodes that floor
cannot be pushed lower without changing the averaging rule, so it is reported
as a diagnostic rather than suppressed. `tests/degenerate.py` prints it and
fails if any *realizable* strategy exceeds 0.15.

### Reproducibility

`tests/predict.py` seeds `random` and `numpy.random` once, in the unprivileged
child, before the agent's module is imported. Verified two ways:

* the reference scored twice in separate processes returns a bit-identical
  `reward = 0.481799`;
* the "unseeded module RNG" degenerate returns an identical reward and an
  identical `per_episode_lead` vector on a re-run, which it would not without
  the seeding.

The generator is also deterministic: re-running `provenance/data_gen.py`
reproduces both JSON files byte for byte.

## Verifier security

Root never imports, execs or evals the agent's code.

```
run_verifier.py (root)
  chmod 700 /logs/verifier                     <- reward channel locked
  rm reward.json / rewards.json, write 0.0     <- destroy anything preplanted
  chmod 600 /tests/test_episodes.json          <- ground truth root-only
  copy /app/solution.py -> /tmp/ewb_*/  mode 444, dir 755 (child cannot write)
  setpriv --reuid nobody --regid nogroup --clear-groups python3 /tests/predict.py ...
      |  scrubbed env, cwd = the 755 workdir
      |  {"cmd":"step","x":[...]} ---> one row
      |  <--- {"d":0|1}                one decision
  grade.py + reward.py on the returned bit vectors
  write reward.txt (644), metrics.json, chmod 755 /logs/verifier
```

`/logs/verifier` is relaxed from 700 to 755 only after the child is dead and
the real reward has been written, because that directory is a bind mount the
harness reads back from the host, possibly as a non-root user. At 755 with a
root-owned 644 `reward.txt`, a surviving `nobody` process still cannot write it.

### Streaming instead of a payload file

The child holds **no episode array at all**. Root sends one observation and
waits for the decision before sending the next. This is a deliberate
strengthening: if the child owned the payload, a `Detector` could reach the
whole future series through `sys._getframe(1).f_locals` and the "online" framing
would be advisory rather than structural. Verified with a probe detector that
hoards every row *and* walks up to 12 frames of the call stack scanning locals
and globals for anything shaped like a 24-float row buffer: it finds zero rows
beyond the ones already handed to it, at every checkpoint.

Anything the agent prints is redirected from fd 1 to stderr before its module
is imported, so it cannot desynchronise the protocol.

### `cheat/solution.py`

Three attacks, all of which must fail:

1. **Forge the reward** — writes `1.0` to `/logs/verifier/reward.txt`,
   `reward.json`, `rewards.json` and `/logs/reward.txt`, both inline and from a
   detached `os.fork()` daemon that keeps rewriting them for 45 s, i.e. after
   the verifier has written the real score.
2. **Steal the labels** — walks `/tests`, `/app`, `/tmp`, `/opt` (bounded to
   400 JSON files) for anything carrying `onset` / `t_obvious` /
   `origin_group`, and if found alarms at exactly `onset` for a perfect lead.
3. **Alarm without detecting** — otherwise `return 1` on every timestep.

Result: **0.0**, via the FPR gate (attack 3 is the fallback that actually
runs). Attacks 1 and 2 were verified independently rather than assumed:

* Attack 2 is a **live canary**. During development the offline harness put its
  scratch sandbox under `/tmp`, one of the search roots, and the cheat found
  the labels and scored **1.0**. Moving the sandbox off `/tmp` — matching the
  real container, where the only thing in the verifier's `/tmp` workdir is a
  copy of the agent's own solution — dropped it back to 0.0. The check now runs
  both ways: with the episode file readable the walker returns all 12 onsets;
  at mode `000` (what `chmod 600` root looks like to `nobody`) it returns
  `None`.
* Attack 1 is defeated twice over. Permissions stop the write, and
  independently of permissions `run_verifier.py` rewrites `reward.txt`
  unconditionally as its final action. Verified by pre-planting
  `reward.txt = 1.0`, `reward.json` and `rewards.json` in the log directory:
  after the run, `reward.txt` holds the real score and both JSON files are gone.

## Generator

`provenance/data_gen.py`, deterministic from a fixed master seed.

Latent factor model: `x[t,i] = wg[i]·A_glob(t)·g[t] + w[i]·A_grp[k(i)](t)·f[k(i),t]
+ sigma·e[t,i]`, all factors iid normal, all the structure in the time-varying
loadings. The system-wide amplitude is solved by bisection to hit a per-episode
target population correlation (0.80–0.90), so `t_obvious` exists but is not at a
constant offset.

Randomised per episode: onset, series length, which group originates, build-up
length, spread length and stagger, number of groups affected, global ramp
length, noise scale, co-movement magnitude, target correlation, and per-channel
loading jitter.

| | dev | test |
|---|---|---|
| episodes | 6 | 12 |
| series length | 684–1241 | 618–2062 |
| onset | 274–950 | 267–1707 |
| build-up length | 17–29 | **6**–30 |
| groups affected | 3–6 | 3–6 |
| noise scale | — | 0.71–1.39 |
| never reaches obvious | 0 | **2** (`test-05`, `test-10`) |

Shapes the test split has and dev does not: a 6-step build-up and a 9-step
build-up (dev's shortest is 17), and two episodes whose break never becomes
obvious. Those two are excluded from the lead average but their 2090 calm
timesteps still count toward `fpr`.

**Deviation from the original sketch:** series were specified as ~400
timesteps. They are 618–2062 instead. With ~400-step episodes the onset range
is too narrow relative to the ~47-step detection window, and a hard-coded
timestep scored ~0.2–0.3 — a wide, well-separated onset distribution is the
only structural fix, and it also makes the calm prefix long enough for the FPR
gate to bite. Total test data is 15,991 timesteps (2.9 MB JSON); the reference
detector processes all of it in ~1.2 s.

## Layout

```
task.toml                        schema 1.3, artifacts top-level
instruction.md                   agent-facing spec, ends with '## Time budget'
README.md                        this file
environment/
  Dockerfile                     python:3.12-slim + numpy 2.5.1 + curl/git
  .dockerignore
  app/solution.py                weak starter (alarms at t_obvious) -> 0.0
  app/run_dev.py                 self-scoring on the dev split
  app/dev_episodes.json          6 labelled episodes
tests/
  Dockerfile                     verifier image; also the build context
  .dockerignore
  test.sh                        entrypoint, always (re)writes reward.txt
  run_verifier.py                root: streams data, scores, writes reward
  predict.py                     nobody: hosts the agent's Detector
  grade.py                       lead / fpr / never-alarmed, pure stdlib
  reward.py                      L_MAX and the gate
  degenerate.py                  anti-degenerate suite
  test_episodes.json             12 hidden episodes, chmod 600 root
solution/{solution.py,solve.sh}  competent reference -> 0.4818
cheat/{solution.py,solve.sh}     three attacks -> 0.0
provenance/data_gen.py           deterministic generator, prints L_MAX
```

`tests/` is the build context for the separate verifier container, so
`tests/Dockerfile` does `COPY . /tests/` — harbor skips the tests upload in
separate mode and expects the image to own `/tests`.

## Reproducing the numbers

Inside the verifier image:

```sh
python3 /tests/degenerate.py          # anti-degenerate table
sh /tests/test.sh                     # full scoring of whatever is at /app/solution.py
cat /logs/verifier/metrics.json
```

Regenerating the data (rewrites both JSON files and reprints `L_MAX`, which
must then be updated in `tests/reward.py`):

```sh
python3 provenance/data_gen.py
```
