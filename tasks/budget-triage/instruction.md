# Budget triage

An anomaly-detection system is firing far more alerts than the on-call team can
investigate. You have a fixed investigation budget: **k = 20 alerts per period**,
no matter how many alerts arrive. Your job is to decide what gets looked at.

This is a prioritisation and calibration problem, not a detection problem. You
are not asked to find every incident; you are asked to spend a small, fixed
capacity as well as it can possibly be spent.

## What you write

Edit **`/app/solution.py`** so that it defines:

```python
def rank(alerts) -> list[int]:
    """alerts: list of dicts with keys 'id', 'features' (list[float]),
    'age_days' (float), and 'period' (int). Return alert ids, best first.
    Ranking is per period; the verifier will take your ordering restricted to
    each period and investigate the top k of it."""
```

`rank` is called **once**, with every alert from every period in a single list.
Return a single ordering over all of them. The verifier restricts that ordering
to one period at a time and investigates the first `k = 20` ids it finds there,
so a global sort by "how good is this alert" behaves exactly like sorting inside
each period.

`is_incident` and `severity` are **never** visible to `rank`. The only keys in
the dicts you receive are `id`, `features`, `age_days` and `period`.

## How you are scored

Every alert has a hidden `is_incident` (bool) and a hidden `severity`
(float > 0). Investigating alert *a* is worth:

```
u(a) = severity(a) * max(0, 1 - age_days(a) / T_MAX)   if is_incident(a)
     = -COST                                            otherwise
```

with **`T_MAX = 30.0`** and **`COST = 0.1`**.

Read that carefully, because it is the whole task:

* A **true incident caught early** is worth a lot. The same incident caught at
  20 days is worth a third as much, and at 30 days or older it is worth exactly
  nothing — the decay factor floors at zero.
* A **false alert** is worth `-0.1`. Small, but not zero: it burnt one of your
  20 slots and it drags your score down. A ranker that is blind to precision
  gets punished for it.
* So the thing worth ranking by is neither "most likely to be an incident" nor
  "freshest". It is the **expected utility** of spending a slot:
  `p(incident) * severity * decay - (1 - p(incident)) * COST`. You have to
  estimate the first two and combine them with the third.

Per period, your score is `u` summed over your top 20. Then:

```
achieved = Σ over periods of Σ u(top-k of your order)
optimal  = Σ over periods of Σ u(top-k under the TRUE utility order)
floor    = Σ over periods of (k × mean u over every alert in the period)

reward   = clamp((achieved - floor) / (optimal - floor), 0.0, 1.0)
```

* `optimal` is computed from the hidden labels by sorting on `u` and taking the
  top k. Because your period score depends only on *which* 20 alerts you
  surface, that is a **provable optimum** — the best any ordering could do, not
  the score of some reference implementation. Nobody's code sets the ceiling.
* `floor` is the **analytic** expected total of a uniformly random order — the
  period mean times k, computed rather than sampled.
* So **0.0 means you did no better than triaging at random** and **1.0 means you
  triaged provably optimally**. Both ends are real.

Full marks are genuinely hard: the features are informative but noisy, so a
perfectly calibrated model still cannot know which alerts are incidents. Getting
meaningfully above zero is very achievable.

## Hard gate: your ordering must be a permutation

The list you return must be a permutation of **exactly** the alert ids you were
given:

* every id present,
* each id **exactly once**,
* **no** ids that were not in the input.

If the ordering fails this check, that period scores **0.0** — and because the
ordering is checked globally, a failure on any period means **the whole task
scores 0.0**. Length mismatches, duplicates, dropped ids and invented ids are
all failures. Do not filter the list down to your top 20; return all of them,
ordered.

## What you get to work with

* **`/app/dev_alerts.json`** — a development split of ~6 periods **with the
  hidden fields exposed** (`is_incident` and `severity` are both present). Use
  it to fit whatever you like.
* **`/app/run_dev.py`** — scores a solution against the dev split with exactly
  the arithmetic above. Run `python3 /app/run_dev.py` for your current
  `/app/solution.py`, or add `--baselines` to see the oracle (1.0) and the
  starter (~0.0) for comparison. It prints achieved, optimal, floor, the
  normalised score, precision@k, and the mean age of the incidents you caught.

The graded split is different data: **12 periods**, drawn from the same
generating distribution but different seeds, and never visible to you. It
includes at least one period with an unusually low incident rate and one where
the alerts are unusually old. A solution tuned to the exact composition of the
dev split will not transfer; one that estimates probability, severity and decay
will.

Python 3.12 with **numpy** is installed. The standard library and numpy are all
you need and all you get — there is no package index to reach and no GPU.

## Two practical constraints

**Your file must be self-contained.** At scoring time the container has
`/app/solution.py` and nothing else from your working directory —
`/app/dev_alerts.json` is *not* there. If you fit a model, bake the coefficients
into `solution.py` as literals. A `rank()` that tries to open the dev file at
scoring time will fail and score 0.0.

**Randomness is pinned for you.** The harness seeds `random` and `numpy.random`
to 0 before your code is imported, so the same `solution.py` always earns the
same reward. Do not rely on entropy for behaviour.

Your `rank()` runs as an unprivileged user with a few minutes of wall clock.
Keep it to a straightforward pass over the alerts; there is no need for anything
expensive.

## Time budget

You have **60 minutes**. There is no warning before the deadline: when the time
is up, `/app/solution.py` is collected exactly as it stands at that moment and
scored. Keep the file in a valid, submittable state the whole way through —
write a working ordering first, then improve it — rather than leaving a
half-finished edit that cannot be imported.
