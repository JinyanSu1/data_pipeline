# early-warning-budget

Detect a regime break as early as possible while alarming on at most 5% of calm
timesteps. The reward rule is a hard gate:

```python
FPR_LIMIT = 0.05
if fpr > FPR_LIMIT:
    return 0.0                       # no partial credit
return clamp(mean_lead_steps / 46.7, 0.0, 1.0)
```

**The rule is printed verbatim in `instruction.md`**, together with the warning
that "in practice you want to be *far* below the gate", and `/app/run_dev.py`
scores a candidate detector on labelled dev episodes and prints `fpr` and
whether the gate passes. Nothing about the constraint is hidden or inferable
only from the hidden test set.

## Results

| | reward | fpr | mean_lead |
|---|---|---|---|
| codex | 0.734 | 0.00000 | 34.3 |
| codex | 0.734 | 0.00000 | 34.3 |
| codex | 0.739 | 0.00000 | 34.5 |
| codex | 0.739 | 0.00000 | 34.5 |
| k3 | 0.651 - 0.739 | 0.00000 (all 4) | 30.5 - 34.5 |
| **grok** | **0.000** | **0.399** | 11.5 |
| **grok** | **0.000** | **0.360** | 10.8 |
| **grok** | **0.000** | **0.177** | 22.1 |
| grok | 0.653 | 0.00000 | 30.5 |

## The gate is the whole task, and grok misses it 3 times in 4

codex and k3 land on `fpr = 0.00000` in every single run -- exactly zero, not
merely under the limit. That is the signature of iterating against `run_dev.py`
until the gate is clear.

grok's three failures are 3.5x, 7.2x and 8x over a limit it was told in code.
Its one success is `fpr = 0.00000` and `mean_lead = 30.5` -- the same regime as
the other two agents. So the capability is there; the checking is what varies.

## It is not a precision/earliness tradeoff

The obvious reading -- grok alarms aggressively to win lead time -- is wrong.
Its failing runs have **both** worse fpr and worse lead (10.8-22.1 against
codex's 34.5). Lifting the gate entirely would still leave them at ~0.23-0.47
against codex's 0.74. Alarming more did not buy earliness; the alarms are just
noise.

## What this task measures

Not detection skill -- all three agents can build a detector. It measures
whether an agent **verifies its output against a stated, locally checkable
constraint before submitting**. That is a different axis from raw capability,
and this task isolates it better than the other three.

Compare with `drift-vs-anomaly`, which has a nominally similar gate
(`drift_fpr_limit = 0.1`) that no run came within 25x of touching. A gate only
measures restraint when it binds.
