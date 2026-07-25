# Where the variance comes from

n=4 per (agent, task), 2026-07-25. Raw values, sorted:

| task | codex gpt-5.6-sol | grok-4.5 | kimi k3-256k |
|---|---|---|---|
| budget-triage | 0.510, 0.511, **0.637, 0.643** | 0.531, 0.553, 0.553, 0.588 | 0.490, 0.497, 0.543, 0.576 |
| changepoint-restraint | 0.458, 0.504, **0.667, 0.685** | 0.000, 0.450, 0.544, 0.607 | 0.000, 0.000, **0.542, 0.696** |
| drift-vs-anomaly | 0.766, 0.776, 0.809, 0.822 | 0.673, 0.737, 0.742, 0.764 | 0.638, 0.658, 0.736, 0.796 |
| early-warning-budget | 0.734, 0.734, 0.739, 0.739 | **0.000, 0.000, 0.000, 0.653** | 0.651, 0.655, 0.696, 0.739 |

## The means are the least useful column

codex on budget-triage averages 0.575. No run scored anything near 0.575. Two
scored ~0.51 and two scored ~0.64, and the gap between the clusters is larger
than the spread inside either. The mean names a value the agent never produces
and a solution that does not exist.

The same holds for codex on changepoint-restraint (0.458/0.504 against
0.667/0.685), for grok on early-warning-budget (three zeros and a 0.653), and
for k3 on changepoint-restraint (two zeros and 0.542/0.696).

Where a mean *is* meaningful: drift-vs-anomaly, the one task whose repeats
scatter rather than cluster, for all three agents.

## Why bimodal

Each of the three bimodal tasks has one decision that dominates the score:

* **budget-triage** -- model the severity cap, or do not. Verified in code: the
  two 0.64 solutions use `erf`/`cap` to compute a truncated log-normal mean, and
  the two 0.51 solutions contain neither. 103-104 lines against 70-82.
* **early-warning-budget** -- a hard gate at `fpr > 0.05` zeroes the reward. An
  agent either respects it or misses it by 3.5-8x. grok's one passing run scored
  `fpr=0.00000`; its three failures scored 0.177, 0.360, 0.399.
* **changepoint-restraint** -- a null-series gate plus a precision/recall
  tradeoff. k3 and grok fail the gate outright on some runs; codex never does,
  and its variance is a milder version of the same thing (false positives 9-14
  on good runs against 27-34 on bad ones).

So the observed spread is not execution noise. It is a discrete
discovered/not-discovered event, sampled four times.

## What this means for the benchmark

**A single run of these tasks is close to a coin flip.** Reporting one number
per (agent, task) will produce rankings that reverse on re-run. This is not
hypothetical -- it happened three times while the data was being collected:

1. "kimi fails every task" -- from a batch where 3 of 4 trials never finished
   for reasons unrelated to kimi.
2. "grok fails changepoint-restraint systematically" -- n=1. At n=4 it scores
   0.450, 0.544, 0.607 and one crash.
3. "grok fails early-warning-budget systematically, zero-width interval" -- n=3,
   all zeros. The fourth run scored 0.653.

**n=4 estimates membership, not probability.** That grok passed the
early-warning gate once in four tells us the passing mode exists. It does not
distinguish a 25% mode from a 10% one, and the difference matters if these
numbers are meant to rank anything.

**Report the cluster, not the mean.** For a bimodal cell, `0.51 or 0.64,
2/4 each` carries the finding; `0.575` hides it.
