# budget-triage

Rank alerts under a fixed investigation budget (k=20 per period). Reward is
normalised between a random ordering and a provable optimum computed from the
hidden labels.

| | n=4 | mean | reference solution |
|---|---|---|---|
| codex gpt-5.6-sol | 0.510, 0.511, 0.637, 0.643 | 0.575 | 0.496 |
| grok-4.5 | 0.531, 0.553, 0.553, 0.588 | 0.556 | |
| kimi k3-256k | 0.490, 0.497, 0.543, 0.576 | 0.527 | |

## The three agents are not separable here

The intervals overlap almost completely: 0.510-0.643, 0.531-0.588, 0.490-0.576.
Any ranking drawn from this cell is an artifact of which sample you looked at.
It is the least discriminating of the four tasks.

## codex is bimodal, and the mechanism is in the code

Two runs cluster at ~0.51 and two at ~0.64, with nothing between. Comparing the
four submitted `solution.py` files:

| reward | lines | `erf` | `cap` |
|---|---|---|---|
| 0.510 | 82 | no | no |
| 0.511 | 70 | no | no |
| 0.637 | 103 | yes | yes |
| 0.643 | 104 | yes | yes |

The high runs compute a **truncated** log-normal severity -- `E[min(exp(Z),
cap)]` with the cap at the observed maximum, using `erf` for the normal CDF. The
low runs use an untruncated estimate. That single modelling choice is worth
~0.13 of reward, and it is present or absent, never partial.

One high run additionally marginalised over an unobservable period-to-period
rate shift with 5-point Gauss-Hermite quadrature; the other did not and scored
the same. So the quadrature is not what separates the clusters -- the cap is.

## Why beating the reference solution is expected

The reference (0.496) says so itself:

> Competent, not optimal. It exists to prove the task is solvable and to give a
> sane comparison point; it deliberately does NOT define full marks. [...] this
> scores roughly half way from random to it.

It fits logistic regression for `p_hat` and ridge for `sev_hat`, then ranks by
expected utility -- with the same naive log-normal debias the low-scoring agent
runs use. Its 0.496 is a designed midpoint, not a bar.

## The reachable ceiling is 0.671, not 1.0

`bayes_optimal_budget_triage.py` replays the generator to recover every
per-period parameter, then ranks by the exact expected utility
`p * E[clip(severity)] * decay(age) - (1-p) * COST`. That is what an agent that
knew the generative process perfectly -- but not which alerts turned out to be
incidents -- would submit. The replay is checked alert-by-alert against
`tests/test_alerts.json` (6206/6206 identical) before the number is reported.

| | reward | share of reachable |
|---|---|---|
| **Bayes-optimal** | **0.6706** | 100% |
| codex, best run | 0.643 | **95.9%** |
| codex, mean | 0.575 | 85.7% |
| grok, mean | 0.556 | 82.9% |
| k3, mean | 0.527 | 78.6% |
| reference solution | 0.496 | 74.0% |

Reward 1.0 corresponds to the optimum computed *from the labels*; the remaining
33% is irreducible. Knowing each alert's incident probability exactly still
leaves you unable to know which ones fired.

Two things follow.

**The task is not short of difficulty, it is short of headroom.** The three
agents look inseparable here because they are all crowded into 79-86% of a
ceiling that codex's best run reaches 96% of. There is roughly 0.03 of reward
left to win, which is smaller than codex's own run-to-run spread.

**Any statement of the form "scored X% of optimal" should use 0.671.** Saying
codex reached "64% of optimal" is arithmetically true against the label-based
ceiling and badly misleading: almost all of the missing 36% cannot be obtained
by anyone.
