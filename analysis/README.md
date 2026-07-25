# Per-task analysis

Written by hand from the n=4 run of 2026-07-25 (codex `gpt-5.6-sol`,
`grok-4.5`, kimi `k3-256k`). The numbers live in `../results/`, which
`collect_results.py` deletes and rebuilds on every run -- these notes live here
so they survive that.

What the tables cannot show is why a cell has the spread it has, and the spread
turned out to be the most informative thing in the whole run. Three of the four
tasks are **bimodal**: repeats do not scatter around a mean, they land in one of
two clusters, and which cluster depends on a single decision the agent either
makes or does not.

Read `variance.md` first; the per-task files assume it.

| file | task | what separates a good run from a bad one |
|---|---|---|
| `budget-triage.md` | budget-triage | whether the agent caps the severity distribution |
| `changepoint-restraint.md` | changepoint-restraint | threshold calibration; method varies every run |
| `drift-vs-anomaly.md` | drift-vs-anomaly | nothing discrete -- the only task with continuous spread |
| `early-warning-budget.md` | early-warning-budget | whether the agent checks its own false-positive rate |

## A caveat that applies to every file here

n=4. Every "always" and "never" below is "in four runs", and this session
produced three separate occasions where a claim that looked systematic at n=3
was refuted by the fourth sample. Treat cluster *membership* as observed and
cluster *probability* as barely estimated.
