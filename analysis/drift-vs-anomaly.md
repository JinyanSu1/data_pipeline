# drift-vs-anomaly

Distinguish gradual drift from point anomalies in a stream. Reward equals
`event_f1` once a drift false-positive gate passes.

| | n=4 | mean | reference solution |
|---|---|---|---|
| codex gpt-5.6-sol | 0.766, 0.776, 0.809, 0.822 | 0.793 | 0.756 |
| grok-4.5 | 0.673, 0.737, 0.742, 0.764 | 0.729 | |
| kimi k3-256k | 0.638, 0.658, 0.736, 0.796 | 0.707 | |

## The one task with continuous spread

No clustering in any of the three agents -- values scatter across their range
rather than landing in two groups. This is the only cell block where the mean
summarises the distribution honestly, and the only place a mean-based ranking is
defensible.

It is also the only near-separation in the whole run: codex's interval
[0.766-0.822] and grok's [0.673-0.764] do not overlap. Barely -- 0.766 against
0.764, a margin of 0.002 at n=4. Worth re-running before leaning on it.

## The gate never binds

Every one of the twelve runs passed, and none came close:

| agent | drift_fpr range | limit |
|---|---|---|
| codex | 0.00047 - 0.00117 | 0.1 |
| grok | 0.00082 - 0.00328 | 0.1 |
| k3 | 0.00023 - 0.00387 | 0.1 |

The tightest run is 25x under the limit; the loosest is over 250x under. So
despite having the same gate-plus-F1 shape as `early-warning-budget`, this task
does not measure restraint at all -- `reward == event_f1` exactly, in all twelve
runs.

That makes it a useful control. When an agent scores 0 on
`early-warning-budget` but 0.7 here, the difference is not detection ability.

## Alarm count does not predict score here

k3's best run (0.796) raised **33** drift alarm steps -- the most of any run in
the block -- while its 0.736 run raised 2. With the gate this far from binding,
being conservative buys nothing, and the ordering of scores is unrelated to
alarm volume. The opposite of the other three tasks, where selectivity is
exactly what separates good runs from bad.
