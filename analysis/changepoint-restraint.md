# changepoint-restraint

Locate an unknown number of change points, with a gate on false alarms in
null series (no change point at all). Reward equals the event F1 once the gate
passes.

| | n=4 | mean | reference solution |
|---|---|---|---|
| codex gpt-5.6-sol | 0.458, 0.504, 0.667, 0.685 | 0.578 | **0.650** |
| grok-4.5 | 0.000, 0.450, 0.544, 0.607 | 0.400 | |
| kimi k3-256k | 0.000, 0.000, 0.542, 0.696 | 0.309 | |

The only task where the reference solution beats every agent's mean, and the
only one where an agent's best run (k3's 0.696, codex's 0.685) is needed to
approach it.

## Two different failure modes, easy to confuse in the table

Both k3 and grok have zeros, and they mean different things:

* **k3's zeros are the gate.** `null_fp_rate = 3.0` against a threshold of 0.5
  -- it alarmed on null series repeatedly. The solution ran, produced an
  ordering, and was scored at zero for lack of restraint.
* **grok's zero is a crash.** `NonZeroAgentExitCodeError`, exit 143 = SIGTERM,
  867s into a 3600s budget. Root cause below.
* **codex never fails the gate** -- `null_fp_rate = 0.0` and
  `null_gate_passed = True` in all four runs.

## codex's variance is calibration, not a missing insight

Unlike budget-triage, there is no single feature present in the good runs and
absent in the bad ones. Each run picks a different method:

| reward | method markers | fn | fp | mean_abs_offset |
|---|---|---|---|---|
| 0.458 | bootstrap | 37 | 27 | 1.44 |
| 0.504 | BIC + penalty | 31 | 34 | 1.42 |
| 0.667 | CUSUM | 25 | 14 | 1.69 |
| 0.685 | penalty | 26 | 9 | 1.68 |

The good runs are not better at finding change points -- false negatives barely
move (37/31 against 25/26), and their localisation is slightly *worse* (1.69
against 1.44). They are 2-3x more selective: 9-14 false positives against
27-34.

So the spread here comes from how conservatively the detection threshold ends up
tuned, and codex re-derives that threshold from a different algorithm family
every run. This variance is less likely to shrink with more samples than
budget-triage's, because its source is method diversity rather than a single
binary discovery.

## grok's crash: the agent killed itself

The last tool call before death, from `chat_history.jsonl`:

```sh
pkill -f "run_dev" 2>/dev/null; pkill -f "solution" 2>/dev/null; sleep 0.3; true
```

described as "Kill slow PELT evaluation". `pkill -f` matches the **full command
line**, and Harbor launches the agent as

```
grok --no-auto-update --single '<entire instruction text>' ...
```

The instruction for this task contains the word "solution" five times, so the
grok process's own cmdline matched its own pattern. SIGTERM, exit 143, five
seconds after the call started; the terminal log for that call is empty because
nothing survived to write it.

This is the agent's own command, not a harness defect -- but passing the
instruction through argv is what made a plausible cleanup command lethal. Any
`pkill -f <word that appears in the task text>` self-destructs, on any backend.
Routing the instruction through stdin or a file would remove the whole class.
Left unchanged for now: it is a real behaviour worth measuring, and changing it
would invalidate comparison with the runs already collected.
