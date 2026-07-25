#!/usr/bin/env python3
"""Collect Harbor trial results into results/<model_effort>/<task>/.

    .venv/bin/python collect_results.py

Rebuilds results/ from scratch on every run, so re-running it after a new job
lands is always correct. Nothing here is hand-maintained: the numbers come from
the trial's own result.json, and the label comes from its lock.json.

Two things are less obvious than they look:

  * Jobs live under more than one root. Harbor's -o defaults to the *relative*
    path "jobs", so a run launched from a different cwd lands somewhere else --
    which happened once here, putting a whole codex job in /data/projects/jobs.
    run_runpod.sh now pins -o, but the stray job is still real data.
  * The reasoning effort is usually absent from lock.json. Harbor only records
    kwargs the caller passed, and `high` for codex comes from the agent class's
    own CliFlag default. Reading that default out of Harbor at collection time
    keeps the label honest if Harbor ever changes it.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
JOB_ROOTS = [REPO / "jobs", Path("/data/projects/jobs")]

# Runs that measured a misconfiguration rather than a model. Dropped entirely,
# because a row reading "gpt-5-codex: 0.000" is worse than no row at all -- it
# looks like a model result and is not one. The underlying job dirs are
# untouched; only this view excludes them.
EXCLUDED_LABELS: dict[str, str] = {
    "gpt-5-codex_high": (
        "model rejected server-side: 'The gpt-5-codex model is not supported "
        "when using Codex with a ChatGPT account'. The agent never ran a turn."
    ),
    "kimi-cli": (
        "launched without --model, so every trial died in agent construction "
        "with ValueError before reaching the environment."
    ),
}

# Failures of the harness or the provider, never of the model. A pod that never
# became reachable says nothing about the agent that was going to run on it, and
# leaving these in reads as an agent failure in the `!k` column. They stay in
# jobs/ -- they are real operational events worth keeping -- but they are not
# measurements and do not belong in a results table.
INFRA_EXCEPTIONS = {
    # RunPod never made the pod SSH-reachable inside the build timeout.
    "EnvironmentStartTimeoutError",
    # The SSH control channel died. Every occurrence so far was the `kill 0`
    # sentinel bug in this repo's own backend, fixed by the setsid patch.
    "RunPodSSHError",
}

# Populated by main(); read by write_summary() to explain what is missing.
DROPPED: list[dict] = []


def agent_flag_default(agent_name: str, flag: str) -> str | None:
    """The default Harbor would apply for *flag*, or None if not applicable."""
    try:
        from harbor.agents.factory import AgentFactory
        from harbor.models.agent.name import AgentName
        from harbor.utils.import_path import import_class

        import_path = AgentFactory._AGENT_MAP.get(AgentName(agent_name))
        if not import_path:
            return None
        cls = import_class(import_path, label="agent")
        for cli_flag in getattr(cls, "CLI_FLAGS", []):
            # The field is `kwarg`, not `name` -- CliFlag is keyed by the
            # --agent-kwarg it accepts.
            if cli_flag.kwarg == flag:
                return cli_flag.default
    except Exception:
        return None
    return None


def label_for(lock: dict, task_path: str) -> str:
    """Directory name for this trial: <model>_<effort>, or the agent name."""
    agent = lock.get("agent") or {}
    name = agent.get("name") or "unknown"

    # The cheat baseline is the same oracle agent pointed at a task copy whose
    # solution/ was swapped for cheat/. Only the path distinguishes them.
    if "cheat" in task_path:
        return "cheat"
    if name in ("oracle", "nop"):
        return name

    model = (agent.get("model_name") or name).split("/")[-1]
    effort = (agent.get("kwargs") or {}).get("reasoning_effort")
    if effort is None:
        effort = agent_flag_default(name, "reasoning_effort")
    return f"{model}_{effort}" if effort else model


def read_trial(trial_dir: Path) -> dict | None:
    result_path = trial_dir / "result.json"
    lock_path = trial_dir / "lock.json"
    if not result_path.is_file() or not lock_path.is_file():
        return None

    result = json.loads(result_path.read_text())
    lock = json.loads(lock_path.read_text())
    task = lock.get("task") or {}
    agent = lock.get("agent") or {}

    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if reward is None:
        # A trial whose verifier never ran has no reward at all. That is NOT the
        # same as scoring zero, and the two must not collapse into one number.
        text = trial_dir / "verifier" / "reward.txt"
        if text.is_file():
            try:
                reward = float(text.read_text().strip())
            except ValueError:
                reward = None

    execution = result.get("agent_execution") or {}
    started, finished = execution.get("started_at"), execution.get("finished_at")
    seconds = None
    if started and finished:
        seconds = round(
            (
                dt.datetime.fromisoformat(finished) - dt.datetime.fromisoformat(started)
            ).total_seconds()
        )

    return {
        "task": task.get("name") or trial_dir.name.split("__")[0],
        "label": label_for(lock, str(task.get("path") or "")),
        "reward": reward,
        "agent": agent.get("name"),
        "model": agent.get("model_name"),
        "exception": (result.get("exception_info") or {}).get("exception_type"),
        "agent_seconds": seconds,
        "backend": (lock.get("environment") or {}).get("import_path") or "docker",
        "task_digest": task.get("digest"),
        "trial_dir": str(trial_dir),
        "trial_id": trial_dir.name.split("__")[-1],
        "job": trial_dir.parent.name,
        "verifier_ran": (trial_dir / "verifier" / "reward.txt").is_file(),
    }


def live_tasks() -> set[str]:
    """Task names that still exist on disk.

    Older jobs cover tasks this repo no longer has (crisis-early-detection and
    friends). Their numbers are real but orphaned -- nothing can be re-run or
    compared against them -- so they are left out rather than padding the table
    with rows whose task definition is gone.
    """
    tasks_dir = REPO / "tasks"
    return {d.name for d in tasks_dir.iterdir() if (d / "task.toml").is_file()}


def collect() -> tuple[list[dict], list[dict]]:
    """Return (kept, dropped) trial records."""
    tasks = live_tasks()
    kept: list[dict] = []
    dropped: list[dict] = []
    for root in JOB_ROOTS:
        if not root.is_dir():
            continue
        for job_dir in sorted(root.iterdir()):
            if not job_dir.is_dir():
                continue
            for trial_dir in sorted(job_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                record = read_trial(trial_dir)
                if not record:
                    continue
                if record["exception"] in INFRA_EXCEPTIONS:
                    record["dropped_because"] = (
                        f"{record['exception']} is a harness/provider failure, "
                        "not a model outcome"
                    )
                    dropped.append(record)
                elif record["label"] in EXCLUDED_LABELS:
                    record["dropped_because"] = EXCLUDED_LABELS[record["label"]]
                    dropped.append(record)
                elif record["task"] not in tasks:
                    record["dropped_because"] = "task no longer exists in tasks/"
                    dropped.append(record)
                else:
                    kept.append(record)
    return kept, dropped


def write_one(record: dict) -> None:
    # One directory per trial, named by Harbor's trial suffix. Writing straight
    # into <label>/<task>/ silently overwrote repeats, which is exactly the
    # data -k 3 exists to produce.
    out = RESULTS / record["label"] / record["task"] / record["trial_id"]
    out.mkdir(parents=True, exist_ok=True)
    src = Path(record["trial_dir"])

    (out / "meta.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["reward"] is not None:
        (out / "reward.txt").write_text(f"{record['reward']:.6f}\n")

    for src_rel, dst_name in (
        ("verifier/metrics.json", "metrics.json"),
        ("verifier/test-stdout.txt", "verifier-stdout.txt"),
        ("artifacts/app/solution.py", "solution.py"),
    ):
        candidate = src / src_rel
        if candidate.is_file():
            shutil.copy2(candidate, out / dst_name)


def write_summary(trials: list[dict]) -> None:
    labels = sorted({t["label"] for t in trials})
    tasks = sorted({t["task"] for t in trials})

    # Every repeat is kept. Collapsing to one number per cell would hide both
    # the spread that -k 3 was run to measure and the failures that Harbor's
    # --max-retries quietly replaced with a successful attempt.
    groups: dict[tuple[str, str], list[dict]] = {}
    for t in trials:
        groups.setdefault((t["label"], t["task"]), []).append(t)

    # Averaging across task revisions is silently wrong: budget-triage's oracle
    # scored 0.000 before its verifier's SystemExit bug was fixed and 0.496
    # after, and a mean of 0.330 describes nothing that exists. Harbor records
    # a content digest per trial, so keep only the newest revision.
    #
    # Filtered per (label, task) rather than per task, because the cheat
    # baseline is deliberately a modified copy of the task and therefore always
    # carries a different digest -- a global filter would delete that column.
    for key, runs in list(groups.items()):
        runs.sort(key=lambda r: (r["job"], r["trial_id"]))
        current = runs[-1]["task_digest"]
        fresh = [r for r in runs if r["task_digest"] == current]
        for stale in (r for r in runs if r["task_digest"] != current):
            stale["dropped_because"] = (
                f"ran against an older revision of {stale['task']} "
                f"({(stale['task_digest'] or '?')[:19]}, current is "
                f"{(current or '?')[:19]})"
            )
            DROPPED.append(stale)
        groups[key] = fresh

    for (label, task), runs in groups.items():
        scored = [r["reward"] for r in runs if r["reward"] is not None]
        aggregate = {
            "label": label,
            "task": task,
            "n_trials": len(runs),
            "n_scored": len(scored),
            "rewards": scored,
            "mean": (sum(scored) / len(scored)) if scored else None,
            "min": min(scored) if scored else None,
            "max": max(scored) if scored else None,
            "failures": [r["exception"] for r in runs if r["exception"]],
        }
        path = RESULTS / label / task / "aggregate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2) + "\n")

    with (RESULTS / "summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["label", "task", "trial_id", "reward", "exception", "agent_seconds", "job"]
        )
        # One row per trial, not per cell: the CSV is the raw record, and any
        # averaging belongs to whoever reads it.
        for label in labels:
            for task in tasks:
                for t in groups.get((label, task), []):
                    writer.writerow(
                        [
                            label,
                            task,
                            t["trial_id"],
                            t["reward"],
                            t["exception"] or "",
                            t["agent_seconds"] or "",
                            t["job"],
                        ]
                    )

    lines = ["# Results", ""]
    lines.append("Reward is normalised: 0 = random baseline, 1 = provable optimum")
    lines.append("computed from the hidden labels (unreachable by construction).")
    lines.append("")
    lines.append("`--` means no trial. `n/a` means the trial produced no reward at")
    lines.append("all because its verifier never ran -- distinct from scoring 0.000.")
    lines.append("")
    lines.append("Cells with repeats show `mean [min-max] n=k`. A `!k` suffix")
    lines.append("counts trials that raised, including ones a retry later replaced.")
    lines.append("")
    lines.append("| task | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for task in tasks:
        cells = []
        for label in labels:
            runs = groups.get((label, task))
            if not runs:
                cells.append("--")
                continue
            scored = [r["reward"] for r in runs if r["reward"] is not None]
            failures = [r for r in runs if r["exception"]]
            if not scored:
                cells.append(f"n/a ({runs[-1]['exception']})")
                continue
            if len(scored) == 1:
                cell = f"{scored[0]:.3f}"
            else:
                mean = sum(scored) / len(scored)
                cell = f"{mean:.3f} [{min(scored):.3f}-{max(scored):.3f}] n={len(scored)}"
            if failures:
                cell += f" !{len(failures)}"
            cells.append(cell)
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    # Anything that produced no number at all is listed rather than silently
    # missing: an absent row and a failed run look identical in a table, and
    # only one of them means "not tried yet".
    unmeasured = [t for t in trials if t["reward"] is None]
    if unmeasured:
        lines += ["", "## Trials with no measurement", ""]
        lines.append("The verifier never ran, so there is no score -- not a zero.")
        lines.append("")
        for t in sorted(unmeasured, key=lambda x: (x["label"], x["task"])):
            lines.append(
                f"- `{t['label']}` / `{t['task']}` ({t['trial_id']}): {t['exception']}"
            )

    if DROPPED:
        lines += ["", "## Excluded from the table", ""]
        lines.append("Present in jobs/ but not collected here, with the reason.")
        lines.append("")
        seen: set[tuple[str, str]] = set()
        for t in DROPPED:
            key = (t["label"], t["dropped_because"])
            if key in seen:
                continue
            seen.add(key)
            count = sum(
                1
                for d in DROPPED
                if d["label"] == t["label"]
                and d["dropped_because"] == t["dropped_because"]
            )
            lines.append(f"- `{t['label']}` ({count} trials): {t['dropped_because']}")

    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    global DROPPED
    trials, DROPPED = collect()
    if not trials:
        print("No trials found.", file=sys.stderr)
        return 1

    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir()

    for record in trials:
        write_one(record)
    write_summary(trials)

    labels = sorted({t["label"] for t in trials})
    print(f"{len(trials)} trials -> {RESULTS}")
    for label in labels:
        rows = [t for t in trials if t["label"] == label]
        scored = [t for t in rows if t["reward"] is not None]
        print(f"  {label:24} {len(rows)} trials, {len(scored)} scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
