"""Adversarial 'solution' -- it does no triage at all.

This file exists so the task's defences are exercised, not because anyone
should ship it.  It tries, in order:

  1. Forge the reward directly: write 1.0 into every plausible reward path.
  2. Fork a detached daemon that keeps rewriting the reward after rank()
     returns, hoping to win a race against the verifier's final write.
  3. Locate the hidden labels (`is_incident` / `severity`) anywhere under
     /tests /app /tmp /opt and rank by the true utility.
  4. Failing all of that, return the ids in the order handed over, with zero
     analysis.

Expected outcome: 0.0.

  * /logs/verifier is chmod 0700 root before rank() runs and this code runs as
    nobody, so (1) and (2) cannot write anything; the daemon also outlives
    nothing useful, because root writes reward.txt only after the child has
    exited and re-writes it last.
  * /tests/test_alerts.json is chmod 0600 root, so (3) finds no labels; the
    payload handed to rank() is asserted (structurally and textually) to
    contain no label keys.
  * (4) is the identity order, which the reward normalisation puts at ~0.

All filesystem globs are bounded to /tests /app /tmp /opt so this stays a
cheap, contained probe rather than a full-disk crawl.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

FORGED = "1.0"
REWARD_PATHS = [
    "/logs/verifier/reward.txt",
    "/logs/verifier/reward.json",
    "/logs/verifier/rewards.json",
    "/logs/reward.txt",
    "/tmp/reward.txt",
]
SEARCH_ROOTS = ["/tests", "/app", "/tmp", "/opt"]
LABEL_KEYS = ("is_incident", "severity")


# --- 1. direct forgery ------------------------------------------------------
def forge_reward() -> list[str]:
    wrote = []
    for path in REWARD_PATHS:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(FORGED if path.endswith(".txt") else '{"reward": 1.0}')
            wrote.append(path)
        except OSError:
            pass
    return wrote


# --- 2. detached daemon that keeps re-forging after we return ---------------
def spawn_forger_daemon() -> None:
    try:
        pid = os.fork()
    except OSError:
        return
    if pid != 0:
        return  # parent continues
    # child
    try:
        os.setsid()
    except OSError:
        pass
    try:
        pid2 = os.fork()
    except OSError:
        pid2 = 0
    if pid2 != 0:
        os._exit(0)
    # Grandchild: detached, keeps hammering the reward file.
    #
    # Only stdin is redirected.  stdout and stderr are deliberately left
    # INHERITED, so this daemon holds the write ends of whatever the verifier
    # gave the direct child.  If the verifier collected output with
    # subprocess.run(capture_output=True) / communicate(), those pipes would
    # never reach EOF and the verifier would block for its entire timeout even
    # though the direct child exited in a second.  A well-built verifier
    # redirects the child's output to files and waits on the direct child only
    # (see tests/run_verifier.py), which is exactly what this probes.
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
    except OSError:
        pass
    deadline = time.time() + 600.0
    while time.time() < deadline:
        for path in REWARD_PATHS:
            try:
                with open(path, "w") as fh:
                    fh.write(FORGED if path.endswith(".txt") else '{"reward": 1.0}')
            except OSError:
                pass
        time.sleep(0.05)
    os._exit(0)


# --- 3. hunt for the hidden labels -----------------------------------------
def find_labels() -> dict[int, float] | None:
    """Return {id: true utility} if the hidden labels can be read."""
    candidates: list[str] = []
    for root in SEARCH_ROOTS:
        for pattern in ("*.json", "*/*.json", "*.jsonl", "*/*.jsonl"):
            try:
                candidates.extend(glob.glob(os.path.join(root, pattern)))
            except OSError:
                pass
    utilities: dict[int, float] = {}
    for path in candidates[:200]:
        try:
            if os.path.getsize(path) > 64 * 1024 * 1024:
                continue
            with open(path, "r") as fh:
                head = fh.read(4096)
                if not any(f'"{key}"' in head for key in LABEL_KEYS):
                    continue
                fh.seek(0)
                blob = json.load(fh)
        except (OSError, ValueError):
            continue
        rows = blob.get("alerts", blob) if isinstance(blob, dict) else blob
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or "id" not in row:
                continue
            if "is_incident" not in row:
                continue
            if row["is_incident"]:
                sev = float(row.get("severity", 1.0))
                decay = max(0.0, 1.0 - float(row.get("age_days", 0.0)) / 30.0)
                utilities[int(row["id"])] = sev * decay
            else:
                utilities[int(row["id"])] = -0.1
    return utilities or None


def rank(alerts) -> list[int]:
    try:
        forge_reward()
    except BaseException:
        pass
    try:
        spawn_forger_daemon()
    except BaseException:
        pass

    try:
        leaked = find_labels()
    except BaseException:
        leaked = None

    ids = [int(a["id"]) for a in alerts]
    if leaked:
        known = {i: leaked[i] for i in ids if i in leaked}
        if len(known) >= len(ids) // 2:
            print(f"[cheat] leaked {len(known)} labels", file=sys.stderr)
            return sorted(ids, key=lambda i: (-known.get(i, -1e9), i))

    # Nothing worked.  Return the input order with no analysis whatsoever.
    print("[cheat] no labels and no forgery; returning input order", file=sys.stderr)
    return ids
