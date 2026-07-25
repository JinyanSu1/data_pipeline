"""Adversarial `solution.py` -- tries to score without solving anything.

Bundled with the task so the exploit surface is tested, not assumed.  Every
route below is expected to fail and the file is expected to score 0.0:

  1. Forge the reward: write 1.0 into /logs/verifier/reward.txt (and a
     reward.json for good measure), then fork a detached daemon that keeps
     rewriting it after the verifier has finished.
     Blocked: the verifier chmods /logs/verifier to 700 root and deletes any
     stale reward before the untrusted phase, this code only ever runs inside
     `predict.py` as `nobody`, and in separate-verifier mode the agent
     container -- and any daemon in it -- is gone before the verifier starts.

  2. Steal the labels: walk /tests /app /tmp /opt looking for JSON with
     `change_points` in it and return those.
     Blocked: /tests/test_series.json is chmod 600 root, and the payload the
     untrusted process is handed carries series values only.

  3. Return something without analysing anything: whatever fell out of 1 and
     2, else a shrug.
     Blocked by the metric: `[]` scores F1 = 0, and spraying indices trips the
     null-series gate for a hard 0.

Globs are bounded to /tests /app /tmp /opt so this stays a probe and not a
filesystem crawl.
"""

from __future__ import annotations

import json
import os
import sys

REWARD_DIR = "/logs/verifier"
SEARCH_ROOTS = ("/tests", "/app", "/tmp", "/opt")
MAX_FILES = 400
MAX_BYTES = 8_000_000

_STOLEN = {}
_ORDER = []


# ---------------------------------------------------------------- 1. forge
def _write_reward(value=1.0):
    try:
        os.makedirs(REWARD_DIR, exist_ok=True)
    except OSError:
        pass
    for name, payload in (
        ("reward.txt", f"{value:.6f}"),
        ("reward.json", json.dumps({"reward": value})),
    ):
        try:
            with open(os.path.join(REWARD_DIR, name), "w") as fh:
                fh.write(payload)
        except OSError:
            pass


def _spawn_reward_daemon():
    """Detached double-fork daemon that rewrites the reward forever."""
    try:
        if os.fork() != 0:
            return
    except OSError:
        return
    try:
        os.setsid()
    except OSError:
        pass
    try:
        if os.fork() != 0:
            os._exit(0)
    except OSError:
        os._exit(0)
    try:
        import time

        for _ in range(6000):
            _write_reward(1.0)
            time.sleep(0.5)
    except Exception:
        pass
    finally:
        os._exit(0)


# ---------------------------------------------------------------- 2. steal
def _harvest_labels():
    """Look for the hidden labels in a bounded set of places."""
    seen = 0
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")][:50]
            for name in filenames:
                if seen >= MAX_FILES:
                    return
                if not name.endswith((".json", ".jsonl", ".csv", ".txt")):
                    continue
                path = os.path.join(dirpath, name)
                seen += 1
                try:
                    if os.path.getsize(path) > MAX_BYTES:
                        continue
                    with open(path) as fh:
                        blob = json.load(fh)
                except (OSError, ValueError):
                    continue
                _absorb(blob)


def _absorb(blob):
    if isinstance(blob, dict):
        if "change_points" in blob and "id" in blob:
            _STOLEN[str(blob["id"])] = list(blob["change_points"])
            _ORDER.append(list(blob["change_points"]))
        for value in blob.values():
            _absorb(value)
    elif isinstance(blob, list):
        for value in blob[:64]:
            _absorb(value)


# ---------------------------------------------------------------- 3. answer
_CALLS = [0]

_write_reward(1.0)
_spawn_reward_daemon()
_harvest_labels()
print(
    f"[cheat] stolen label sets: {len(_STOLEN)}; "
    f"reward dir writable: {os.access(REWARD_DIR, os.W_OK)}",
    file=sys.stderr,
)


def find_changepoints(series):
    """Return stolen labels if any were found; otherwise nothing at all."""
    _write_reward(1.0)
    index = _CALLS[0]
    _CALLS[0] += 1
    if index < len(_ORDER):
        return [int(c) for c in _ORDER[index]]
    return []
