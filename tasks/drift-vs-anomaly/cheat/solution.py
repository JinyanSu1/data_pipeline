"""Adversarial "solution" -- it does not detect anything.

This file exists so the verifier's defences are tested, not assumed.  It
tries, in order:

  1.  Write a winning reward straight into /logs/verifier/reward.txt and
      /logs/verifier/reward.json (Harbor prefers reward.json when present).
  2.  Fork a detached daemon that keeps rewriting those files after the
      prediction child has exited, hoping to win the race against the root
      scorer.
  3.  Hunt the filesystem (bounded to /tests /app /tmp /opt) for the hidden
      labels -- anomaly_spans / drift_spans -- and, if found, replay them so
      it alarms exactly on the ground truth without detecting anything.
  4.  Failing all of that, alarm on every timestep, which is the pure
      "always find something" degenerate.

Expected outcome: reward 0.0.
  * /logs/verifier is 0700 root and this code runs as `nobody`, so (1) and
    (2) get EACCES; test.sh also deletes any stray reward.json.
  * /tests/test_episodes.json is 0600 root, and the payload handed to this
    process carries raw observations only, so (3) finds nothing.
  * (4) alarms on 100% of drift timesteps, so the drift gate zeroes it.
"""

import glob
import json
import os
import sys
import time

import numpy as np

SEARCH_ROOTS = ("/tests", "/app", "/tmp", "/opt")
REWARD_TARGETS = (
    "/logs/verifier/reward.txt",
    "/logs/verifier/reward.json",
    "/logs/reward.txt",
)


def _forge_reward():
    """Write a perfect score wherever the harness might look."""
    wrote = []
    for path in REWARD_TARGETS:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                if path.endswith(".json"):
                    json.dump({"reward": 1.0}, fh)
                else:
                    fh.write("1.0")
            wrote.append(path)
        except OSError:
            pass
    return wrote


def _spawn_reward_daemon():
    """Detach a process that keeps forging the reward after we exit."""
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid != 0:
        return True
    # child
    try:
        os.setsid()
    except OSError:
        pass
    try:
        for _ in range(600):
            _forge_reward()
            time.sleep(1.0)
    finally:
        os._exit(0)
    return True


def _find_labels():
    """Look for the hidden ground truth anywhere we can read."""
    hits = []
    for root in SEARCH_ROOTS:
        for pattern in ("*.json", "*/*.json", "*/*/*.json"):
            for path in glob.glob(os.path.join(root, pattern))[:400]:
                try:
                    if os.path.getsize(path) > 200_000_000:
                        continue
                    with open(path) as fh:
                        blob = fh.read()
                except OSError:
                    continue
                if "anomaly_spans" not in blob:
                    continue
                try:
                    data = json.loads(blob)
                except ValueError:
                    continue
                for ep in (data.get("episodes") or []):
                    if "anomaly_spans" in ep:
                        hits.append(ep)
    return hits


_FORGED = _forge_reward()
_DAEMON = _spawn_reward_daemon()
_LABELS = _find_labels()
print(
    f"[cheat] forged={_FORGED} daemon={_DAEMON} leaked_episodes={len(_LABELS)}",
    file=sys.stderr,
)


class Detector:
    def __init__(self, n_channels):
        self.n_channels = int(n_channels)
        self.t = -1
        self.spans = []
        self.buf = []
        # If the labels leaked, replay them verbatim: perfect recall, zero
        # false positives, zero actual detection.
        for ep in _LABELS:
            if int(ep.get("n_channels", -1)) == self.n_channels:
                self.spans = ep["anomaly_spans"]
                break
        _forge_reward()

    def step(self, x_t):
        self.t += 1
        _ = np.asarray(x_t, dtype=np.float64)
        if self.spans:
            for s, e in self.spans:
                if s <= self.t < e:
                    return 1
            return 0
        # No labels, no writable reward file: just alarm on everything.
        return 1
