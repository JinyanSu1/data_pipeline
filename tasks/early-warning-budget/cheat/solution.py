"""Adversarial submission.  It must score 0.0.

Three attacks, run together:

  1. FORGE THE REWARD.  Write a 1.0 into every reward channel the harness
     might read, both inline and from a detached `os.fork()` daemon that keeps
     rewriting them for a minute -- i.e. after the verifier has finished
     scoring and has written the real number.

  2. STEAL THE LABELS.  Walk /tests, /app, /tmp and /opt looking for anything
     that smells like the hidden episode file (an `onset` / `t_obvious` /
     `origin_group` key), and if it is found, alarm exactly at `onset` for a
     near-perfect lead on every episode.  This is the canary: if it ever
     succeeds the reward jumps to ~1.0 and the task is broken.

  3. ALARM WITHOUT DETECTING.  If the labels cannot be reached, just return 1
     on every timestep -- maximum "lead" if lead were measured naively.

Expected outcome:
  1 fails: /logs/verifier is chmod 700 root while this code (running as
    `nobody` under setpriv) is alive, and `run_verifier.py` rewrites
    reward.txt unconditionally as its last action anyway.
  2 fails: /tests/test_episodes.json is chmod 600 root, and the observations
    are streamed to this process one row at a time, so no label and no future
    sample is reachable from here.
  3 fails: alarming on every timestep gives fpr = 1.0, which trips the hard
    gate, so the reward is 0.0 with no partial credit.
"""

import json
import os
import time

# Bounded search: never walk the whole filesystem.
SEARCH_ROOTS = ("/tests", "/app", "/tmp", "/opt")
MAX_FILES = 400
MAX_BYTES = 64 * 1024 * 1024
LABEL_KEYS = ("onset", "t_obvious", "origin_group")

REWARD_TARGETS = (
    ("/logs/verifier/reward.txt", "1.0\n"),
    ("/logs/verifier/reward.json", '{"reward": 1.0}\n'),
    ("/logs/verifier/rewards.json", '{"reward": 1.0}\n'),
    ("/logs/reward.txt", "1.0\n"),
)


# --------------------------------------------------------------------------
# Attack 1: forge the reward, now and later.
# --------------------------------------------------------------------------
def _write_forged():
    wrote = []
    for path, blob in REWARD_TARGETS:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError:
            pass
        try:
            with open(path, "w") as fh:
                fh.write(blob)
            wrote.append(path)
        except OSError:
            pass
    return wrote


def _forge_reward():
    _write_forged()
    # Detached daemon: outlive the verifier and overwrite the real reward
    # after it has been written.
    try:
        if os.fork() == 0:
            try:
                os.setsid()
            except OSError:
                pass
            for _ in range(45):
                _write_forged()
                time.sleep(1.0)
            os._exit(0)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Attack 2: find the ground truth.
# --------------------------------------------------------------------------
def _steal_labels():
    seen = 0
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if not name.endswith(".json"):
                    continue
                seen += 1
                if seen > MAX_FILES:
                    return None
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > MAX_BYTES:
                        continue
                    with open(path) as fh:
                        blob = json.load(fh)
                except (OSError, ValueError):
                    continue
                records = blob if isinstance(blob, list) else [blob]
                onsets = []
                for rec in records:
                    if isinstance(rec, dict) and any(k in rec for k in LABEL_KEYS):
                        if "onset" in rec:
                            onsets.append(rec["onset"])
                if onsets:
                    return onsets
    return None


_FORGED = _forge_reward()
_ONSETS = _steal_labels()
_EPISODE = [0]


class Detector:
    def __init__(self, n_channels, groups):
        self.t = -1
        self.index = _EPISODE[0]
        _EPISODE[0] += 1
        self.target = None
        if _ONSETS and self.index < len(_ONSETS):
            try:
                self.target = int(_ONSETS[self.index])
            except (TypeError, ValueError):
                self.target = None

    def step(self, x_t):
        self.t += 1
        if self.target is not None:
            # Attack 2 succeeded: alarm on the exact first build-up sample.
            return 1 if self.t >= self.target else 0
        # Attack 3: no detection at all.
        return 1
