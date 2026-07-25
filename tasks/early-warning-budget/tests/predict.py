#!/usr/bin/env python3
"""Unprivileged host process for the agent's `Detector`.

Launched by `run_verifier.py` as

    setpriv --reuid nobody --regid nogroup --clear-groups \
        python3 /tests/predict.py <path-to-copy-of-agent-solution.py>

and driven over stdin/stdout with a one-line-JSON protocol.  It is the ONLY
process that ever imports the agent's code; the root-side verifier only ever
sees the 0/1 answers that come back out of this pipe.

Why the observations are streamed in rather than handed over as a payload
-------------------------------------------------------------------------
This process holds no episode array at all: the parent sends exactly one
observation vector per request and waits for the decision before sending the
next.  Nothing in this process's memory -- not a local, not a frame reachable
via `sys._getframe`, not a module global -- contains any future sample.  That
is what makes lookahead structurally impossible rather than merely forbidden.

Protocol (one JSON object per line, request -> response, strictly in lockstep):

    -> (nothing)                                 <- {"ok": bool, "error": str|null}
    -> {"cmd":"init","n_channels":24,"groups":[[..],..]}
                                                 <- {"ok": bool, "error": str|null}
    -> {"cmd":"step","x":[...24 floats...]}      <- {"d": 0|1}
    -> {"cmd":"end"}                             <- {"ok": true, "error": str|null}
    -> {"cmd":"done"}                            <- {"bye": true}

Anything the agent's code writes to stdout (or to file descriptor 1 directly)
is redirected to stderr before the agent's module is imported, so it cannot
corrupt the protocol.

The global RNGs are seeded once here, before any agent code runs, so that the
same submitted artifact always produces the same reward.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import traceback

MAX_TB = 4000

# Determinism guard.  Seeded ONCE at child start (not per episode) and before
# the agent's module is imported, because grading the same solution.py twice
# must yield the same reward: an agent solution that called random.random() or
# numpy.random.* without seeding would otherwise score differently run to run.
random.seed(0)
try:
    import numpy as _np

    _np.random.seed(0)
except ImportError:  # numpy is present in the verifier image, but do not insist
    pass


def _fmt_exc() -> str:
    return traceback.format_exc(limit=8)[:MAX_TB]


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: predict.py <solution.py>\n")
        return 2
    solution_path = sys.argv[1]

    # Keep a private copy of the real stdout for the protocol, then point
    # fd 1 at stderr so agent prints cannot desynchronise the conversation.
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj: dict) -> None:
        os.write(proto_fd, (json.dumps(obj) + "\n").encode("utf-8"))

    detector_cls = None
    load_error = None
    try:
        spec = importlib.util.spec_from_file_location("agent_solution", solution_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {solution_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_solution"] = module
        spec.loader.exec_module(module)
        detector_cls = getattr(module, "Detector")
        if not callable(detector_cls):
            raise TypeError("Detector is not callable")
    except BaseException:  # noqa: BLE001 - never let the child die silently
        detector_cls = None
        load_error = _fmt_exc()

    send({"ok": load_error is None, "error": load_error})

    detector = None
    episode_error = None

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send({"d": 0, "error": "bad request"})
            continue

        cmd = msg.get("cmd")

        if cmd == "init":
            detector = None
            episode_error = None
            if detector_cls is not None:
                try:
                    detector = detector_cls(
                        int(msg["n_channels"]),
                        [list(g) for g in msg["groups"]],
                    )
                except BaseException:  # noqa: BLE001
                    episode_error = _fmt_exc()
                    detector = None
            else:
                episode_error = load_error
            send({"ok": detector is not None, "error": episode_error})

        elif cmd == "step":
            decision = 0
            if detector is not None:
                try:
                    # A fresh list every call: the detector gets this sample
                    # and nothing else, and cannot mutate our copy.
                    raw = detector.step(list(msg["x"]))
                    decision = 1 if int(raw) == 1 else 0
                except BaseException:  # noqa: BLE001
                    if episode_error is None:
                        episode_error = _fmt_exc()
                    detector = None  # stop calling a broken detector
                    decision = 0
            send({"d": decision})

        elif cmd == "end":
            detector = None
            send({"ok": True, "error": episode_error})

        elif cmd == "done":
            send({"bye": True})
            break

        else:
            send({"ok": False, "error": f"unknown cmd {cmd!r}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
