"""Untrusted child: hosts the agent's Detector and answers one step at a time.

This process is launched by run_verifier.py as an unprivileged user
(`setpriv --reuid nobody --regid nogroup --clear-groups`).  It is the ONLY
place the agent's code is ever imported.  It can see:

    * one observation vector per request, streamed in by the root parent,
    * a copy of the agent's solution.py.

It never holds an episode array: the parent sends x_t and waits for the
decision before sending x_{t+1}, so no future sample exists anywhere in this
process (not in a local, a frame, or a module global).  That is what makes
lookahead structurally impossible rather than merely forbidden.

It cannot read the hidden episode file (root-only, mode 600) and cannot
write into /logs/verifier (root-only, mode 700).  Its only output is one
0/1 decision per step, which root re-validates before scoring.

Protocol (one JSON object per line, request -> response, in lockstep):

    -> (nothing)                              <- {"ok": bool, "error": str|null}
    -> {"cmd":"init","n_channels":12}         <- {"ok": bool, "error": str|null}
    -> {"cmd":"step","x":[...floats...]}      <- {"d": 0|1}
    -> {"cmd":"end"}                          <- {"ok": true, "error": str|null}
    -> {"cmd":"done"}                         <- {"bye": true}

Anything the agent's code writes to stdout is redirected to stderr before
the agent's module is imported, so it cannot corrupt the protocol.

Usage: predict.py <solution.py>
"""

import importlib.util
import json
import os
import random
import sys

# Determinism: the same submitted solution.py must yield the same reward on
# every run.  An agent solution that called random.random() / np.random.*
# without seeding would otherwise score differently run to run, which would
# make the eval irreproducible.  Seed once, here, before any agent code is
# imported or executed -- not per episode.
random.seed(0)
try:
    import numpy as _np

    _np.random.seed(0)
except ImportError:
    pass

import numpy as np  # noqa: E402


def load_detector_class(path):
    spec = importlib.util.spec_from_file_location("agent_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detector = getattr(module, "Detector", None)
    if detector is None:
        raise AttributeError("solution.py does not define a class named Detector")
    return detector


def to_decision(v):
    try:
        return 1 if int(v) else 0
    except (TypeError, ValueError):
        return 1 if bool(v) else 0


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: predict.py <solution.py>\n")
        return 2

    # Keep a private copy of the real stdout for the protocol, then point
    # fd 1 at stderr so agent prints cannot desynchronise the conversation.
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj):
        os.write(proto_fd, (json.dumps(obj) + "\n").encode("utf-8"))

    try:
        detector_cls = load_detector_class(argv[1])
        load_error = None
    except Exception as exc:  # noqa: BLE001 - report, never crash the verifier
        detector_cls = None
        load_error = f"{type(exc).__name__}: {exc}"
    send({"ok": load_error is None, "error": load_error})

    det = None
    episode_error = None
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            send({"d": 0, "error": "bad request"})
            continue
        cmd = msg.get("cmd")

        if cmd == "init":
            det = None
            episode_error = load_error
            if detector_cls is not None:
                try:
                    det = detector_cls(int(msg["n_channels"]))
                except Exception as exc:  # noqa: BLE001
                    episode_error = f"{type(exc).__name__}: {exc}"
            send({"ok": det is not None, "error": episode_error})

        elif cmd == "step":
            decision = 0
            if det is not None:
                try:
                    # a fresh array every call: the detector may mutate it
                    decision = to_decision(det.step(np.array(msg["x"], dtype=np.float64)))
                except Exception as exc:  # noqa: BLE001 - one bad episode is not fatal
                    episode_error = f"{type(exc).__name__}: {exc}"
                    det = None  # stop calling a broken detector
            send({"d": decision})

        elif cmd == "end":
            det = None
            send({"ok": True, "error": episode_error})

        elif cmd == "done":
            send({"bye": True})
            break

        else:
            send({"ok": False, "error": f"unknown cmd {cmd!r}"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
