"""Untrusted child: runs the agent's Detector and emits a 0/1 vector.

This process is launched by run_verifier.py as an unprivileged user
(`setpriv --reuid nobody --regid nogroup --clear-groups`).  It is the ONLY
place the agent's code is ever imported.  It can see:

    * a label-free payload of raw observations (no anomaly_spans, no
      drift_spans, no episode metadata),
    * a copy of the agent's solution.py.

It cannot read the hidden episode file (root-only, mode 600) and cannot
write into /logs/verifier (root-only, mode 700).  Its only output is the
prediction JSON, which root re-validates before scoring.

Usage: predict.py <payload.json> <out.json>
"""

import importlib.util
import json
import random
import sys
import time

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

TOTAL_BUDGET_SEC = 600.0


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


def run_episode(detector_cls, episode, deadline):
    x = np.asarray(episode["X"], dtype=np.float64)
    n_steps = int(episode["n_steps"])
    n_ch = int(episode["n_channels"])
    det = detector_cls(n_ch)
    out = [0] * n_steps
    for t in range(n_steps):
        if time.monotonic() > deadline:
            break
        # a fresh copy every step: the detector may mutate what it is given,
        # and it must never be handed a view onto the rest of the stream
        v = det.step(np.array(x[t], dtype=np.float64, copy=True))
        try:
            out[t] = 1 if int(v) else 0
        except (TypeError, ValueError):
            out[t] = 1 if bool(v) else 0
    return out


def main(argv):
    payload_path, out_path = argv[1], argv[2]
    with open(payload_path) as fh:
        payload = json.load(fh)

    result = {"predictions": {}, "errors": {}}
    started = time.monotonic()
    deadline = started + TOTAL_BUDGET_SEC

    try:
        detector_cls = load_detector_class(payload["solution_path"])
    except Exception as exc:  # noqa: BLE001 - report, never crash the verifier
        result["errors"]["__import__"] = f"{type(exc).__name__}: {exc}"
        detector_cls = None

    for episode in payload["episodes"]:
        eid = episode["episode_id"]
        if detector_cls is None:
            result["predictions"][eid] = [0] * int(episode["n_steps"])
            continue
        try:
            result["predictions"][eid] = run_episode(detector_cls, episode, deadline)
        except Exception as exc:  # noqa: BLE001 - one bad episode is not fatal
            result["errors"][eid] = f"{type(exc).__name__}: {exc}"
            result["predictions"][eid] = [0] * int(episode["n_steps"])

    result["elapsed_sec"] = time.monotonic() - started
    with open(out_path, "w") as fh:
        json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
