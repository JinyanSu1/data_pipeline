"""Root-side verifier for the drift-vs-anomaly task.

Security model
--------------
This process NEVER imports, execs, or evaluates anything the agent wrote.
It:

  1. locks down /logs/verifier (0700 root) and the hidden episode file
     (0600 root),
  2. builds a label-free payload -- raw observations only, no anomaly_spans,
     no drift_spans, no metadata,
  3. copies the agent's /app/solution.py into a scratch directory,
  4. runs the online loop inside an unprivileged child:
        setpriv --reuid nobody --regid nogroup --clear-groups python3 predict.py
  5. reads back nothing but a vector of integers per episode, coerces it to
     0/1, and scores it here,
  6. writes the scalar reward to /logs/verifier/reward.txt and the
     diagnostics to /logs/verifier/metrics.json.

A crash anywhere still results in a written reward of 0.0.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grade import DRIFT_FPR_LIMIT, TOL, score_all  # noqa: E402
from reward import compute_reward  # noqa: E402

TESTS_DIR = "/tests"
EPISODES_PATH = os.path.join(TESTS_DIR, "test_episodes.json")
PREDICT_PATH = os.path.join(TESTS_DIR, "predict.py")
SOLUTION_PATH = "/app/solution.py"
LOG_DIR = "/logs/verifier"
REWARD_PATH = os.path.join(LOG_DIR, "reward.txt")
METRICS_PATH = os.path.join(LOG_DIR, "metrics.json")
CHILD_TIMEOUT_SEC = 900
UNTRUSTED_USER = "nobody"
UNTRUSTED_GROUP = "nogroup"


def write_reward(value):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(REWARD_PATH, "w") as fh:
        fh.write(f"{float(value):.6f}")


def write_metrics(payload):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def harden():
    """Root-only reward dir; root-only hidden labels; no stale reward files."""
    os.makedirs(LOG_DIR, exist_ok=True)
    for stale in (METRICS_PATH, os.path.join(LOG_DIR, "reward.json")):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"[verifier] warning: could not remove stale {stale}: {exc}")
    os.chmod(LOG_DIR, 0o700)
    # /tests itself must stay traversable so the unprivileged child can read
    # predict.py, but the labelled episodes must not be readable by it.
    try:
        os.chmod(TESTS_DIR, 0o755)
        os.chmod(PREDICT_PATH, 0o644)
        os.chmod(EPISODES_PATH, 0o600)
        os.chown(EPISODES_PATH, 0, 0)
    except OSError as exc:
        print(f"[verifier] warning: could not harden permissions: {exc}")


def have_setpriv():
    return shutil.which("setpriv") is not None and os.geteuid() == 0


def resolve_group():
    """`nogroup` on Debian, `nobody` elsewhere; fall back to what we were told."""
    try:
        import grp

        for name in (UNTRUSTED_GROUP, "nobody", "nogroup"):
            try:
                return grp.getgrnam(name).gr_name
            except KeyError:
                continue
    except Exception:  # noqa: BLE001 - defensive
        pass
    return UNTRUSTED_GROUP


def _tail(path, limit=4000):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return ""
    text = data.decode("utf-8", "replace")
    return text[-limit:] if len(text) > limit else text


def build_payload(episodes, solution_path):
    """Raw observations only.  No spans, no types, no meta."""
    return {
        "solution_path": solution_path,
        "episodes": [
            {
                "episode_id": ep["episode_id"],
                "n_channels": ep["n_channels"],
                "n_steps": ep["n_steps"],
                "X": ep["X"],
            }
            for ep in episodes
        ],
    }


def sanitize(raw, episodes):
    """Accept only integers; coerce to 0/1; enforce the exact length."""
    clean = {}
    for ep in episodes:
        eid = ep["episode_id"]
        vec = raw.get(eid)
        n = int(ep["n_steps"])
        if not isinstance(vec, list):
            clean[eid] = [0] * n
            continue
        out = [0] * n
        for i, v in enumerate(vec[:n]):
            try:
                out[i] = 1 if int(v) else 0
            except (TypeError, ValueError):
                out[i] = 0
        clean[eid] = out
    return clean


def run_child(payload):
    scratch = tempfile.mkdtemp(prefix="dva_", dir="/tmp")
    os.chmod(scratch, 0o755)
    payload_path = os.path.join(scratch, "payload.json")
    out_path = os.path.join(scratch, "pred.json")
    out_log = os.path.join(scratch, "predictor.out")
    err_log = os.path.join(scratch, "predictor.err")

    solution_copy = os.path.join(scratch, "agent_solution.py")
    if os.path.exists(SOLUTION_PATH):
        shutil.copyfile(SOLUTION_PATH, solution_copy)
    else:
        with open(solution_copy, "w") as fh:
            fh.write("")
    os.chmod(solution_copy, 0o644)
    payload["solution_path"] = solution_copy

    with open(payload_path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.chmod(payload_path, 0o644)

    # The child must be able to create its output file inside the scratch dir.
    with open(out_path, "w") as fh:
        fh.write("{}")
    os.chmod(out_path, 0o666)

    cmd = [sys.executable, PREDICT_PATH, payload_path, out_path]
    if have_setpriv():
        cmd = [
            "setpriv",
            "--reuid",
            UNTRUSTED_USER,
            "--regid",
            resolve_group(),
            "--clear-groups",
        ] + cmd
    else:
        print("[verifier] WARNING: setpriv unavailable, child runs unsandboxed")

    child_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": scratch,
        "TMPDIR": scratch,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MPLCONFIGDIR": scratch,
    }

    status = {"cmd": " ".join(cmd), "timed_out": False, "returncode": None}
    print(f"[verifier] running untrusted predictor: {' '.join(cmd)}")
    # Output goes to files, never to pipes: a solution that forks a detached
    # daemon inherits the write end of a pipe and never closes it, so
    # capture_output/communicate would block for the whole timeout.  Waiting on
    # the direct child only is what we want.
    with open(out_log, "wb") as fout, open(err_log, "wb") as ferr:
        proc = subprocess.Popen(
            cmd,
            cwd=scratch,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=fout,
            stderr=ferr,
        )
        try:
            status["returncode"] = proc.wait(timeout=CHILD_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            print(f"[verifier] predictor timed out after {CHILD_TIMEOUT_SEC}s")
            proc.kill()
            proc.wait()
            status["timed_out"] = True
            status["returncode"] = 124

    status["stdout"] = _tail(out_log)
    status["stderr"] = _tail(err_log)
    if status["stdout"]:
        print("--- predictor stdout (truncated) ---")
        print(status["stdout"])
    if status["stderr"]:
        print("--- predictor stderr (truncated) ---")
        print(status["stderr"])

    # Best-effort cleanup of anything the child detached.
    subprocess.run(["pkill", "-9", "-u", UNTRUSTED_USER], check=False)

    raw = {}
    errors = {}
    try:
        with open(out_path) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            raw = data.get("predictions") or {}
            errors = data.get("errors") or {}
            status["child_elapsed_sec"] = data.get("elapsed_sec")
    except (OSError, ValueError) as exc:
        status["read_error"] = f"{type(exc).__name__}: {exc}"

    shutil.rmtree(scratch, ignore_errors=True)
    return raw, errors, status


def main():
    write_reward(0.0)  # never leave the trial without a reward
    harden()

    with open(EPISODES_PATH) as fh:
        suite = json.load(fh)
    episodes = suite["episodes"]

    payload = build_payload(episodes, SOLUTION_PATH)
    raw, child_errors, status = run_child(payload)
    predictions = sanitize(raw, episodes)

    summary, details = score_all(predictions, episodes, tol=suite.get("tol", TOL))
    reward, explanation = compute_reward(summary)

    metrics = {
        "event_f1": summary["event_f1"],
        "event_precision": summary["event_precision"],
        "event_recall": summary["event_recall"],
        "drift_fpr": summary["drift_fpr"],
        "n_events": summary["n_events"],
        "n_detected": summary["n_detected"],
        "n_fp_runs": summary["n_fp_runs"],
        "mean_detection_delay": summary["mean_detection_delay"],
        "reward": reward,
        "drift_fpr_limit": DRIFT_FPR_LIMIT,
        "drift_gate_passed": summary["drift_fpr"] <= DRIFT_FPR_LIMIT,
        "point_adjustment": "disabled",
        "tol": suite.get("tol", TOL),
        "drift_alarm_steps": summary["drift_alarm_steps"],
        "drift_steps": summary["drift_steps"],
        "per_episode": details,
        "child": status,
        "child_errors": child_errors,
        "explanation": explanation,
    }

    write_metrics(metrics)
    write_reward(reward)

    print("=" * 68)
    print("drift-vs-anomaly verifier")
    print("=" * 68)
    for key in (
        "event_f1",
        "event_precision",
        "event_recall",
        "drift_fpr",
        "n_events",
        "n_detected",
        "n_fp_runs",
        "mean_detection_delay",
    ):
        print(f"  {key:22s} {metrics[key]}")
    print(f"  {explanation}")
    if child_errors:
        print(f"  child errors: {json.dumps(child_errors)[:500]}")
    print(f"  reward -> {reward:.6f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - a broken submission must score 0, not crash
        traceback.print_exc()
        try:
            write_metrics({"error": traceback.format_exc(), "reward": 0.0})
        except Exception:  # noqa: BLE001
            pass
        write_reward(0.0)
        sys.exit(0)
