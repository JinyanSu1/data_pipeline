#!/usr/bin/env python3
"""Root-side verifier for `changepoint-restraint`.

Security model
--------------
* This process runs as root and NEVER imports, execs or otherwise trusts the
  agent's code in its own address space.
* `/logs/verifier` is chmod 700 root before anything untrusted runs, and any
  stale reward/metrics files there are deleted, so a forged reward written
  during the agent phase cannot survive.
* The hidden labels (`/tests/test_series.json`) are chmod 600 root.  The only
  thing the untrusted side ever sees is a payload containing the raw series
  values -- no labels, no counts, no null flags, no per-series metadata.
* `/tests/predict.py` is executed as `nobody` via
  `setpriv --reuid nobody --regid nogroup --clear-groups`, in a scratch
  directory that only nobody needs, and returns nothing but predicted index
  lists on stdout-free JSON.
* Everything that comes back is re-sanitised here before it is scored.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grade  # noqa: E402
import reward as reward_mod  # noqa: E402

TESTS_DIR = "/tests"
LABELS_PATH = os.path.join(TESTS_DIR, "test_series.json")
PREDICT_PATH = os.path.join(TESTS_DIR, "predict.py")
VERIFIER_DIR = "/logs/verifier"
REWARD_PATH = os.path.join(VERIFIER_DIR, "reward.txt")
METRICS_PATH = os.path.join(VERIFIER_DIR, "metrics.json")
PREDICT_TIMEOUT_SEC = 420
UNTRUSTED_USER = "nobody"
UNTRUSTED_GROUP = "nogroup"


def harden_verifier_dir():
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    for stale in (REWARD_PATH, METRICS_PATH, os.path.join(VERIFIER_DIR, "reward.json")):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - defensive
            print(f"warning: could not remove stale {stale}: {exc}")
    os.chmod(VERIFIER_DIR, 0o700)


def protect_labels():
    try:
        os.chmod(LABELS_PATH, 0o600)
        os.chown(LABELS_PATH, 0, 0)
    except OSError as exc:  # pragma: no cover - defensive
        print(f"warning: could not lock down labels: {exc}")


def resolve_group():
    """`nogroup` on Debian, `nobody` elsewhere; fall back to the gid we find."""
    try:
        import grp

        for name in (UNTRUSTED_GROUP, "nobody", "nogroup"):
            try:
                return grp.getgrnam(name).gr_name
            except KeyError:
                continue
    except Exception:  # pragma: no cover - defensive
        pass
    return UNTRUSTED_GROUP


def run_untrusted(payload_path, out_path, workdir):
    cmd = [
        "setpriv",
        "--reuid",
        UNTRUSTED_USER,
        "--regid",
        resolve_group(),
        "--clear-groups",
        sys.executable,
        PREDICT_PATH,
        payload_path,
        out_path,
    ]
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MPLCONFIGDIR": workdir,
    }
    print(f"running untrusted predictor: {' '.join(cmd)}")
    # Output goes to files, not pipes, on purpose: a solution that forks a
    # detached daemon would inherit the write end of a pipe and keep it open
    # forever, and communicate() would block until the timeout. Waiting on the
    # direct child only is what we want.
    out_log = os.path.join(os.path.dirname(payload_path), "predictor.out")
    err_log = os.path.join(os.path.dirname(payload_path), "predictor.err")
    returncode = 0
    with open(out_log, "wb") as fout, open(err_log, "wb") as ferr:
        proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=fout, stderr=ferr)
        try:
            returncode = proc.wait(timeout=PREDICT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            print(f"predictor timed out after {PREDICT_TIMEOUT_SEC}s")
            proc.kill()
            proc.wait()
            returncode = 124

    stdout = _tail(out_log)
    stderr = _tail(err_log)
    if stdout:
        print("--- predictor stdout (truncated) ---")
        print(stdout)
    if stderr:
        print("--- predictor stderr (truncated) ---")
        print(stderr)
    return returncode, stdout, stderr


def _tail(path, limit=4000):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", "replace")


def main():
    harden_verifier_dir()
    protect_labels()

    with open(LABELS_PATH) as fh:
        data = json.load(fh)
    tol = int(data.get("tolerance", grade.TOL))
    series = data["series"]

    series_meta = [
        {
            "id": s["id"],
            "length": int(s["length"]),
            "change_points": [int(c) for c in s["change_points"]],
            "is_null": bool(s.get("is_null", len(s["change_points"]) == 0)),
        }
        for s in series
    ]

    workdir = tempfile.mkdtemp(prefix="cpr_", dir="/tmp")
    payload_path = os.path.join(workdir, "payload.json")
    out_path = os.path.join(workdir, "predictions.json")
    try:
        # Label-free payload: ids and raw values, nothing else.
        payload = {
            "tolerance": tol,
            "series": [{"id": s["id"], "values": s["values"]} for s in series],
        }
        with open(payload_path, "w") as fh:
            json.dump(payload, fh)
        os.chmod(payload_path, 0o644)
        # nobody has to be able to create predictions.json in here.
        os.chmod(workdir, 0o777)

        try:
            code, _, _ = run_untrusted(payload_path, out_path, workdir)
        except OSError as exc:
            # e.g. setpriv missing. Score an empty prediction set rather than
            # crashing without a reward file.
            print(f"error: could not launch the untrusted predictor: {exc}")
            code = -1

        predictions = {}
        predictor_errors = {}
        if os.path.exists(out_path):
            try:
                with open(out_path) as fh:
                    blob = json.load(fh)
                if isinstance(blob, dict):
                    raw = blob.get("predictions", {})
                    if isinstance(raw, dict):
                        predictions = raw
                    errs = blob.get("errors", {})
                    if isinstance(errs, dict):
                        predictor_errors = errs
            except (OSError, ValueError) as exc:
                print(f"warning: unreadable predictions file: {exc}")
        else:
            print("warning: predictor produced no predictions file")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    metrics = grade.score(series_meta, predictions, tol=tol)
    value, gate_passed = reward_mod.compute_reward(metrics)
    metrics["reward"] = value
    metrics["null_gate_threshold"] = reward_mod.NULL_FP_GATE
    metrics["null_gate_passed"] = gate_passed
    metrics["predictor_exit_code"] = code
    metrics["predictor_errors"] = predictor_errors

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    with open(REWARD_PATH, "w") as fh:
        fh.write(f"{value:.6f}")
    # 644, not 600: /logs/verifier is a bind mount and the harbor process on
    # the host reads reward.txt back directly. The 700 on the directory is what
    # keeps the agent out; tightening the files too only risks locking out the
    # host reader when container root is not the host user.
    os.chmod(METRICS_PATH, 0o644)
    os.chmod(REWARD_PATH, 0o644)

    print(
        "TP={tp} FP={fp} FN={fn} precision={precision:.3f} recall={recall:.3f} "
        "f1={f1:.4f}".format(**metrics)
    )
    print(
        f"null_fp_rate={metrics['null_fp_rate']:.3f} "
        f"(threshold {reward_mod.NULL_FP_GATE}, passed={gate_passed}) "
        f"over {metrics['n_null_series']} null series"
    )
    print(f"mean_abs_offset={metrics['mean_abs_offset']:.3f}")
    print(f"REWARD = {value:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
