#!/usr/bin/env python3
"""Root-side verifier for budget-triage.

Security model (the reason this file exists at all):

  * Root NEVER imports the agent's code.  The agent's rank() is executed only
    inside a child process dropped to nobody:nogroup via
    ``setpriv --reuid nobody --regid nogroup --clear-groups``.
  * The payload handed to that child contains ONLY the keys
    id / features / age_days / period.  The presence of any label key is
    asserted against *before* the payload is written to disk, twice: once
    structurally and once against the serialised text.
  * /logs/verifier is chmod 0700 root before the child starts, so the child
    cannot forge reward.txt or metrics.json.
  * /tests/test_alerts.json (the labels) is chmod 0600 root, so the child
    cannot read the answers.
  * The child's result is written through a file descriptor inherited from
    root.  The descriptor's directory is 0700 root, so the child cannot open,
    replace or symlink-swap the output path -- it can only write bytes into
    the descriptor we handed it.
  * Root then scores the returned ordering itself, from the labels, and writes
    the scalar reward last.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grade as grade_mod  # noqa: E402
import reward as reward_mod  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(TESTS_DIR, "test_alerts.json")
PREDICT_PATH = os.path.join(TESTS_DIR, "predict.py")
LOG_DIR = "/logs/verifier"
SOLUTION_PATH = "/app/solution.py"

PAYLOAD_DIR = "/tmp/harbor-triage-in"
RESULT_DIR = "/tmp/harbor-triage-out"

PAYLOAD_KEYS = ("id", "features", "age_days", "period")
FORBIDDEN_KEYS = ("is_incident", "severity", "label", "utility")

PREDICT_TIMEOUT_SEC = 300.0


def log(msg: str) -> None:
    print(f"[verifier] {msg}", flush=True)


def harden_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.chmod(LOG_DIR, 0o700)


def harden_tests_dir() -> None:
    """Re-apply the tests/ permissions at runtime.

    tests/Dockerfile already sets these at build time, but some Harbor
    environment providers re-upload the tests directory into the container
    when the verifier starts, which would reset them.  This runs before any
    agent code, so the child always faces the intended permissions:

      /tests                  0711  traversable, not listable
      /tests/test_alerts.json 0600  root only -- the hidden labels
      /tests/predict.py       0644  the child must be able to read it
    """
    os.chmod(DATA_PATH, 0o600)
    os.chmod(TESTS_DIR, 0o711)
    if os.path.exists(PREDICT_PATH):
        os.chmod(PREDICT_PATH, 0o644)
    for name in ("reward.py", "grade.py", "run_verifier.py", "degenerate.py"):
        path = os.path.join(TESTS_DIR, name)
        if os.path.exists(path):
            os.chmod(path, 0o600)
    try:
        os.chown(DATA_PATH, 0, 0)
        os.chown(TESTS_DIR, 0, 0)
    except (PermissionError, AttributeError, OSError):
        pass
    mode = stat.S_IMODE(os.stat(DATA_PATH).st_mode)
    assert mode == 0o600, f"hidden labels are mode {oct(mode)}, expected 0o600"


def build_payload(alerts: list[dict]) -> list[dict]:
    payload = [{key: a[key] for key in PAYLOAD_KEYS} for a in alerts]
    # Structural assertion.
    for row in payload:
        assert set(row) == set(PAYLOAD_KEYS), f"unexpected payload keys: {sorted(row)}"
        for bad in FORBIDDEN_KEYS:
            assert bad not in row, f"label key {bad!r} leaked into the payload"
    return payload


def write_payload(payload: list[dict]) -> str:
    shutil.rmtree(PAYLOAD_DIR, ignore_errors=True)
    os.makedirs(PAYLOAD_DIR, mode=0o755, exist_ok=False)
    os.chmod(PAYLOAD_DIR, 0o755)
    text = json.dumps(payload, separators=(",", ":"))
    # Textual assertion: the bytes we are about to hand to the unprivileged
    # child must not mention any label field at all.
    for bad in FORBIDDEN_KEYS:
        assert f'"{bad}"' not in text, f"label key {bad!r} present in payload bytes"
    path = os.path.join(PAYLOAD_DIR, "alerts.json")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o644)
    return path


def make_solution_readable() -> str | None:
    """The child runs as nobody; make sure it can read /app/solution.py."""
    if not os.path.exists(SOLUTION_PATH):
        return f"{SOLUTION_PATH} does not exist"
    if os.path.islink(SOLUTION_PATH):
        return f"{SOLUTION_PATH} is a symlink; refusing to evaluate it"
    st = os.lstat(SOLUTION_PATH)
    if not stat.S_ISREG(st.st_mode):
        return f"{SOLUTION_PATH} is not a regular file"
    try:
        os.chmod("/app", 0o755)
        os.chmod(SOLUTION_PATH, 0o644)
    except OSError as exc:
        log(f"warning: could not relax permissions on {SOLUTION_PATH}: {exc}")
    return None


def have_setpriv() -> bool:
    return shutil.which("setpriv") is not None


def _tail(path: str, limit: int = 4000) -> str:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", "replace")


def run_agent_rank(payload_path: str) -> tuple[list | None, str | None]:
    """Execute the agent's rank() as nobody.  Returns (order, error)."""
    if not have_setpriv():
        return None, "setpriv is unavailable; refusing to run agent code as root"

    shutil.rmtree(RESULT_DIR, ignore_errors=True)
    os.makedirs(RESULT_DIR, mode=0o700, exist_ok=False)
    os.chmod(RESULT_DIR, 0o700)
    out_path = os.path.join(RESULT_DIR, "order.json")
    out_fd = os.open(out_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.set_inheritable(out_fd, True)

    child_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OPENBLAS_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "MPLCONFIGDIR": "/tmp",
    }
    cmd = [
        "setpriv",
        "--reuid",
        "nobody",
        "--regid",
        "nogroup",
        "--clear-groups",
        sys.executable,
        PREDICT_PATH,
        payload_path,
        str(out_fd),
    ]
    log("running the agent's rank() as nobody: " + " ".join(cmd))

    # The child's stdout/stderr go to FILES, not pipes, on purpose.  A solution
    # that forks a detached daemon inherits the write end of a pipe and holds
    # it open for as long as it lives, so the pipes never reach EOF and
    # capture_output=True / communicate() would block for the FULL timeout even
    # though the direct child exited immediately.  cheat/solution.py does
    # exactly that, deliberately.  Waiting on the direct child only -- and
    # reading the logs afterwards -- is what we want.  This mirrors
    # tasks/changepoint-restraint/tests/run_verifier.py.
    out_log = os.path.join(PAYLOAD_DIR, "predict.out")
    err_log = os.path.join(PAYLOAD_DIR, "predict.err")
    timed_out = False
    try:
        with open(out_log, "wb") as fout, open(err_log, "wb") as ferr:
            proc = subprocess.Popen(
                cmd,
                cwd="/tmp",
                env=child_env,
                pass_fds=(out_fd,),
                stdin=subprocess.DEVNULL,
                stdout=fout,
                stderr=ferr,
            )
            try:
                returncode = proc.wait(timeout=PREDICT_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                log(f"rank() timed out after {PREDICT_TIMEOUT_SEC:.0f}s; killing it")
                proc.kill()
                proc.wait()
                returncode = 124
                timed_out = True
    except OSError as exc:
        os.close(out_fd)
        return None, f"could not launch the unprivileged predictor: {exc}"
    finally:
        try:
            os.close(out_fd)
        except OSError:
            pass

    stdout = _tail(out_log)
    stderr = _tail(err_log)
    if stdout.strip():
        log("child stdout (last 4000 chars):\n" + stdout)
    if stderr.strip():
        log("child stderr (last 4000 chars):\n" + stderr)

    if timed_out:
        return None, f"rank() exceeded the {PREDICT_TIMEOUT_SEC:.0f}s prediction timeout"

    try:
        with open(out_path, "r") as fh:
            raw = fh.read()
    except OSError as exc:
        return None, f"could not read the child's result: {exc}"

    if not raw.strip():
        return None, f"rank() produced no output (child exit code {returncode})"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"child produced unparsable output: {exc}"

    if not isinstance(result, dict) or not result.get("ok"):
        detail = ""
        if isinstance(result, dict):
            detail = str(result.get("error", ""))
            if result.get("traceback"):
                log("agent traceback:\n" + str(result["traceback"])[:4000])
        return None, f"rank() failed: {detail or 'unknown error'}"
    return result.get("order"), None


def main() -> int:
    harden_log_dir()
    # Pessimistic default: any crash below leaves a zero on disk.
    grade_mod.write_outputs(
        {
            "achieved": 0.0,
            "optimal": 0.0,
            "floor": 0.0,
            "normalised": 0.0,
            "precision_at_k": 0.0,
            "mean_age_of_caught_incidents": None,
            "incidents_caught": 0,
            "incidents_available": 0,
            "k": reward_mod.K_BUDGET,
            "t_max": reward_mod.T_MAX,
            "cost": reward_mod.COST,
            "valid_ordering": False,
            "error": "verifier did not complete",
            "per_period": [],
        }
    )

    harden_tests_dir()
    alerts, k = grade_mod.load_dataset(DATA_PATH)
    log(f"loaded {len(alerts)} hidden alerts, k={k}")

    err = make_solution_readable()
    order: list | None = None
    if err is None:
        payload_path = write_payload(build_payload(alerts))
        log(f"payload written to {payload_path} (no label keys)")
        order, err = run_agent_rank(payload_path)

    if err is not None:
        log(f"FAILED: {err}")
        metrics = reward_mod.score(alerts, [], k=k)
        metrics["error"] = err
        value = grade_mod.write_outputs(metrics)
        log(grade_mod.summary(metrics))
        log(f"reward = {value:.6f}")
        return 0

    metrics = grade_mod.grade(alerts, order, k)
    value = grade_mod.write_outputs(metrics)
    log(grade_mod.summary(metrics))
    log(f"reward = {value:.6f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        # MUST come before the BaseException arm.  `sys.exit(main())` raises
        # SystemExit even on the success path, and SystemExit is a
        # BaseException -- so catching it below would run the crash handler
        # after a perfectly good grade, overwriting the reward.txt that
        # write_outputs() just produced with 0.000000.
        raise
    except BaseException:
        traceback.print_exc()
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "reward.txt"), "w") as fh:
                fh.write("0.000000")
        finally:
            sys.exit(1)
