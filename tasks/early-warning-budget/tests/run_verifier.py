#!/usr/bin/env python3
"""Root-side verifier for the `early-warning-budget` task.

Security shape (this is the whole point of the file):

  * This process runs as root and NEVER imports, execs or evals the agent's
    code.  It only writes JSON requests to a pipe and reads JSON answers back.
  * The agent's `Detector` is hosted by `/tests/predict.py`, launched through
    `setpriv --reuid nobody --regid nogroup --clear-groups`.  That child has no
    write access anywhere that matters and no access to the labels.
  * `/logs/verifier` is chmod 700 root for the whole time the untrusted child
    is alive, so a forged `reward.txt` (including from a detached `os.fork()`
    daemon that outlives the child) is impossible.  It is relaxed to 755 only
    after the child is dead and the real reward has been written, because the
    directory is a bind mount that the harness has to read back from the host.
  * `/tests/test_episodes.json` -- the ground truth -- is chmod 600 root.
  * The observations are STREAMED to the child one timestep at a time and the
    decision for step t is read back before step t+1 is sent.  The child never
    holds a future sample in memory, so lookahead is structurally impossible
    rather than merely forbidden by the instructions.

Output:
  /logs/verifier/reward.txt    scalar float, always written
  /logs/verifier/metrics.json  untransformed diagnostics
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/tests")

from grade import grade  # noqa: E402
from reward import FPR_LIMIT, L_MAX, compute_reward  # noqa: E402

TESTS_DIR = Path("/tests")
EPISODES_PATH = TESTS_DIR / "test_episodes.json"
PREDICT_PATH = TESTS_DIR / "predict.py"
AGENT_SOLUTION = Path("/app/solution.py")
LOGS_DIR = Path("/logs/verifier")
REWARD_PATH = LOGS_DIR / "reward.txt"
METRICS_PATH = LOGS_DIR / "metrics.json"

PREDICT_TIMEOUT_SEC = 900.0


class PredictTimeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise PredictTimeout("prediction phase exceeded its time budget")


def write_reward(value: float) -> None:
    tmp = REWARD_PATH.with_suffix(".tmp")
    tmp.write_text(f"{float(value):.6f}\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, REWARD_PATH)


def label_free_payload(episodes: list[dict]) -> list[dict]:
    """Strip every label.  Only these keys are ever sent to the child."""
    return [
        {
            "episode_id": ep["episode_id"],
            "n_channels": int(ep["n_channels"]),
            "groups": [list(g) for g in ep["groups"]],
            "x": ep["x"],
        }
        for ep in episodes
    ]


def build_child_command(solution_copy: Path) -> tuple[list[str], bool]:
    """setpriv drop to nobody:nogroup; report whether the drop is real."""
    setpriv = shutil.which("setpriv")
    if setpriv and os.geteuid() == 0:
        return (
            [
                setpriv,
                "--reuid", "nobody",
                "--regid", "nogroup",
                "--clear-groups",
                sys.executable,
                str(PREDICT_PATH),
                str(solution_copy),
            ],
            True,
        )
    # Never silently run agent code as root in the real container.
    return ([sys.executable, str(PREDICT_PATH), str(solution_copy)], False)


class Child:
    def __init__(self, cmd: list[str], cwd: Path, stderr_file):
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/nonexistent",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C",
        }
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            cwd=str(cwd),
            env=env,
            close_fds=True,
        )

    def exchange(self, request: dict | None) -> dict:
        if request is not None:
            payload = (json.dumps(request, separators=(",", ":")) + "\n").encode()
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("detector process closed the pipe")
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        for closer in (self.proc.stdin, self.proc.stdout):
            try:
                closer.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass


def run_online(payload: list[dict], solution_copy: Path, workdir: Path,
               notes: list[str]) -> dict[str, list[int]]:
    """Drive the agent's Detector in lockstep.  Returns episode_id -> 0/1 list."""
    cmd, dropped = build_child_command(solution_copy)
    if not dropped:
        notes.append(
            "WARNING: could not drop privileges (setpriv missing or not root); "
            "agent code ran with the verifier's own privileges."
        )

    decisions: dict[str, list[int]] = {ep["episode_id"]: [0] * len(ep["x"])
                                       for ep in payload}

    stderr_path = workdir / "detector_stderr.log"
    started = time.time()
    with stderr_path.open("wb") as errf:
        child = Child(cmd, workdir, errf)
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, PREDICT_TIMEOUT_SEC)
        try:
            hello = child.exchange(None)
            if not hello.get("ok"):
                notes.append("detector import failed: "
                             + str(hello.get("error"))[:1500])
                return decisions

            for ep in payload:
                eid = ep["episode_id"]
                ack = child.exchange({
                    "cmd": "init",
                    "n_channels": ep["n_channels"],
                    "groups": ep["groups"],
                })
                if not ack.get("ok"):
                    notes.append(f"{eid}: Detector.__init__ failed: "
                                 + str(ack.get("error"))[:600])
                    child.exchange({"cmd": "end"})
                    continue

                row_decisions = decisions[eid]
                for t, row in enumerate(ep["x"]):
                    reply = child.exchange({"cmd": "step", "x": row})
                    row_decisions[t] = 1 if reply.get("d") == 1 else 0

                done = child.exchange({"cmd": "end"})
                if done.get("error"):
                    notes.append(f"{eid}: Detector.step raised: "
                                 + str(done.get("error"))[:600])

            child.exchange({"cmd": "done"})
        except PredictTimeout:
            notes.append(
                f"detector timed out after {PREDICT_TIMEOUT_SEC:.0f}s; "
                "remaining timesteps scored as no-alarm"
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"detector transport error: {type(exc).__name__}: {exc}")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            child.close()

    notes.append(f"prediction wall time: {time.time() - started:.1f}s")
    tail = stderr_path.read_bytes()[-2000:].decode("utf-8", "replace").strip()
    if tail:
        notes.append("detector stderr (tail): " + tail)
    return decisions


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Lock the reward channel down BEFORE any untrusted code can exist, and
    # drop anything the agent may have planted during its own phase.
    os.chmod(LOGS_DIR, 0o700)
    for stale in ("reward.json", "rewards.json", "metrics.json"):
        try:
            (LOGS_DIR / stale).unlink()
        except FileNotFoundError:
            pass
    write_reward(0.0)

    notes: list[str] = []
    metrics: dict = {
        "reward": 0.0,
        "mean_lead_steps": 0.0,
        "fpr": 0.0,
        "n_never_alarmed": 0,
        "per_episode_lead": [],
        "l_max": L_MAX,
        "fpr_limit": FPR_LIMIT,
        "gate_passed": False,
        "notes": notes,
    }
    reward = 0.0

    workdir = None
    try:
        os.chmod(EPISODES_PATH, 0o600)
        episodes = json.loads(EPISODES_PATH.read_text())

        if not AGENT_SOLUTION.is_file():
            notes.append(f"{AGENT_SOLUTION} not found; scoring as never-alarmed")
            decisions: dict[str, list[int]] = {}
        else:
            workdir = Path(tempfile.mkdtemp(prefix="ewb_", dir="/tmp"))
            os.chmod(workdir, 0o755)  # child (nobody) may read, may not write
            solution_copy = workdir / "agent_solution.py"
            shutil.copyfile(AGENT_SOLUTION, solution_copy)
            os.chmod(solution_copy, 0o444)
            notes.append(
                f"agent solution: {AGENT_SOLUTION.stat().st_size} bytes")

            payload = label_free_payload(episodes)
            decisions = run_online(payload, solution_copy, workdir, notes)
            del payload

        result = grade(episodes, decisions)
        reward = compute_reward(result["mean_lead_steps"], result["fpr"])

        metrics.update({
            "reward": reward,
            "mean_lead_steps": result["mean_lead_steps"],
            "fpr": result["fpr"],
            "n_never_alarmed": result["n_never_alarmed"],
            "per_episode_lead": result["per_episode_lead"],
            "n_scored_episodes": result["n_scored_episodes"],
            "calm_steps_total": result["calm_steps_total"],
            "calm_alarms_total": result["calm_alarms_total"],
            "gate_passed": result["fpr"] <= FPR_LIMIT,
            "per_episode": result["per_episode"],
        })
    except Exception as exc:  # noqa: BLE001
        import traceback
        notes.append("verifier error: " + traceback.format_exc(limit=10)[:4000])
        reward = 0.0
        metrics["reward"] = 0.0
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
        try:
            METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
            os.chmod(METRICS_PATH, 0o644)
        except Exception:
            pass
        write_reward(reward)
        # The untrusted child is gone; reopen the directory so the harness can
        # read reward.txt back off the bind mount as a non-root host user.
        os.chmod(LOGS_DIR, 0o755)

    print(f"reward={reward:.6f} "
          f"mean_lead={metrics['mean_lead_steps']:.3f} "
          f"fpr={metrics['fpr']:.5f} "
          f"never_alarmed={metrics['n_never_alarmed']}")
    for note in notes:
        print("note:", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
