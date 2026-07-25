"""Regression tests for the in-band exit-status sentinel.

    python harbor_runpod/test_remote_script.py

``_remote_script`` emits plain POSIX sh, so these run it through the local
``/bin/sh`` instead of mocking. That is the whole point: the bug this guards
against was invisible to any test that inspected the generated string rather
than executing it.

Background: RunPod has no exec API, so the remote command's exit status is
carried back in-band by a ``__HARBOR_RUNPOD_RC:<n>:RC__`` sentinel. An agent
that signals its own process group on the way out -- kimi-cli ends its run with
``kill 0`` -- used to take the sentinel's ``printf`` down with it, and exec()
then raised RunPodSSHError on a trial that had actually completed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harbor_runpod.runpod_env import RunPodEnvironment  # noqa: E402

_RC = re.compile(r"__HARBOR_RUNPOD_RC:(-?\d+):RC__")
_FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        _FAILURES.append(label)


def run(command: str) -> tuple[int | None, str]:
    """Build the remote script for *command* and execute it with /bin/sh."""
    env = object.__new__(RunPodEnvironment)
    env._remote_shell = "sh"
    script = env._remote_script(command, cwd=None, env=None, user=None)
    # start_new_session is not optional. The command under test signals its own
    # process group; without a session of its own that signal reaches this test
    # runner and kills it. Against unpatched code the run then dies instead of
    # reporting -- which is exactly what happened the first time, and a test
    # that takes down its harness on failure is worse than no test.
    proc = subprocess.run(
        ["/bin/sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        start_new_session=True,
    )
    match = _RC.search(proc.stdout)
    return (int(match.group(1)) if match else None), proc.stdout


def test_group_signalling_agent() -> None:
    print("agent that ends with `kill 0` (the kimi-cli shape)")
    rc, out = run('echo marker; kill 0 2>/dev/null')
    check(rc is not None, "sentinel survives the process-group signal")
    # 143 = 128 + SIGTERM. Harbor's kimi agent matches on exactly this string
    # and swallows it as normal completion, so the number is load-bearing.
    check(rc == 143, f"status is 143, which Harbor reads as success (got {rc})")
    check("marker" in out, "the agent's own stdout is still captured")


def test_normal_exit_status_unchanged() -> None:
    print("ordinary commands are unaffected")
    for expected in (0, 7, 143):
        rc, _ = run(f"exit {expected}")
        check(rc == expected, f"exit {expected} round-trips (got {rc})")


def test_stdout_precedes_sentinel() -> None:
    print("stdout is not corrupted by the wrapper")
    rc, out = run("printf 'hello world'")
    check(rc == 0, f"status 0 (got {rc})")
    check(out.startswith("hello world"), f"stdout leads the sentinel (got {out!r})")


def main() -> int:
    for test in (
        test_group_signalling_agent,
        test_normal_exit_status_unchanged,
        test_stdout_precedes_sentinel,
    ):
        test()

    print()
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}):")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
