"""Regression tests for the two teardown paths that cost money or corrupt data.

Plain stdlib, no pytest: this package has no test dependencies and these are
the paths that cannot be exercised without either a RunPod account or a fault
injection, so they need to be checkable offline.

    python harbor_runpod/test_teardown.py

Both cases are cancellation paths. ``asyncio.CancelledError`` is a
``BaseException``, so ``except Exception`` does not see it and neither does
``except (TimeoutError, asyncio.TimeoutError)`` -- which is exactly how both
bugs got there in the first place.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harbor_runpod.runpod_env import RunPodEnvironment  # noqa: E402

_LOGGER = logging.getLogger("test_teardown")
_FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        _FAILURES.append(label)


def _bare_env() -> RunPodEnvironment:
    """An instance without ``__init__``.

    ``BaseEnvironment.__init__`` wants a parsed task, a network policy and a
    trial-paths object; none of the code under test touches any of it. Building
    the real thing here would test Harbor's constructor, not these two methods.
    """
    env = object.__new__(RunPodEnvironment)
    env.logger = _LOGGER
    return env


# ---------------------------------------------------------------------------
# 1. _run_process must kill the ssh client when the agent phase is cancelled.
# ---------------------------------------------------------------------------


async def test_run_process_kills_child_on_cancel() -> None:
    print("_run_process: cancellation kills the child")
    env = _bare_env()

    spawned: list[asyncio.subprocess.Process] = []
    real_exec = asyncio.create_subprocess_exec

    async def recording_exec(*args, **kwargs):
        process = await real_exec(*args, **kwargs)
        spawned.append(process)
        return process

    asyncio.create_subprocess_exec = recording_exec  # type: ignore[assignment]
    try:
        # No timeout_sec: this is the agent-phase shape, where the deadline is
        # enforced by Harbor's asyncio.wait_for one level up.
        task = asyncio.ensure_future(env._run_process(["sleep", "60"]))
        while not spawned:
            await asyncio.sleep(0.01)
        pid = spawned[0].pid

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        else:
            cancelled = False
    finally:
        asyncio.create_subprocess_exec = real_exec  # type: ignore[assignment]

    check(cancelled, "CancelledError propagates (caller still sees the timeout)")

    # process.wait() reaped it, so the pid is gone rather than a zombie.
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    except PermissionError:  # pragma: no cover - pid recycled into another uid
        alive = True
    check(not alive, f"child pid {pid} is dead, not left writing during log download")


async def test_run_process_still_returns_normally() -> None:
    print("_run_process: the uncancelled path is unchanged")
    env = _bare_env()
    code, stdout, _ = await env._run_process(["sh", "-c", "printf hello; exit 3"])
    check(code == 3, f"exit code preserved (got {code})")
    check(stdout == b"hello", f"stdout preserved (got {stdout!r})")


# ---------------------------------------------------------------------------
# 2. stop() must reconcile a pod that create_pod made but never reported.
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, pods: list[dict], *, list_raises: bool = False) -> None:
        self._pods = pods
        self._list_raises = list_raises
        self.deleted: list[str] = []

    async def list_pods(self, **_filters):
        if self._list_raises:
            raise RuntimeError("RunPod API unreachable")
        return self._pods

    async def delete_pod(self, pod_id: str) -> None:
        self.deleted.append(pod_id)


def _leaked_env(client: FakeClient, session_id: str = "abc") -> RunPodEnvironment:
    env = _bare_env()
    env.session_id = session_id
    env._client = client
    env._pod_id = ""  # the leak: created remotely, never recorded locally
    env._ssh_host = None
    env._ssh_port = None
    env._ssh_key_dir = None
    env._ssh_key = None
    return env


async def test_stop_terminates_leaked_pod() -> None:
    print("stop(): a pod created but never recorded is found by name")
    client = FakeClient([{"name": "harbor-abc", "id": "pod-123"}])
    await _leaked_env(client).stop(delete=True)
    check(client.deleted == ["pod-123"], f"leaked pod terminated (got {client.deleted})")


async def test_stop_does_not_touch_other_pods() -> None:
    print("stop(): a sibling's pod is never terminated")
    # The separate verifier env gets its own session_id, so its pod must be
    # invisible here. Over-reach would be far worse than the leak.
    client = FakeClient(
        [
            {"name": "harbor-someone-else", "id": "pod-999"},
            {"name": "harbor-abc-verifier", "id": "pod-888"},
        ]
    )
    await _leaked_env(client).stop(delete=True)
    check(client.deleted == [], f"no pod terminated (got {client.deleted})")


async def test_stop_survives_list_failure() -> None:
    print("stop(): a failed reconciliation does not raise")
    # stop() runs from a finally that is usually already unwinding an
    # exception; raising here would replace the error the caller reports.
    client = FakeClient([], list_raises=True)
    try:
        await _leaked_env(client).stop(delete=True)
        raised = False
    except Exception as exc:  # noqa: BLE001
        raised = True
        print(f"    unexpected: {exc!r}")
    check(not raised, "teardown swallowed the API failure and warned instead")


async def main() -> int:
    for test in (
        test_run_process_kills_child_on_cancel,
        test_run_process_still_returns_normally,
        test_stop_terminates_leaked_pod,
        test_stop_does_not_touch_other_pods,
        test_stop_survives_list_failure,
    ):
        await test()

    print()
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}):")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)  # the warnings are expected
    sys.exit(asyncio.run(main()))
