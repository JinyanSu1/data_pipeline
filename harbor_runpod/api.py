"""A thin, dependency-free client for the RunPod public REST API.

Only the stdlib is used (``urllib.request``), so this module can be imported
in environments where ``requests``/``httpx`` are unavailable. Blocking HTTP
calls are pushed onto a worker thread so the client presents an ``async``
surface consistent with the rest of Harbor's environment providers.

Every endpoint below was taken from the live OpenAPI document published at
``https://rest.runpod.io/v1/openapi.json`` (``info.title = "Runpod API"``),
not from prose documentation:

===============================  =========================================
Operation                        Endpoint
===============================  =========================================
create pod                       ``POST   /v1/pods``           -> 201 Pod
get pod                          ``GET    /v1/pods/{podId}``   -> 200 Pod
list pods                        ``GET    /v1/pods``           -> 200 Pods
stop pod (keep disk)             ``POST   /v1/pods/{podId}/stop``
start a stopped pod              ``POST   /v1/pods/{podId}/start``
terminate pod (delete)           ``DELETE /v1/pods/{podId}``   -> 204
list registry credentials        ``GET    /v1/containerregistryauth``
create registry credentials      ``POST   /v1/containerregistryauth``
delete registry credentials      ``DELETE /v1/containerregistryauth/{id}``
list network volumes             ``GET    /v1/networkvolumes``
===============================  =========================================

Authentication is HTTP bearer (``components.securitySchemes.ApiKey`` is
``{"type": "http", "scheme": "bearer"}``), i.e. ``Authorization: Bearer <key>``.

Why v1 and not v2
-----------------
RunPod's docs say v1 "is deprecated and will be retired in the near future"
and that "all new integrations should use REST API v2 instead"
(https://docs.runpod.io/api-reference/overview). This client still targets v1
deliberately, because v2's ``CreatePodRequest`` cannot express three things
this provider depends on:

* **no ``dockerEntrypoint``** — v2 offers only ``args`` (the CMD). Harbor's
  SSH bootstrap has to become PID 1 *regardless* of the task image's
  ``ENTRYPOINT``, which requires overriding the entrypoint.
* **no ``supportPublicIp``** — direct-TCP SSH (the only mode that can move
  files; the ``ssh.runpod.io`` proxy rejects SCP/SFTP) needs a public IP.
* **no ``interruptible``** — spot pods, the main cost lever, are v1-only.

``base_url`` is a constructor argument, so migrating is a one-line change plus
a field remap once v2 grows those fields. Verified against both live specs:
``https://rest.runpod.io/v1/openapi.json`` and
``https://api.runpod.io/v2/openapi.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_BASE_URL = "https://rest.runpod.io/v1"

# Verified against the OpenAPI document (PodCreateInput / Pod schemas).
PODS_PATH = "/pods"
POD_PATH = "/pods/{pod_id}"
POD_STOP_PATH = "/pods/{pod_id}/stop"
POD_START_PATH = "/pods/{pod_id}/start"
CONTAINER_REGISTRY_AUTH_PATH = "/containerregistryauth"
CONTAINER_REGISTRY_AUTH_ITEM_PATH = "/containerregistryauth/{auth_id}"
NETWORK_VOLUMES_PATH = "/networkvolumes"

# ``Pod.desiredStatus`` enum from the OpenAPI document.
POD_STATUS_RUNNING = "RUNNING"
POD_STATUS_EXITED = "EXITED"
POD_STATUS_TERMINATED = "TERMINATED"

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_DEFAULT_TIMEOUT_SEC = 60.0
_DEFAULT_MAX_ATTEMPTS = 4

logger = logging.getLogger(__name__)


class RunPodAPIError(RuntimeError):
    """A non-2xx response (or an unparseable body) from the RunPod REST API."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        method: str | None = None,
        path: str | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.method = method
        self.path = path
        self.body = body

    @property
    def is_retryable(self) -> bool:
        return self.status is None or self.status in _RETRYABLE_STATUS

    @property
    def is_not_found(self) -> bool:
        return self.status == 404


class RunPodTransport:
    """The single seam between :class:`RunPodClient` and the network.

    Tests replace this object wholesale, so no HTTP traffic is required to
    exercise the client or the environment that sits on top of it.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        user_agent: str = "harbor-runpod/0.1 (+stdlib-urllib)",
    ) -> None:
        if not api_key:
            raise ValueError("A RunPod API key is required.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent

    @property
    def base_url(self) -> str:
        return self._base_url

    def _build_url(self, path: str, query: Mapping[str, Any] | None) -> str:
        url = f"{self._base_url}{path}"
        if not query:
            return url
        pairs: list[tuple[str, str]] = []
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, bool):
                pairs.append((key, "true" if value else "false"))
            elif isinstance(value, (list, tuple)):
                pairs.extend((key, str(item)) for item in value)
            else:
                pairs.append((key, str(value)))
        if not pairs:
            return url
        return f"{url}?{urllib.parse.urlencode(pairs)}"

    def request_sync(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Perform one blocking request and return the decoded JSON body.

        Returns ``None`` for empty bodies (e.g. the ``204`` from DELETE /pods).
        """
        url = self._build_url(path, query)
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url, data=payload, headers=headers, method=method.upper()
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read() or b""
            text = raw.decode("utf-8", errors="replace")
            raise RunPodAPIError(
                f"RunPod API {method.upper()} {path} failed with HTTP "
                f"{exc.code}: {text[:2000]}",
                status=exc.code,
                method=method.upper(),
                path=path,
                body=text,
            ) from exc
        except urllib.error.URLError as exc:
            raise RunPodAPIError(
                f"RunPod API {method.upper()} {path} could not be reached: "
                f"{exc.reason}",
                status=None,
                method=method.upper(),
                path=path,
            ) from exc

        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunPodAPIError(
                f"RunPod API {method.upper()} {path} returned HTTP {status} with a "
                "body that is not valid JSON.",
                status=status,
                method=method.upper(),
                path=path,
                body=raw.decode("utf-8", errors="replace"),
            ) from exc

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self.request_sync, method, path, body=body, query=query
        )


class RunPodClient:
    """Typed-ish wrappers over the handful of RunPod endpoints Harbor needs."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: RunPodTransport | None = None,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        logger_: logging.Logger | None = None,
    ) -> None:
        if transport is None:
            transport = RunPodTransport(
                api_key or "", base_url=base_url, timeout_sec=timeout_sec
            )
        self._transport = transport
        self._max_attempts = max(1, max_attempts)
        self._logger = logger_ or logger

    @property
    def transport(self) -> RunPodTransport:
        return self._transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Issue a request, retrying only transport/throttling failures.

        Mutating calls are retried too: ``POST /pods`` is the only
        non-idempotent one, and a retry there can leak a pod, so it is
        retried *only* when the failure proves the request never produced a
        pod (connection error or 429/5xx before allocation). Callers that
        must not double-create pass ``max_attempts=1`` at construction.
        """
        last_error: RunPodAPIError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._transport.request(
                    method, path, body=body, query=query
                )
            except RunPodAPIError as exc:
                if not exc.is_retryable or attempt == self._max_attempts:
                    raise
                last_error = exc
                delay = min(2 ** (attempt - 1), 8) * (0.5 + random.random())
                self._logger.warning(
                    "RunPod %s %s failed (status=%s, attempt %d/%d); retrying in %.1fs",
                    method.upper(),
                    path,
                    exc.status,
                    attempt,
                    self._max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_error is not None  # unreachable; loop either returns or raises
        raise last_error

    # ------------------------------------------------------------------
    # Pods
    # ------------------------------------------------------------------

    async def create_pod(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /pods`` -> ``201`` with a ``Pod`` body."""
        result = await self._request("POST", PODS_PATH, body=payload)
        if not isinstance(result, dict):
            raise RunPodAPIError(
                f"RunPod POST {PODS_PATH} returned {type(result).__name__}, "
                "expected a Pod object.",
                method="POST",
                path=PODS_PATH,
            )
        return result

    async def get_pod(
        self,
        pod_id: str,
        *,
        include_machine: bool = False,
        include_network_volume: bool = False,
    ) -> dict[str, Any]:
        """``GET /pods/{podId}`` -> ``200`` with a ``Pod`` body."""
        path = POD_PATH.format(pod_id=pod_id)
        query: dict[str, Any] = {}
        if include_machine:
            query["includeMachine"] = True
        if include_network_volume:
            query["includeNetworkVolume"] = True
        result = await self._request("GET", path, query=query or None)
        if not isinstance(result, dict):
            raise RunPodAPIError(
                f"RunPod GET {path} returned {type(result).__name__}, expected a Pod.",
                method="GET",
                path=path,
            )
        return result

    async def list_pods(self, **filters: Any) -> list[dict[str, Any]]:
        """``GET /pods`` -> ``200`` with a ``Pods`` array."""
        result = await self._request("GET", PODS_PATH, query=filters or None)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Defensive: tolerate a future envelope without breaking callers.
            pods = result.get("pods")
            if isinstance(pods, list):
                return pods
        return []

    async def stop_pod(self, pod_id: str) -> None:
        """``POST /pods/{podId}/stop`` — halts the container, keeps the disk.

        Storage is still billed while a pod is stopped.
        """
        await self._request("POST", POD_STOP_PATH.format(pod_id=pod_id))

    async def start_pod(self, pod_id: str) -> None:
        """``POST /pods/{podId}/start`` — resume a stopped pod."""
        await self._request("POST", POD_START_PATH.format(pod_id=pod_id))

    async def delete_pod(self, pod_id: str) -> None:
        """``DELETE /pods/{podId}`` -> ``204``. Terminates and releases storage."""
        await self._request("DELETE", POD_PATH.format(pod_id=pod_id))

    # ------------------------------------------------------------------
    # Container registry credentials
    # ------------------------------------------------------------------

    async def list_container_registry_auths(self) -> list[dict[str, Any]]:
        """``GET /containerregistryauth`` -> ``ContainerRegistryAuths``."""
        result = await self._request("GET", CONTAINER_REGISTRY_AUTH_PATH)
        return result if isinstance(result, list) else []

    async def create_container_registry_auth(
        self, *, name: str, username: str, password: str
    ) -> dict[str, Any]:
        """``POST /containerregistryauth``.

        The created record's ``id`` is what ``PodCreateInput.containerRegistryAuthId``
        expects. RunPod requires ``name`` to be unique across the account.
        """
        result = await self._request(
            "POST",
            CONTAINER_REGISTRY_AUTH_PATH,
            body={"name": name, "username": username, "password": password},
        )
        if not isinstance(result, dict):
            raise RunPodAPIError(
                f"RunPod POST {CONTAINER_REGISTRY_AUTH_PATH} returned "
                f"{type(result).__name__}, expected a ContainerRegistryAuth.",
                method="POST",
                path=CONTAINER_REGISTRY_AUTH_PATH,
            )
        return result

    async def delete_container_registry_auth(self, auth_id: str) -> None:
        await self._request(
            "DELETE", CONTAINER_REGISTRY_AUTH_ITEM_PATH.format(auth_id=auth_id)
        )

    async def ensure_container_registry_auth(
        self, *, name: str, username: str, password: str
    ) -> str:
        """Return the id of a registry-auth record named *name*, creating it once.

        RunPod never returns stored credentials, so an existing record with the
        right name is reused as-is; rotate by deleting it in the RunPod console
        (or via :meth:`delete_container_registry_auth`) and re-running.
        """
        for existing in await self.list_container_registry_auths():
            if existing.get("name") == name and existing.get("id"):
                return str(existing["id"])
        created = await self.create_container_registry_auth(
            name=name, username=username, password=password
        )
        auth_id = created.get("id")
        if not auth_id:
            raise RunPodAPIError(
                "RunPod created a container registry auth without returning an id.",
                method="POST",
                path=CONTAINER_REGISTRY_AUTH_PATH,
            )
        return str(auth_id)

    # ------------------------------------------------------------------
    # Network volumes
    # ------------------------------------------------------------------

    async def list_network_volumes(self) -> list[dict[str, Any]]:
        """``GET /networkvolumes``. Each entry carries a ``dataCenterId``."""
        result = await self._request("GET", NETWORK_VOLUMES_PATH)
        return result if isinstance(result, list) else []


# ----------------------------------------------------------------------
# Pod response helpers
#
# These read the ``Pod`` schema exactly as published: ``publicIp`` is a
# nullable string, ``portMappings`` is a nullable object whose example is
# ``{"22": 10341}`` (private port -> public port) and which "is not yet
# determined and will be empty" while the pod initializes.
# ----------------------------------------------------------------------


def pod_ssh_endpoint(pod: Mapping[str, Any]) -> tuple[str, int] | None:
    """Return ``(host, port)`` for direct-TCP SSH into *pod*, or ``None``.

    ``None`` means "not ready yet" rather than "unsupported": both the public
    IP and the port mapping are populated asynchronously by RunPod after the
    pod is created.
    """
    host = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    if not host or not isinstance(mappings, Mapping):
        return None
    port = mappings.get("22", mappings.get(22))
    if port is None:
        return None
    try:
        return str(host), int(port)
    except (TypeError, ValueError):
        return None


def pod_is_terminal(pod: Mapping[str, Any]) -> bool:
    """True when RunPod will never bring this pod up (EXITED/TERMINATED)."""
    return pod.get("desiredStatus") in (POD_STATUS_EXITED, POD_STATUS_TERMINATED)


def format_ports(ports: Sequence[str]) -> list[str]:
    """Normalize a ``ports`` list, guaranteeing ``22/tcp`` is present.

    ``PodCreateInput.ports`` entries are ``"<port>/<protocol>"`` where protocol
    is ``http`` or ``tcp``; SSH needs the raw TCP form.
    """
    normalized = [str(port).strip() for port in ports if str(port).strip()]
    if "22/tcp" not in normalized:
        normalized.insert(0, "22/tcp")
    return normalized
