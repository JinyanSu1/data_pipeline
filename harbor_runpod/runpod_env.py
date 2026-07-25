"""A Harbor environment provider backed by RunPod On-Demand Pods.

RunPod is not a sandbox API: it rents a container on a GPU/CPU host and gives
you nothing but a network endpoint. Three consequences shape this module.

1. **No build service.** RunPod only *pulls* an image from a registry. Harbor's
   Docker backend builds ``environment/Dockerfile`` locally. This provider
   bridges the gap by building locally and pushing to a registry you configure
   (:func:`RunPodEnvironment._resolve_image`), tagging by Harbor's own
   ``environment_id`` content hash so repeated runs reuse the pushed image.

2. **No exec API.** RunPod has no "run this command in my pod" endpoint. The
   only general-purpose channel is SSH, so every ``exec``/upload/download here
   is an ``ssh`` subprocess against the pod's direct-TCP endpoint
   (``publicIp`` + ``portMappings["22"]``). File transfer is a single streamed
   tar per call — no scp, which also keeps the code working on pods where only
   an exec channel is available.

3. **One container per pod.** ``environment/docker-compose.yaml`` tasks are
   rejected up front (see :meth:`_validate_definition` and ``capabilities``)
   rather than silently degraded to the first service.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

from harbor.environments.base import (
    BaseEnvironment,
    ExecResult,
    SandboxBuildFailedError,
    ServiceOperationsUnsupportedError,
)
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.definition import (
    COMPOSE_FILE_NAME,
    DOCKERFILE_NAME,
    SNAPSHOT_HASH_LEN,
    effective_exec_cwd,
    parse_dockerfile_workdir,
    require_agent_environment_definition,
    should_use_prebuilt_docker_image,
)
from harbor.environments.tar_transfer import extract_dir, pack_dir
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
    TaskOS,
    classify_network_allowlist_entry,
    NetworkAllowlistEntryType,
)
from harbor.models.trial.paths import TrialPaths

from harbor_runpod.api import (
    RunPodAPIError,
    RunPodClient,
    format_ports,
    pod_is_terminal,
    pod_ssh_endpoint,
)

ENVIRONMENT_TYPE = "runpod"

# Bumping this invalidates every pushed image tag, because the SSH layer we
# bake into the image is part of what the tag identifies.
_IMAGE_LAYER_VERSION = "v1"

_DEFAULT_START_TIMEOUT_SEC = 900
_DEFAULT_SSH_CONNECT_TIMEOUT_SEC = 15
_DEFAULT_EXEC_TIMEOUT_SEC = 60 * 60 * 24
_POD_POLL_INTERVAL_SEC = 5.0
_SSH_POLL_INTERVAL_SEC = 5.0

# Wraps every remote command so the real exit status survives the SSH
# transport. ``ssh`` itself exits 255 on connection failure, which is
# indistinguishable from a command that happens to exit 255; the sentinel
# removes the ambiguity, and its absence is proof of a transport failure.
_RC_PREFIX = "__HARBOR_RUNPOD_RC:"
_RC_SUFFIX = ":RC__"
_RC_PATTERN = re.compile(
    re.escape(_RC_PREFIX) + r"(\d+)" + re.escape(_RC_SUFFIX) + r"\s*\Z"
)

_TIMEOUT_RETURN_CODE = 124  # matches coreutils `timeout`

# A non-interactive `ssh host cmd` session inherits sshd's environment, not the
# container's, so everything RunPod injected via PodCreateInput.env and
# everything the image set with ENV would be invisible - including PATH. PID 1
# is our bootstrap, so /proc/1/environ is exactly the container environment.
# (`runpodctl exec` does the same thing for the same reason.) Harbor's own
# env vars are exported after this block so they win on conflict.
_CONTAINER_ENV_IMPORT = r"""
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' __hb_env; do
    case "$__hb_env" in
      PWD=*|OLDPWD=*|SHLVL=*|_=*|PUBLIC_KEY=*) continue ;;
      *=*) export "$__hb_env" ;;
    esac
  done < /proc/1/environ 2>/dev/null
  unset __hb_env
fi
""".strip()

NetworkEnforcement = Literal["none", "best-effort-iptables"]

# Fields of ``PodCreateInput`` a caller may override verbatim via
# ``--environment-kwarg pod_overrides='{"minDownloadMbps": 500}'``. Kept as a
# passthrough so new RunPod fields do not require a code change here.
_POD_CREATE_KNOWN_FIELDS = frozenset(
    {
        "allowedCudaVersions",
        "cloudType",
        "computeType",
        "containerDiskInGb",
        "containerRegistryAuthId",
        "countryCodes",
        "cpuFlavorIds",
        "cpuFlavorPriority",
        "dataCenterIds",
        "dataCenterPriority",
        "dockerEntrypoint",
        "dockerStartCmd",
        "env",
        "globalNetworking",
        "gpuCount",
        "gpuTypeIds",
        "gpuTypePriority",
        "imageName",
        "interruptible",
        "locked",
        "minDiskBandwidthMBps",
        "minDownloadMbps",
        "minRAMPerGPU",
        "minUploadMbps",
        "minVCPUPerGPU",
        "name",
        "networkVolumeId",
        "ports",
        "supportPublicIp",
        "templateId",
        "vcpuCount",
        "volumeInGb",
        "volumeMountPath",
    }
)


class RunPodConfigurationError(RuntimeError):
    """Raised for a missing or contradictory provider configuration.

    Always raised before any billable RunPod call, so a misconfigured run
    costs nothing.
    """


class RunPodSSHError(RuntimeError):
    """The SSH transport to the pod failed (as opposed to the command failing)."""


def _bootstrap_script() -> str:
    """The pod's PID 1: install the trial's public key, then run sshd.

    Overriding the image's start command is deliberate. Harbor drives the
    container through ``exec``; the task's own CMD (``sleep infinity`` in every
    compose-free Harbor task) contributes nothing, and replacing it with sshd
    means the container's lifetime is exactly the lifetime of our control
    channel.

    Written in POSIX ``sh``, not bash: the base image may be Alpine or
    distroless-adjacent. Every optional step is guarded so a missing tool
    degrades instead of aborting PID 1.
    """
    return r"""
set -u
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

SSHD=""
if [ -x /usr/sbin/sshd ]; then SSHD=/usr/sbin/sshd; fi
if [ -z "$SSHD" ]; then SSHD=$(command -v sshd 2>/dev/null || true); fi
if [ -z "$SSHD" ]; then
  # Not baked into the image (prebuilt [environment].docker_image path).
  # Try the distro package manager; this needs egress at boot.
  if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      openssh-server >/dev/null 2>&1 || true
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache openssh-server >/dev/null 2>&1 || true
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y openssh-server >/dev/null 2>&1 || true
  elif command -v yum >/dev/null 2>&1; then
    yum install -y openssh-server >/dev/null 2>&1 || true
  fi
  if [ -x /usr/sbin/sshd ]; then SSHD=/usr/sbin/sshd; fi
  if [ -z "$SSHD" ]; then SSHD=$(command -v sshd 2>/dev/null || true); fi
fi

if [ -z "$SSHD" ]; then
  echo "harbor-runpod: image has no sshd and none could be installed." >&2
  echo "harbor-runpod: build the image through this provider, or bake" >&2
  echo "harbor-runpod: openssh-server into [environment].docker_image." >&2
  # Stay up so the failure is visible in the RunPod console instead of
  # looking like a crash-loop.
  while true; do sleep 3600; done
fi

mkdir -p /run/sshd /var/run/sshd
ssh-keygen -A >/dev/null 2>&1 || true
CONF=/etc/ssh/sshd_config
[ -f "$CONF" ] || : > "$CONF"
# OpenSSH takes the FIRST occurrence of a keyword, so drop existing ones
# before appending ours.
sed -i -E '/^[[:space:]]*#?[[:space:]]*(PermitRootLogin|PasswordAuthentication|PubkeyAuthentication|UseDNS|PermitUserEnvironment)[[:space:]]/d' \
  "$CONF" >/dev/null 2>&1 || true
{
  echo 'PermitRootLogin prohibit-password'
  echo 'PasswordAuthentication no'
  echo 'PubkeyAuthentication yes'
  echo 'UseDNS no'
} >> "$CONF"
exec "$SSHD" -D -e
""".strip()


def _ssh_layer_dockerfile(base_image: str) -> str:
    """A one-``RUN`` layer that guarantees the pushed image can host sshd."""
    return f"""# Generated by harbor_runpod ({_IMAGE_LAYER_VERSION}).
# The task image is used verbatim; this layer only adds the SSH control
# channel RunPod requires, because RunPod has no exec API of its own.
FROM {base_image}
USER root
RUN set -e; \\
    if [ ! -x /usr/sbin/sshd ] && ! command -v sshd >/dev/null 2>&1; then \\
      if command -v apt-get >/dev/null 2>&1; then \\
        DEBIAN_FRONTEND=noninteractive apt-get update && \\
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
          openssh-server ca-certificates iptables && \\
        rm -rf /var/lib/apt/lists/*; \\
      elif command -v apk >/dev/null 2>&1; then \\
        apk add --no-cache openssh-server ca-certificates iptables; \\
      elif command -v dnf >/dev/null 2>&1; then \\
        dnf install -y openssh-server iptables && dnf clean all; \\
      elif command -v yum >/dev/null 2>&1; then \\
        yum install -y openssh-server iptables && yum clean all; \\
      else \\
        echo "harbor-runpod: no supported package manager for openssh-server" >&2; \\
        exit 1; \\
      fi; \\
    fi
EXPOSE 22
"""


def _sanitize_image_name(name: str) -> str:
    """Reduce a Harbor environment name to a legal Docker repository path."""
    lowered = name.lower().replace("/", "-")
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", lowered).strip("-._")
    return cleaned or "harbor-task"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
    else:
        parts = [str(item).strip() for item in value]
    parts = [item for item in parts if item]
    return parts or None


class RunPodEnvironment(BaseEnvironment):
    """Runs a Harbor task inside a single RunPod On-Demand (or spot) Pod."""

    @classmethod
    @override
    def preflight(cls) -> None:
        """Fail the whole job before any trial is queued if we cannot possibly run.

        Only checks that cost nothing and cannot be wrong: the API key, and the
        local ``ssh`` client that is this provider's only control channel.
        """
        if not os.environ.get("RUNPOD_API_KEY"):
            raise SystemExit(
                "RunPod requires RUNPOD_API_KEY to be set. Create a key at "
                "https://console.runpod.io/user/settings and export it."
            )
        for binary in ("ssh", "ssh-keygen"):
            if shutil.which(binary) is None:
                raise SystemExit(
                    f"RunPod environment requires the {binary!r} binary on PATH: "
                    "RunPod has no exec API, so Harbor drives pods over SSH."
                )
        # Harbor imports this lazily inside parse_dockerfile_workdir, which
        # __init__ calls -- and Harbor only installs it with the extras for the
        # remote backends. A venv that runs `--env docker` fine can still be
        # missing it, in which case every trial dies at construction with a
        # bare ModuleNotFoundError. Say so here instead, once, for free.
        try:
            import dockerfile_parse  # noqa: F401,PLC0415
        except ImportError as exc:
            raise SystemExit(
                "RunPod environment requires the 'dockerfile-parse' package "
                "(Harbor uses it to read WORKDIR out of a task's Dockerfile, "
                "and only installs it with the extras for remote backends). "
                "Install it with: pip install 'dockerfile-parse>=2.0.1'"
            ) from exc

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        api_key: str | None = None,
        image: str | None = None,
        image_registry: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
        gpu_type_ids: str | list[str] | None = None,
        cloud_type: str | None = None,
        data_center_ids: str | list[str] | None = None,
        network_volume_id: str | None = None,
        volume_in_gb: int | None = None,
        container_disk_in_gb: int | None = None,
        interruptible: bool | None = None,
        ssh_key_path: str | Path | None = None,
        ssh_user: str = "root",
        remote_shell: str = "bash",
        network_enforcement: NetworkEnforcement | None = None,
        start_timeout_sec: int | None = None,
        pod_overrides: dict[str, Any] | None = None,
        **kwargs,
    ):
        # ``capabilities`` is read during ``BaseEnvironment.__init__`` by the
        # network/GPU validators, so anything it depends on must exist first.
        self._network_enforcement: NetworkEnforcement = self._resolve_enforcement(
            network_enforcement
        )

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._image_override = image or os.environ.get("HARBOR_RUNPOD_IMAGE")
        self._image_registry = image_registry or os.environ.get(
            "HARBOR_RUNPOD_IMAGE_REGISTRY"
        )
        self._registry_username = registry_username or os.environ.get(
            "HARBOR_RUNPOD_REGISTRY_USERNAME"
        )
        self._registry_password = registry_password or os.environ.get(
            "HARBOR_RUNPOD_REGISTRY_PASSWORD"
        )
        self._gpu_type_ids = _split_list(gpu_type_ids) or _split_list(
            os.environ.get("HARBOR_RUNPOD_GPU_TYPE_IDS")
        )
        self._cloud_type = (
            cloud_type or os.environ.get("HARBOR_RUNPOD_CLOUD_TYPE") or "SECURE"
        ).upper()
        self._data_center_ids = _split_list(data_center_ids) or _split_list(
            os.environ.get("HARBOR_RUNPOD_DATA_CENTER_IDS")
        )
        self._network_volume_id = network_volume_id or os.environ.get(
            "HARBOR_RUNPOD_NETWORK_VOLUME_ID"
        )
        self._volume_in_gb = self._int_option(volume_in_gb, "HARBOR_RUNPOD_VOLUME_GB")
        self._container_disk_in_gb = self._int_option(
            container_disk_in_gb, "HARBOR_RUNPOD_CONTAINER_DISK_GB"
        )
        self._interruptible = (
            interruptible
            if interruptible is not None
            else _env_flag("HARBOR_RUNPOD_INTERRUPTIBLE", False)
        )
        self._ssh_user = ssh_user
        self._remote_shell = remote_shell
        self._start_timeout_sec = (
            self._int_option(start_timeout_sec, "HARBOR_RUNPOD_START_TIMEOUT_SEC")
            or _DEFAULT_START_TIMEOUT_SEC
        )
        self._pod_overrides = dict(pod_overrides or {})

        configured_key = ssh_key_path or os.environ.get("HARBOR_RUNPOD_SSH_KEY")
        self._configured_ssh_key = Path(configured_key).expanduser() if configured_key else None
        self._ssh_key_dir: tempfile.TemporaryDirectory[str] | None = None
        self._ssh_key: Path | None = None

        self._workdir = parse_dockerfile_workdir(self._environment_definition_path)
        if self._workdir is None and self._environment_definition_path.is_file():
            self._workdir = "/workspace"

        self._client: RunPodClient | None = None
        self._pod_id: str | None = None
        self._ssh_host: str | None = None
        self._ssh_port: int | None = None
        self._image_name: str | None = None

    # ------------------------------------------------------------------
    # Identity, capabilities, validation
    # ------------------------------------------------------------------

    @staticmethod
    @override
    def type() -> str:
        # A plain string, not an ``EnvironmentType`` member: RunPod is an
        # out-of-tree provider and Harbor's enum is closed. ``BaseEnvironment``
        # documents ``str`` precisely so third-party providers can do this.
        return ENVIRONMENT_TYPE

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # RunPod's sizing fields (``vcpuCount``, ``minRAMPerGPU``,
        # ``minVCPUPerGPU``) select a machine that has *at least* the asked-for
        # resources. Nothing caps the container afterwards, so these are
        # requests, never limits.
        return EnvironmentResourceCapabilities(
            cpu_request=True,
            memory_request=True,
        )

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        enforcing = self._network_enforcement == "best-effort-iptables"
        return EnvironmentCapabilities(
            gpus=True,
            tpus=False,
            windows=False,
            mounted=False,
            # A RunPod pod is one container. Compose would need a Docker daemon
            # inside the pod, which RunPod does not grant (no privileged mode,
            # no docker socket). Declared false so Harbor rejects compose tasks
            # up front rather than half-running them.
            docker_compose=False,
            # RunPod exposes no egress firewall. Off by default, which makes
            # BaseEnvironment reject no-network/allowlist tasks instead of
            # running them unprotected. Opt in to in-pod iptables to trade
            # that rejection for a best-effort guarantee.
            disable_internet=enforcing,
            network_allowlist=enforcing,
            network_allowlist_hostnames=enforcing,
            network_allowlist_wildcard_hostnames=False,
            network_allowlist_ipv4_addresses=enforcing,
            network_allowlist_ipv6_addresses=False,
            network_allowlist_ipv4_cidrs=enforcing,
            network_allowlist_ipv6_cidrs=False,
            dynamic_network_policy=enforcing,
        )

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / DOCKERFILE_NAME

    @property
    def _compose_path(self) -> Path:
        return self.environment_dir / COMPOSE_FILE_NAME

    @override
    def _validate_definition(self):
        require_agent_environment_definition(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
        )
        if self._compose_path.is_file():
            raise ServiceOperationsUnsupportedError(
                f"{self.type()} environment cannot run "
                f"{self._compose_path.name}: a RunPod pod is a single container "
                "and RunPod grants no Docker daemon inside it, so multi-service "
                "compose environments (agent container + sidecar services) "
                "cannot be reproduced. Either merge the services into one "
                f"{DOCKERFILE_NAME} and run the sidecar as a background process, "
                "or run this task on a compose-capable environment (docker, "
                "gke, modal)."
            )

    @staticmethod
    def _resolve_enforcement(value: str | None) -> NetworkEnforcement:
        raw = (value or os.environ.get("HARBOR_RUNPOD_NETWORK_ENFORCEMENT") or "none").strip()
        if raw not in ("none", "best-effort-iptables"):
            raise RunPodConfigurationError(
                f"Unknown network_enforcement {raw!r}. Use 'none' (reject tasks "
                "that need egress control) or 'best-effort-iptables' (enforce "
                "inside the pod with iptables)."
            )
        return raw  # ty: ignore[invalid-return-type]

    @staticmethod
    def _int_option(value: int | None, env_var: str) -> int | None:
        if value is not None:
            return int(value)
        raw = os.environ.get(env_var)
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise RunPodConfigurationError(
                f"{env_var} must be an integer, got {raw!r}."
            ) from exc

    # ------------------------------------------------------------------
    # Image resolution: the local-build / registry-pull bridge
    # ------------------------------------------------------------------

    @property
    def _image_tag(self) -> str:
        """Content-addressed tag for the image built from this task.

        ``environment_id`` is Harbor's own hash of the environment directory
        (and ``docker_image``), so two runs of an unchanged task produce the
        same tag and the second one skips build and push entirely.
        """
        # Lowercased, not just stripped: a Docker repository name must be
        # lowercase, and GitHub usernames routinely are not -- so a perfectly
        # reasonable 'ghcr.io/JinyanSu1' would otherwise sail through preflight
        # and die much later as `invalid tag: repository name must be
        # lowercase`, after the build. The registry *host* is case-insensitive
        # and the credentials are unaffected, so this only ever fixes.
        registry = (self._image_registry or "").rstrip("/").lower()
        repo = _sanitize_image_name(self.environment_name)
        digest = hashlib.sha256(
            f"{self.environment_id}:{_IMAGE_LAYER_VERSION}".encode()
        ).hexdigest()[:SNAPSHOT_HASH_LEN]
        return f"{registry}/{repo}:{digest}"

    async def _resolve_image(self, force_build: bool) -> str:
        """Decide which registry reference the pod should pull.

        Ordered so the cheapest correct answer wins:

        1. an explicit ``image`` override — you already own an SSH-ready image;
        2. ``[environment].docker_image`` when Harbor says a prebuilt image is
           usable (:func:`should_use_prebuilt_docker_image`), pulled as-is;
        3. otherwise build ``environment/Dockerfile`` locally, add the SSH
           layer, and push to the configured registry.
        """
        if self._image_override:
            self.logger.debug("Using pinned RunPod image %s", self._image_override)
            return self._image_override

        if should_use_prebuilt_docker_image(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            force_build=force_build,
        ):
            image = self.task_env_config.docker_image
            assert image  # guaranteed by should_use_prebuilt_docker_image
            self.logger.debug(
                "Using prebuilt image %s; sshd will be installed at pod boot "
                "because the image was not built through this provider.",
                image,
            )
            return image

        if not self._environment_definition_path.is_file():
            raise RunPodConfigurationError(
                f"Task {self.environment_name!r} has neither "
                f"[environment].docker_image nor environment/{DOCKERFILE_NAME}, "
                "so there is nothing for RunPod to pull."
            )

        return await self._build_and_push_image(force_build=force_build)

    async def _build_and_push_image(self, *, force_build: bool) -> str:
        """Build the task image locally and push it where RunPod can pull it.

        Every prerequisite is checked before the first ``docker`` invocation so
        the failure message names the missing piece instead of surfacing as a
        build error 200 lines deep.
        """
        if not self._image_registry:
            raise RunPodConfigurationError(
                f"Task {self.environment_name!r} builds from environment/"
                f"{DOCKERFILE_NAME}, but RunPod can only pull images from a "
                "registry. Set HARBOR_RUNPOD_IMAGE_REGISTRY (e.g. "
                "'docker.io/myorg' or 'ghcr.io/myorg') so the built image can "
                "be pushed, or set [environment].docker_image / "
                "--environment-kwarg image=<ref> to a prebuilt image."
            )
        if shutil.which("docker") is None:
            raise RunPodConfigurationError(
                "Building this task for RunPod needs a local Docker daemon "
                "(RunPod has no build service), but 'docker' is not on PATH. "
                "Set [environment].docker_image or --environment-kwarg "
                "image=<ref> to skip the build."
            )

        tag = self._image_tag
        if not force_build and await self._remote_image_exists(tag):
            self.logger.debug("Reusing pushed RunPod image %s", tag)
            self._image_name = tag
            return tag

        await self._registry_login()

        base_tag = f"harbor-runpod-base:{self.environment_id[:SNAPSHOT_HASH_LEN]}"
        self.logger.debug(
            "Building %s from %s", base_tag, self._environment_definition_path
        )
        await self._run_docker(
            [
                "build",
                "--tag",
                base_tag,
                "--file",
                str(self._environment_definition_path),
                str(self.environment_dir),
            ],
            what=f"build task image for {self.environment_name}",
        )

        with tempfile.TemporaryDirectory(prefix="harbor-runpod-ssh-layer-") as tmpdir:
            layer_dir = Path(tmpdir)
            (layer_dir / DOCKERFILE_NAME).write_text(_ssh_layer_dockerfile(base_tag))
            self.logger.debug("Adding SSH layer and tagging %s", tag)
            await self._run_docker(
                ["build", "--tag", tag, str(layer_dir)],
                what=f"add the SSH layer to {self.environment_name}",
            )

        self.logger.debug("Pushing %s", tag)
        await self._run_docker(
            ["push", tag],
            what=f"push {tag}",
        )
        self._image_name = tag
        return tag

    async def _remote_image_exists(self, tag: str) -> bool:
        """True when *tag* is already in the registry (cheap cache check)."""
        code, _, _ = await self._run_process(
            ["docker", "manifest", "inspect", tag], timeout_sec=120
        )
        return code == 0

    async def _registry_login(self) -> None:
        if not (self._registry_username and self._registry_password):
            # Assume an ambient `docker login`; push will fail loudly if not.
            return
        registry_host = self._image_registry or ""
        host = registry_host.split("/", 1)[0]
        code, _, stderr = await self._run_process(
            ["docker", "login", host, "--username", self._registry_username, "--password-stdin"],
            stdin=self._registry_password.encode(),
            timeout_sec=120,
        )
        if code != 0:
            raise RunPodConfigurationError(
                f"docker login to {host!r} failed: {stderr.decode(errors='replace')[:500]}"
            )

    async def _run_docker(self, args: list[str], *, what: str) -> None:
        code, stdout, stderr = await self._run_process(
            ["docker", *args], timeout_sec=self.task_env_config.build_timeout_sec
        )
        if code != 0:
            detail = (stderr or stdout).decode("utf-8", errors="replace")
            raise SandboxBuildFailedError(
                f"Failed to {what} (docker exited {code}):\n{detail[-4000:]}"
            )

    async def _ensure_registry_auth_id(self) -> str | None:
        """Register pull credentials with RunPod and return their id.

        RunPod calls this a *container registry auth*; the id goes into
        ``PodCreateInput.containerRegistryAuthId``. Public images need none.
        """
        if not (self._registry_username and self._registry_password):
            return None
        client = self._require_client()
        name = f"harbor-{_sanitize_image_name(self._image_registry or 'registry')}"
        try:
            return await client.ensure_container_registry_auth(
                name=name,
                username=self._registry_username,
                password=self._registry_password,
            )
        except RunPodAPIError as exc:
            raise RunPodConfigurationError(
                "Failed to register container registry credentials with RunPod "
                f"(needed to pull from {self._image_registry!r}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Pod lifecycle
    # ------------------------------------------------------------------

    def _require_client(self) -> RunPodClient:
        if self._client is None:
            raise RuntimeError("RunPod client not initialized; call start() first.")
        return self._client

    def _ensure_ssh_key(self) -> tuple[Path, str]:
        """Return ``(private_key_path, public_key_text)``.

        A caller-supplied key is used as-is (handy for reconnecting to a pod
        kept alive with ``delete=False``); otherwise a throwaway ed25519 pair
        is generated per environment so a leaked trial key authorizes nothing
        else.
        """
        if self._configured_ssh_key is not None:
            private = self._configured_ssh_key
            public = private.with_suffix(private.suffix + ".pub")
            if not private.is_file():
                raise RunPodConfigurationError(
                    f"HARBOR_RUNPOD_SSH_KEY points at {private}, which does not exist."
                )
            if not public.is_file():
                raise RunPodConfigurationError(
                    f"Public key {public} not found next to {private}."
                )
            self._ssh_key = private
            return private, public.read_text().strip()

        if self._ssh_key is not None:
            return self._ssh_key, self._ssh_key.with_suffix(".pub").read_text().strip()

        self._ssh_key_dir = tempfile.TemporaryDirectory(prefix="harbor-runpod-key-")
        private = Path(self._ssh_key_dir.name) / "id_ed25519"
        result = subprocess.run(
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                f"harbor-runpod-{self.session_id}",
                "-f",
                str(private),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RunPodConfigurationError(
                "ssh-keygen failed to create a trial key: "
                f"{result.stderr.decode(errors='replace')[:500]}"
            )
        private.chmod(0o600)
        self._ssh_key = private
        return private, private.with_suffix(".pub").read_text().strip()

    def _build_pod_create_payload(
        self, image_name: str, registry_auth_id: str | None, public_key: str
    ) -> dict[str, Any]:
        """Assemble ``PodCreateInput``.

        Field names and value domains come from the published OpenAPI schema;
        every mapping from a Harbor concept is commented where it is not
        one-to-one.
        """
        gpus = self._effective_gpus
        compute_type = "GPU" if gpus > 0 else "CPU"

        env = dict(self._startup_env())
        # RunPod's own convention for seeding authorized_keys. Our bootstrap
        # reads it directly, so it works even in images that know nothing
        # about RunPod.
        env["PUBLIC_KEY"] = public_key

        payload: dict[str, Any] = {
            "name": self._pod_name(),
            "imageName": image_name,
            "computeType": compute_type,
            "cloudType": self._cloud_type,
            "interruptible": self._interruptible,
            "ports": format_ports([]),
            "env": env,
            # Replace the image's entrypoint *and* command: PID 1 must be our
            # bootstrap, or there is no way in.
            "dockerEntrypoint": ["/bin/sh", "-c"],
            "dockerStartCmd": [_bootstrap_script()],
            # Secure Cloud always has a public IP; Community Cloud only when
            # asked. Direct-TCP SSH is the only mode that supports the streamed
            # tar transfers this provider uses.
            "supportPublicIp": True,
        }

        if registry_auth_id:
            payload["containerRegistryAuthId"] = registry_auth_id

        if gpus > 0:
            payload["gpuCount"] = gpus
            gpu_type_ids = self._gpu_type_ids or _split_list(
                self.task_env_config.gpu_types
            )
            if not gpu_type_ids:
                raise RunPodConfigurationError(
                    f"Task requests {gpus} GPU(s) but no GPU type was specified. "
                    "RunPod rents a named SKU, so set [environment].gpu_types in "
                    "task.toml or HARBOR_RUNPOD_GPU_TYPE_IDS / "
                    "--environment-kwarg gpu_type_ids='NVIDIA A40'. Values must "
                    "match RunPod's gpuTypeIds exactly, e.g. "
                    "'NVIDIA A100 80GB PCIe'."
                )
            payload["gpuTypeIds"] = gpu_type_ids
            payload["gpuTypePriority"] = "custom"
            memory_mb = self._effective_memory_mb
            if memory_mb:
                # minRAMPerGPU is per-GPU; Harbor's memory_mb is for the pod.
                payload["minRAMPerGPU"] = max(1, _ceil_div(memory_mb, 1024 * gpus))
            cpus = self._effective_cpus
            if cpus:
                payload["minVCPUPerGPU"] = max(1, _ceil_div(cpus, gpus))
        else:
            cpus = self._effective_cpus
            if cpus:
                payload["vcpuCount"] = cpus
            # CPU pods size RAM through the cpu flavor, not a free field, so
            # memory_mb cannot be requested independently here. Documented in
            # the README rather than silently dropped.

        storage_mb = self._effective_storage_mb
        if self._container_disk_in_gb is not None:
            payload["containerDiskInGb"] = self._container_disk_in_gb
        elif storage_mb:
            payload["containerDiskInGb"] = max(10, _ceil_div(storage_mb, 1024))

        if self._volume_in_gb is not None:
            payload["volumeInGb"] = self._volume_in_gb

        if self._network_volume_id:
            # A network volume pins the pod to the volume's data center; do not
            # also send a conflicting dataCenterIds filter.
            payload["networkVolumeId"] = self._network_volume_id
        elif self._data_center_ids:
            payload["dataCenterIds"] = self._data_center_ids
            payload["dataCenterPriority"] = "custom"

        for key, value in self._pod_overrides.items():
            if key not in _POD_CREATE_KNOWN_FIELDS:
                self.logger.warning(
                    "pod_overrides key %r is not a documented PodCreateInput "
                    "field; sending it anyway.",
                    key,
                )
            payload[key] = value

        return payload

    def _pod_name(self) -> str:
        return f"harbor-{self.session_id}"[:64]

    @override
    async def start(self, force_build: bool) -> None:
        if not self._api_key:
            raise RunPodConfigurationError(
                "RUNPOD_API_KEY is not set; cannot create a RunPod pod."
            )
        self._client = RunPodClient(self._api_key, logger_=self.logger)

        image_name = await self._resolve_image(force_build)
        self._image_name = image_name
        registry_auth_id = await self._ensure_registry_auth_id()
        _, public_key = self._ensure_ssh_key()

        payload = self._build_pod_create_payload(
            image_name, registry_auth_id, public_key
        )
        self.logger.debug(
            "Creating RunPod pod %s from %s (computeType=%s)",
            payload["name"],
            image_name,
            payload["computeType"],
        )
        pod = await self._require_client().create_pod(payload)
        self._pod_id = str(pod.get("id") or "")
        if not self._pod_id:
            raise RunPodAPIError("RunPod created a pod without returning an id.")
        self.logger.debug("RunPod pod %s created", self._pod_id)

        await self._wait_for_ssh()
        await self._apply_network_policy(self.network_policy)

        initial_dirs = self._mount_targets(writable_only=True)
        default_cwd = effective_exec_cwd(
            None, self.task_env_config.workdir, self._workdir
        )
        if default_cwd:
            initial_dirs.append(default_cwd)
        if initial_dirs:
            result = await self.exec(
                self._ensure_dirs_command(initial_dirs),
                cwd="/",
                user=self._reset_dirs_user(),
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"Failed to create environment dirs (exit {result.return_code}): "
                    f"{result.stderr}"
                )

        await self._upload_environment_dir_after_start()

    async def _wait_for_ssh(self) -> None:
        """Poll until the pod publishes an SSH endpoint and accepts our key.

        Two separate waits, because they fail differently: RunPod populating
        ``publicIp``/``portMappings`` means the pod was scheduled; SSH
        answering means the image pulled and our bootstrap ran.
        """
        client = self._require_client()
        pod_id = self._pod_id
        assert pod_id
        deadline = asyncio.get_running_loop().time() + self._start_timeout_sec

        endpoint: tuple[str, int] | None = None
        while endpoint is None:
            pod = await client.get_pod(pod_id)
            if pod_is_terminal(pod):
                raise RuntimeError(
                    f"RunPod pod {pod_id} reached desiredStatus="
                    f"{pod.get('desiredStatus')!r} before it became reachable. "
                    "This usually means the image could not be pulled (check "
                    "container registry credentials) or capacity was withdrawn."
                )
            endpoint = pod_ssh_endpoint(pod)
            if endpoint is None:
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError(
                        f"RunPod pod {pod_id} did not publish a public IP and a "
                        f"mapping for port 22 within {self._start_timeout_sec}s. "
                        "On Community Cloud a pod may land on a machine without "
                        "a public IP; set cloud_type=SECURE."
                    )
                await asyncio.sleep(_POD_POLL_INTERVAL_SEC)

        self._ssh_host, self._ssh_port = endpoint
        self.logger.debug(
            "RunPod pod %s reachable at %s:%s", pod_id, self._ssh_host, self._ssh_port
        )

        last_error = ""
        while asyncio.get_running_loop().time() <= deadline:
            code, _, stderr = await self._ssh_process(
                "true", timeout_sec=_DEFAULT_SSH_CONNECT_TIMEOUT_SEC + 5
            )
            if code == 0:
                return
            last_error = stderr.decode("utf-8", errors="replace").strip()
            await asyncio.sleep(_SSH_POLL_INTERVAL_SEC)

        raise TimeoutError(
            f"RunPod pod {pod_id} never accepted SSH within "
            f"{self._start_timeout_sec}s. Last error: {last_error or 'none'}. "
            "If the image was not built through this provider, it may lack "
            "openssh-server."
        )

    @override
    async def stop(self, delete: bool):
        client = self._client
        pod_id = self._pod_id
        try:
            if client is not None and not pod_id:
                # start() got as far as building a client but never recorded an
                # id. Usually that means it failed before create_pod -- but it
                # also covers the window where create_pod's HTTP request was
                # cancelled in flight: RunPod made the pod, the response never
                # landed, and nothing local knows the id. That pod bills until
                # someone notices it in the console. _pod_name() is
                # deterministic, so reconcile by name before giving up.
                pod_id = await self._find_leaked_pod_id(client)
            if client is None or not pod_id:
                self.logger.debug("No RunPod pod to stop.")
            elif delete:
                self.logger.debug("Terminating RunPod pod %s", pod_id)
                try:
                    await client.delete_pod(pod_id)
                except RunPodAPIError as exc:
                    if exc.is_not_found:
                        self.logger.debug("RunPod pod %s already gone.", pod_id)
                    else:
                        # Never swallow this: an orphaned pod bills by the hour.
                        self.logger.error(
                            "Failed to terminate RunPod pod %s (%s). Terminate it "
                            "manually at https://console.runpod.io/pods to stop "
                            "billing.",
                            pod_id,
                            exc,
                        )
            else:
                self.logger.debug(
                    "Stopping (not deleting) RunPod pod %s; disk storage is still "
                    "billed while it is stopped.",
                    pod_id,
                )
                try:
                    await client.stop_pod(pod_id)
                except RunPodAPIError as exc:
                    self.logger.error("Failed to stop RunPod pod %s: %s", pod_id, exc)
        finally:
            self._pod_id = None
            self._ssh_host = None
            self._ssh_port = None
            self._client = None
            if self._ssh_key_dir is not None:
                self._ssh_key_dir.cleanup()
                self._ssh_key_dir = None
                self._ssh_key = None

    async def _find_leaked_pod_id(self, client: RunPodClient) -> str:
        """Look for a pod this environment created but never recorded.

        Matched on ``_pod_name()``, which embeds ``session_id`` and is unique
        per environment instance -- the separate verifier env gets its own, so
        this can never reach across and terminate a sibling's pod.

        Best effort by construction: it runs on a teardown path that is often
        already unwinding an exception, so a failure here must not replace the
        error the caller is reporting.
        """
        name = self._pod_name()
        try:
            pods = await client.list_pods()
        except Exception as exc:  # noqa: BLE001 - see docstring
            self.logger.warning(
                "Could not list RunPod pods to check for a leaked '%s' (%s). "
                "If a pod by that name exists, terminate it at "
                "https://console.runpod.io/pods to stop billing.",
                name,
                exc,
            )
            return ""

        for pod in pods:
            if pod.get("name") == name:
                found = str(pod.get("id") or "")
                if found:
                    self.logger.warning(
                        "Found RunPod pod %s named '%s' that this environment "
                        "created but never recorded (start() was likely "
                        "interrupted mid-create). Terminating it.",
                        found,
                        name,
                    )
                    return found
        return ""

    # ------------------------------------------------------------------
    # Network policy (opt-in, in-pod)
    # ------------------------------------------------------------------

    @override
    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        if self._network_enforcement != "best-effort-iptables":
            # capabilities() already refused any policy we cannot honour, so
            # the only policy that reaches here is PUBLIC: nothing to do.
            return
        if network_policy.network_mode == NetworkMode.PUBLIC:
            script = self._iptables_reset_script()
        else:
            script = self._iptables_policy_script(network_policy)
        result = await self.exec(script, user="root", timeout_sec=180)
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to apply the network policy inside the RunPod pod "
                f"(exit {result.return_code}): {result.stderr or result.stdout}. "
                "RunPod containers may not carry CAP_NET_ADMIN; without it this "
                "provider cannot enforce network_mode "
                f"{network_policy.network_mode.value!r} and refuses to pretend "
                "otherwise."
            )

    @staticmethod
    def _iptables_reset_script() -> str:
        return (
            "set -e\n"
            "command -v iptables >/dev/null 2>&1 || exit 0\n"
            "iptables -D OUTPUT -j HARBOR_EGRESS 2>/dev/null || true\n"
            "iptables -F HARBOR_EGRESS 2>/dev/null || true\n"
            "iptables -X HARBOR_EGRESS 2>/dev/null || true\n"
        )

    def _iptables_policy_script(self, network_policy: NetworkPolicy) -> str:
        """Build an egress filter for *network_policy*.

        Established connections are always accepted so the SSH session that is
        applying the rules (and every later ``exec``) survives its own policy.
        """
        lines = [
            "set -e",
            'command -v iptables >/dev/null 2>&1 || { echo "iptables missing" >&2; exit 90; }',
            "iptables -N HARBOR_EGRESS 2>/dev/null || iptables -F HARBOR_EGRESS",
            "iptables -C OUTPUT -j HARBOR_EGRESS 2>/dev/null || "
            "iptables -I OUTPUT 1 -j HARBOR_EGRESS",
            "iptables -A HARBOR_EGRESS -o lo -j ACCEPT",
            "iptables -A HARBOR_EGRESS -m conntrack --ctstate ESTABLISHED,RELATED "
            "-j ACCEPT",
        ]
        if network_policy.network_mode == NetworkMode.ALLOWLIST:
            # DNS must stay open or hostname allowlists cannot resolve at all.
            lines.append(
                "iptables -A HARBOR_EGRESS -p udp --dport 53 -j ACCEPT"
            )
            lines.append(
                "iptables -A HARBOR_EGRESS -p tcp --dport 53 -j ACCEPT"
            )
            for entry in network_policy.allowed_hosts:
                kind = classify_network_allowlist_entry(entry)
                if kind in (
                    NetworkAllowlistEntryType.IPV4_ADDRESS,
                    NetworkAllowlistEntryType.IPV4_CIDR,
                ):
                    lines.append(
                        f"iptables -A HARBOR_EGRESS -d {shlex.quote(entry)} -j ACCEPT"
                    )
                elif kind == NetworkAllowlistEntryType.HOSTNAME:
                    # Resolved on the pod at apply time. See README: a hostname
                    # allowlist is only as accurate as this one DNS answer.
                    quoted = shlex.quote(entry)
                    lines.append(
                        f'for ip in $(getent ahostsv4 {quoted} 2>/dev/null | '
                        "awk '{print $1}' | sort -u); do "
                        'iptables -A HARBOR_EGRESS -d "$ip" -j ACCEPT; done'
                    )
        lines.append("iptables -A HARBOR_EGRESS -j REJECT || "
                     "iptables -A HARBOR_EGRESS -j DROP")
        # No IPv6 rule set is maintained, so drop IPv6 egress wholesale rather
        # than leave an unfiltered path around the IPv4 policy.
        lines.append(
            "command -v ip6tables >/dev/null 2>&1 && "
            "{ ip6tables -P OUTPUT DROP 2>/dev/null || true; } || true"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SSH transport
    # ------------------------------------------------------------------

    def _ssh_argv(self, remote_command: str) -> list[str]:
        if not self._ssh_host or not self._ssh_port or not self._ssh_key:
            raise RunPodSSHError(
                "No SSH endpoint for this RunPod pod. Call start() first."
            )
        return [
            "ssh",
            "-i",
            str(self._ssh_key),
            "-p",
            str(self._ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            f"ConnectTimeout={_DEFAULT_SSH_CONNECT_TIMEOUT_SEC}",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            f"{self._ssh_user}@{self._ssh_host}",
            remote_command,
        ]

    async def _ssh_process(
        self,
        remote_command: str,
        *,
        stdin: bytes | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        return await self._run_process(
            self._ssh_argv(remote_command), stdin=stdin, timeout_sec=timeout_sec
        )

    async def _run_process(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run *argv*, returning ``(return_code, stdout, stderr)``.

        The single subprocess seam for this provider: tests replace it to drive
        ``start``/``exec``/transfers without a network.
        """
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin), timeout=timeout_sec
            )
        except (TimeoutError, asyncio.TimeoutError):
            process.kill()
            await process.wait()
            return _TIMEOUT_RETURN_CODE, b"", b"timed out"
        except asyncio.CancelledError:
            # Harbor enforces [agent].timeout_sec with an asyncio.wait_for one
            # level up (trial.py), so a agent that overruns arrives here as
            # CANCELLATION, not as the TimeoutError handled above -- and
            # CancelledError is a BaseException, so without this arm it sails
            # past every handler and the ssh client is never killed. The remote
            # agent then keeps writing while _download_agent_logs() tars the
            # same tree, which is how a half-written trajectory.json gets
            # collected and scored.
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            raise
        return process.returncode or 0, stdout, stderr

    def _remote_script(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
    ) -> str:
        """Compose the shell program that runs on the pod.

        The program is base64-encoded into the ssh argv rather than quoted into
        it: ssh concatenates its remote-command arguments and hands the result
        to a login shell, so a single mis-escaped quote in an agent's command
        would otherwise change what runs.
        """
        parts: list[str] = [_CONTAINER_ENV_IMPORT]
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        for key, value in (env or {}).items():
            parts.append(f"export {key}={shlex.quote(value)}")
        parts.append(command)
        script = "\n".join(parts)

        if user is not None:
            inner = base64.b64encode(script.encode()).decode()
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(user))
            # su (not su -) so the cd above still decides the working directory.
            decode = f"echo {inner} | base64 -d | {self._remote_shell}"
            script = (
                f"su {user_arg} -s /bin/{self._remote_shell} "
                f"-c {shlex.quote(decode)}"
            )

        payload = base64.b64encode(script.encode()).decode()
        # The payload runs in its OWN process group, via setsid.
        #
        # Some agents end by signalling their entire process group. kimi-cli is
        # one: Harbor's kimi agent terminates its run command with `kill 0` and
        # then *expects* exit 143 back, swallowing it as normal completion
        # (agents/installed/kimi_cli.py). Under the docker backend the exit
        # status arrives out-of-band from the daemon, so `kill 0` is harmless.
        # Here it is carried in-band by the printf below -- which is in that
        # same process group, and so was being killed before it could run. ssh
        # then exits 255 with an empty stderr, the sentinel is missing, and
        # exec() correctly refuses to guess: a finished trial surfaced as
        # RunPodSSHError. Deterministic, not flaky -- it hit every kimi trial.
        #
        # `--wait` makes setsid block and hand back the child's real status.
        # Where setsid is absent we degrade to a bare shell rather than fail;
        # the sentinel is then exposed again, but only for group-signalling
        # agents, which is strictly better than refusing to run at all.
        #
        # Trade-off: a payload in its own session no longer dies when the ssh
        # channel does, so an agent killed by Harbor's timeout keeps running on
        # the pod until stop() deletes it. That window is bounded by teardown,
        # which follows immediately.
        runner = (
            f"if command -v setsid >/dev/null 2>&1; "
            f"then setsid --wait {self._remote_shell}; "
            f"else {self._remote_shell}; fi"
        )
        return (
            f"echo {payload} | base64 -d | {runner}; "
            f"printf '{_RC_PREFIX}%s{_RC_SUFFIX}' \"$?\""
        )

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.os == TaskOS.WINDOWS:
            raise RuntimeError(
                f"{self.type()} environment does not support Windows tasks."
            )
        user = self._resolve_user(user)
        merged_env = self._merge_env(env)
        effective_cwd = effective_exec_cwd(
            cwd, self.task_env_config.workdir, self._workdir
        )

        remote = self._remote_script(
            command, cwd=effective_cwd, env=merged_env, user=user
        )
        code, stdout_bytes, stderr_bytes = await self._ssh_process(
            remote, timeout_sec=timeout_sec or _DEFAULT_EXEC_TIMEOUT_SEC
        )
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if code == _TIMEOUT_RETURN_CODE and not stdout:
            self.logger.warning(
                "RunPod exec timed out after %ss: %r", timeout_sec, command
            )
            return ExecResult(
                stdout="",
                stderr=f"Command timed out after {timeout_sec}s",
                return_code=_TIMEOUT_RETURN_CODE,
            )

        match = _RC_PATTERN.search(stdout)
        if match is None:
            # The sentinel is printed unconditionally by the remote script, so
            # its absence means ssh itself failed. Surfacing that as a non-zero
            # ExecResult would let a broken control channel masquerade as a
            # failing test.
            raise RunPodSSHError(
                f"SSH transport to RunPod pod {self._pod_id} failed while running "
                f"{command!r} (ssh exit {code}): {stderr.strip() or 'no stderr'}"
            )

        return ExecResult(
            stdout=stdout[: match.start()],
            stderr=stderr,
            return_code=int(match.group(1)),
        )

    # ------------------------------------------------------------------
    # File transfer — one streamed tar (or cat) per call, no scp
    # ------------------------------------------------------------------

    @override
    async def upload_file(self, source_path: Path | str, target_path: str):
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file {source} does not exist")
        parent = str(PurePosixPath(target_path).parent)
        remote = (
            f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(str(target_path))}"
        )
        code, _, stderr = await self._ssh_process(remote, stdin=source.read_bytes())
        if code != 0:
            raise RunPodSSHError(
                f"Failed to upload {source} to {target_path} (exit {code}): "
                f"{stderr.decode('utf-8', errors='replace')[:1000]}"
            )

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        # tar_transfer.pack_dir keeps modes, symlinks and empty directories,
        # which a per-file loop would silently drop.
        buffer = io.BytesIO()
        pack_dir(source_dir, buffer, compress=True)
        remote = (
            f"mkdir -p {shlex.quote(str(target_dir))} && "
            f"tar -xzf - -C {shlex.quote(str(target_dir))}"
        )
        code, _, stderr = await self._ssh_process(remote, stdin=buffer.getvalue())
        if code != 0:
            raise RunPodSSHError(
                f"Failed to upload directory {source_dir} to {target_dir} "
                f"(exit {code}): {stderr.decode('utf-8', errors='replace')[:1000]}"
            )

    @override
    async def download_file(self, source_path: str, target_path: Path | str):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        remote = f"cat {shlex.quote(source_path)}"
        code, stdout, stderr = await self._ssh_process(remote)
        if code != 0:
            raise RunPodSSHError(
                f"Failed to download {source_path} (exit {code}): "
                f"{stderr.decode('utf-8', errors='replace')[:1000]}"
            )
        target.write_bytes(stdout)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str):
        remote = f"tar -czf - -C {shlex.quote(source_dir)} ."
        code, stdout, stderr = await self._ssh_process(remote)
        if code != 0:
            raise RunPodSSHError(
                f"Failed to download directory {source_dir} (exit {code}): "
                f"{stderr.decode('utf-8', errors='replace')[:1000]}"
            )
        extract_dir(io.BytesIO(stdout), target_dir)

    # ------------------------------------------------------------------
    # Compose service operations: refused, loudly
    # ------------------------------------------------------------------

    @override
    async def stop_service(self, service: str) -> None:
        raise ServiceOperationsUnsupportedError(
            self._service_unsupported_message(service)
        )


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)
