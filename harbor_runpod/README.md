# harbor-runpod

A [Harbor](https://github.com/harbor-framework/harbor) environment backend that
runs a task inside a **RunPod On-Demand (or spot) Pod**.

Use it when a task needs a GPU that your Docker host does not have. Do **not**
use it for the compose-based tasks in this repo — see
[Compose is not supported](#compose-is-not-supported).

## Harbor version

Written against **Harbor v0.20.0** (the current release; use a release, not
`main`). The six modules this backend depends on — `environments/base.py`,
`factory.py`, `capabilities.py`, `definition.py`, `tar_transfer.py` and
`cli/utils.py` — were byte-identical between `v0.20.0` and `main` at the time of
writing, so either works today. That will not stay true: `BaseEnvironment` is an
ABC with nine abstract members, and a new one in a later release turns into an
instantiation `TypeError` here.

```bash
pip install 'harbor==0.20.0'
```

Before moving to a newer Harbor, re-run the signature cross-check described
under [Verification](#verification) against that version's `base.py`.

---

## Registering the backend

**`harbor run --env runpod` does not work, and cannot be made to work without
patching Harbor.** `--env` is parsed by `resolve_environment_spec`
(`src/harbor/cli/utils.py`), which accepts either a member of the **closed**
`EnvironmentType` enum or a `module:Class` import path — distinguished by the
`:`. There is no `runpod` member, and no entry-point mechanism for
environments (the `harbor.plugins` entry point group exists, but
`resolve_plugin_import_path` is only ever called from `cli/job_plugins.py` and
its result must subclass `JobPlugin`).

So the supported spelling is the import path:

```bash
pip install -e ./harbor_runpod          # just puts harbor_runpod on sys.path

harbor run \
  --env harbor_runpod:RunPodEnvironment \
  --task tasks/crisis-early-detection \
  --agent claude-code
```

Everything downstream works normally: `EnvironmentFactory` calls
`create_environment_from_import_path`, `preflight()` runs before trials are
queued, and `--environment-kwarg`/`--ek` reaches this class's constructor.

If you want the literal `--env runpod` spelling, it takes a two-line patch to
Harbor itself: add `RUNPOD = "runpod"` to `EnvironmentType` and an
`_EnvEntry("harbor_runpod.runpod_env", "RunPodEnvironment", None)` row to
`_ENVIRONMENT_REGISTRY` in `src/harbor/environments/factory.py`.

---

## Required configuration

Every setting is available as an environment variable **and** as an
`--environment-kwarg key=value` (the kwarg wins).

| Env var | Kwarg | Required | Meaning |
|---|---|---|---|
| `RUNPOD_API_KEY` | `api_key` | **yes** | RunPod API key. Checked in `preflight()`. |
| `HARBOR_RUNPOD_IMAGE_REGISTRY` | `image_registry` | for `Dockerfile` tasks | Registry prefix to push to, e.g. `docker.io/myorg` or `ghcr.io/myorg`. Images are tagged `<prefix>/<task-name>:<content-hash>`. |
| `HARBOR_RUNPOD_REGISTRY_USERNAME` | `registry_username` | for private registries | Used for `docker login` **and** registered with RunPod so the pod can pull. |
| `HARBOR_RUNPOD_REGISTRY_PASSWORD` | `registry_password` | for private registries | Token/password. |
| `HARBOR_RUNPOD_IMAGE` | `image` | no | Pin an exact image reference and skip build/push entirely. |
| `HARBOR_RUNPOD_GPU_TYPE_IDS` | `gpu_type_ids` | for GPU tasks | Comma-separated RunPod GPU SKUs, e.g. `NVIDIA A100 80GB PCIe`. Falls back to `[environment].gpu_types` in `task.toml`. |
| `HARBOR_RUNPOD_CLOUD_TYPE` | `cloud_type` | no | `SECURE` (default) or `COMMUNITY`. |
| `HARBOR_RUNPOD_DATA_CENTER_IDS` | `data_center_ids` | no | Restrict placement, e.g. `US-KS-2,EU-RO-1`. |
| `HARBOR_RUNPOD_NETWORK_VOLUME_ID` | `network_volume_id` | no | Attach a network volume. **Pins the pod to that volume's data center.** |
| `HARBOR_RUNPOD_VOLUME_GB` | `volume_in_gb` | no | Persistent pod volume size. |
| `HARBOR_RUNPOD_CONTAINER_DISK_GB` | `container_disk_in_gb` | no | Overrides the value derived from `[environment].storage_mb`. |
| `HARBOR_RUNPOD_INTERRUPTIBLE` | `interruptible` | no | `true` for spot pods (cheaper, can be reclaimed mid-trial). |
| `HARBOR_RUNPOD_SSH_KEY` | `ssh_key_path` | no | Reuse a key instead of generating a throwaway one. Expects `<path>` and `<path>.pub`. |
| `HARBOR_RUNPOD_NETWORK_ENFORCEMENT` | `network_enforcement` | no | `none` (default) or `best-effort-iptables`. See [Network policy](#network-policy). |
| `HARBOR_RUNPOD_START_TIMEOUT_SEC` | `start_timeout_sec` | no | Provisioning + SSH readiness budget. Default 900. |
| — | `pod_overrides` | no | Dict merged verbatim into `PodCreateInput`, e.g. `--ek pod_overrides='{"minDownloadMbps": 500}'`. |

Local binaries: `ssh` and `ssh-keygen` are always required (checked in
`preflight()`); `docker` is required only when building a task's `Dockerfile`.

---

## How it works

### Image build and push (the impedance mismatch)

Harbor's Docker backend builds `environment/Dockerfile` locally. **RunPod has no
build service** — it only pulls a reference from a registry. `_resolve_image()`
bridges that, cheapest correct answer first:

1. **`image` / `HARBOR_RUNPOD_IMAGE` is set** → used verbatim. You are asserting
   the image already contains `openssh-server`.
2. **`[environment].docker_image` is usable** (per Harbor's own
   `should_use_prebuilt_docker_image`) → pulled as-is. `sshd` is installed at
   pod boot by the bootstrap script, which needs egress and a supported package
   manager in that image.
3. **Otherwise, build locally and push.** Two builds: the task's `Dockerfile`
   as-is, then a one-`RUN` layer on top that guarantees `openssh-server` (and
   `iptables`) exist. The result is tagged
   `<registry>/<task-name>:<hash>` where the hash covers Harbor's
   `environment_id` (a content hash of `environment/`) **and** the version of
   the SSH layer. Repeated runs of an unchanged task find the tag already in the
   registry via `docker manifest inspect` and skip build *and* push. `--force-build`
   bypasses the cache.

If a `Dockerfile` task has no registry configured, you get a
`RunPodConfigurationError` naming `HARBOR_RUNPOD_IMAGE_REGISTRY` **before any
billable API call** — not a failure deep inside pod creation.

Private registries: RunPod calls the concept a *container registry auth*
(`POST /v1/containerregistryauth` → `{id}`), and the id is passed as
`containerRegistryAuthId` on pod create. This backend finds-or-creates one named
`harbor-<registry>` automatically when you supply credentials. RunPod never
returns stored credentials, so an existing record is reused as-is; to rotate,
delete it in the RunPod console and re-run.

### Exec and file transfer over SSH

RunPod has **no exec API**. `runpodctl exec` is hidden, deprecated, has exactly
one subcommand (`python`), and does not even propagate exit codes — it is not a
usable general channel. So the control plane here is plain SSH:

* Pod create sets `dockerEntrypoint: ["/bin/sh","-c"]` and a bootstrap
  `dockerStartCmd` that writes `$PUBLIC_KEY` to `authorized_keys`, ensures
  `sshd`, and `exec`s it as PID 1. Overriding the *entrypoint* (not just the
  command) is what makes this work regardless of the task image's `ENTRYPOINT`.
* A throwaway ed25519 keypair is generated per environment, so a leaked trial
  key authorizes nothing else.
* `start()` polls `GET /pods/{id}` until `publicIp` and `portMappings["22"]`
  appear, then polls SSH until it answers.
* **Direct TCP only.** RunPod's `ssh.runpod.io` proxy explicitly does not
  support SCP/SFTP, so `supportPublicIp: true` and `22/tcp` are always
  requested.
* Every remote program is base64-encoded into the ssh argv, so an agent command
  containing quotes cannot change what runs. A `__HARBOR_RUNPOD_RC:<n>:RC__`
  sentinel carries the true exit status — without it, ssh's own exit 255 is
  indistinguishable from a command that exited 255. **A missing sentinel raises
  `RunPodSSHError` rather than being reported as a failing command**, so a
  broken control channel can never be scored as a failed test.
* Each remote shell re-imports `/proc/1/environ` first. A non-interactive ssh
  session otherwise inherits sshd's environment, losing everything the image set
  with `ENV` and everything RunPod injected — including `PATH`.
* Transfers are one streamed tar (or `cat`) per call, never scp:
  `upload_dir` pipes `tar_transfer.pack_dir` into `tar -xzf -`, `download_dir`
  reads `tar -czf - -C src .`. This preserves modes, symlinks, and empty
  directories, and works on any pod where only an exec channel exists.

### Resource mapping

| Harbor | RunPod (`PodCreateInput`) |
|---|---|
| `gpus` > 0 | `computeType: GPU`, `gpuCount` |
| `gpu_types` / `gpu_type_ids` | `gpuTypeIds` + `gpuTypePriority: custom` |
| `memory_mb` (GPU pod) | `minRAMPerGPU` = `ceil(memory_mb / 1024 / gpuCount)` |
| `cpus` (GPU pod) | `minVCPUPerGPU` = `ceil(cpus / gpuCount)` |
| `cpus` (CPU pod) | `vcpuCount` |
| `storage_mb` | `containerDiskInGb` = `ceil(storage_mb / 1024)`, min 10 |

`resource_capabilities()` declares `cpu_request`/`memory_request` only, never
limits: RunPod's fields select a machine with *at least* those resources and
nothing caps the container afterwards.

> **CPU-pod caveat:** RunPod sizes RAM on CPU pods through the CPU *flavor*, not
> a free field. `memory_mb` therefore has no effect on a `gpus = 0` task. It is
> not silently reinterpreted; it is simply not expressible.

---

## Compose is not supported

`capabilities.docker_compose = False`, and a task with
`environment/docker-compose.yaml` is rejected in `_validate_definition()` with a
`ServiceOperationsUnsupportedError` — at construction time, before anything
bills.

A RunPod pod **is** a container, scheduled directly on a host. RunPod grants no
privileged mode and no Docker socket, so `docker compose` inside the pod is not
feasible; option (b) was evaluated and rejected on that basis. Rejecting is also
what `BaseEnvironment` asks for: providers must enforce the environment as
specified or refuse the task.

This rules out two of the three tasks in this repo
(`ehr-evidence-retrieval`, `verbatim-extraction`), whose whole isolation
property is that the answer key lives behind an HTTP boundary in a second
container. To run them on RunPod you would have to merge the sidecar into the
agent image and start it as a background process — which **destroys the
isolation guarantee**, because the agent could then read the service's files.
Do not do this silently. `crisis-early-detection` (single `Dockerfile`) is fine.

Per-service operations (`service_exec`, `stop_service`, sidecar artifact
collection) all raise `ServiceOperationsUnsupportedError`.

---

## Network policy

RunPod exposes **no egress firewall**. By default
(`network_enforcement = "none"`) this backend advertises
`disable_internet: false` and `network_allowlist: false`, so
`BaseEnvironment._validate_network_policy_support` **rejects** any task with
`network_mode = "no-network"` or `"allowlist"` rather than running it
unprotected.

All three tasks in this repo use `network_mode = "no-network"`, so they are
rejected by default. That is deliberate: silently granting a task full internet
when it asked for none is worse than not running it.

Opting in:

```bash
harbor run --env harbor_runpod:RunPodEnvironment \
  --ek network_enforcement=best-effort-iptables ...
```

This applies an `iptables` `HARBOR_EGRESS` chain inside the pod at `start()`
(and on every `set_network_policy`), accepting loopback plus
`ESTABLISHED,RELATED` — so the SSH session survives its own policy — then the
allowlist, then `REJECT`. IPv6 egress is dropped wholesale rather than left as
an unfiltered path around the IPv4 rules. If `iptables` is missing or the
container lacks `CAP_NET_ADMIN`, `start()` **fails loudly**; it never falls back
to running unprotected.

Read the name literally — this is *best effort*, and weaker than what a
compose/Kubernetes provider gives you:

* **Hostname entries are resolved once, on the pod, at apply time.** A host
  behind a CDN or DNS round-robin will have some of its addresses allowed and
  others not, and the set drifts as TTLs expire. Prefer IP/CIDR entries.
* DNS (udp/tcp 53) is left open in allowlist mode, or hostnames could not
  resolve at all. That is an exfiltration channel.
* Wildcard hostnames and all IPv6 allowlist forms are declared unsupported, so
  tasks using them are rejected rather than under-enforced.

---

## Cost and datacenter caveats

* **Pods bill by the hour, from creation until termination.** `stop(delete=True)`
  (Harbor's default) issues `DELETE /v1/pods/{id}`. If that call fails the error
  is logged at `error` level with the pod id — check
  <https://console.runpod.io/pods> if you see it.
* **`--no-delete` does not stop billing.** `stop(delete=False)` calls
  `POST /pods/{id}/stop`, which halts the container but keeps the disk, and
  **storage is still billed while stopped**.
* **Network volumes lock a pod to one datacenter.** Attaching
  `networkVolumeId` constrains placement to that volume's `dataCenterId`, which
  can sharply limit GPU availability and cause create failures that look like
  capacity errors. This backend therefore does not send `dataCenterIds` when a
  network volume is set, to avoid an unsatisfiable combination.
* **Spot pods (`interruptible=true`) can be reclaimed mid-trial.** There is no
  checkpoint/resume here; the trial fails.
* GPU SKU strings must match RunPod's `gpuTypeIds` exactly. The enum embedded in
  RunPod's OpenAPI document is stale — trust
  <https://docs.runpod.io/references/gpu-types> instead.
* Task env vars go into `PodCreateInput.env` and are visible in the RunPod
  console. Do not put secrets in `[environment].env` for this backend.

## API version

This backend targets **REST v1** (`https://rest.runpod.io/v1`). RunPod's docs
say v1 "is deprecated and will be retired in the near future" and recommend v2
— but v2's `CreatePodRequest` currently has no `dockerEntrypoint` (only `args`),
no `supportPublicIp`, and no `interruptible`, and this backend needs all three.
`api.py` takes `base_url` as an argument, so the move is a URL swap plus a field
remap once v2 grows them. See the module docstring in `api.py`.
