#!/bin/sh
# Run one task on RunPod instead of the local Docker daemon.
#
#   ./run_runpod.sh tasks/budget-triage                 # oracle smoke test
#   ./run_runpod.sh tasks/budget-triage --agent kimi    # the real thing
#
# Anything after the task directory is forwarded to `harbor run` verbatim.
#
# This wrapper exists because three things about running Harbor here are
# non-obvious and each one fails in a way that does not name its own cause.
# They are handled in preflight() below and commented individually.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV="$ROOT/.venv/bin"
CONSOLE_URL="https://console.runpod.io/pods"

die() {
    printf '%s\n' "$@" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

[ $# -ge 1 ] || die "usage: $0 <task-dir>... [harbor run args...]" \
    "example: $0 tasks/budget-triage --agent kimi" \
    "example: $0 tasks/* --agent kimi -n 4"

# Leading arguments are task directories; the first thing starting with '-'
# begins the harbor passthrough. TASK_DIRS is whitespace-separated and later
# re-split, so a task path containing a space would break -- rejected here
# rather than producing a silently truncated job.
TASK_DIRS=""
TASK_COUNT=0
while [ $# -ge 1 ]; do
    case "$1" in -*) break ;; esac
    dir=$(CDPATH= cd -- "$1" 2>/dev/null && pwd) \
        || die "No such task directory: $1"
    [ -f "$dir/task.toml" ] || die "Not a Harbor task (no task.toml): $dir"
    case "$dir" in *[[:space:]]*) die "Task path contains whitespace: $dir" ;; esac
    TASK_DIRS="$TASK_DIRS $dir"
    TASK_COUNT=$((TASK_COUNT + 1))
    shift
done
[ "$TASK_COUNT" -ge 1 ] || die "No task directory given (first argument was an option)."

# Default to the reference solution. The first RunPod run of any task should be
# a ~1 minute oracle check, not a 3600 s agent run against a broken image.
case " $* " in
    *" --agent "*) ;;
    *) set -- --agent oracle "$@" ;;
esac

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

preflight() {
    # (1) `--task <path>` is NOT a path. Harbor parses it as an `org/name`
    #     package reference and fails with "Task version not found:
    #     tasks/budget-triage@latest". A job config with an absolute `path` is
    #     the spelling that means "local directory".
    #     The .json suffix is load-bearing: harbor picks the config parser off
    #     the file extension, so a bare mktemp name yields
    #     "Unsupported config file format: .xstunudzsv". Renaming rather than
    #     `mktemp --suffix` keeps this working on non-GNU mktemp.
    JOB_CONFIG=$(mktemp)
    mv "$JOB_CONFIG" "$JOB_CONFIG.json"
    JOB_CONFIG="$JOB_CONFIG.json"
    trap 'rm -f "$JOB_CONFIG"' EXIT
    {
        printf '{"tasks":['
        sep=""
        for dir in $TASK_DIRS; do
            printf '%s{"path":"%s"}' "$sep" "$dir"
            sep=","
        done
        printf ']}\n'
    } > "$JOB_CONFIG"

    # (2) harbor_runpod is not installed into the venv, and the `harbor`
    #     console script puts .venv/bin on sys.path[0] -- never the cwd. So an
    #     uninstalled backend is invisible no matter where you run this from.
    [ -f "$ROOT/harbor_runpod/runpod_env.py" ] \
        || die "harbor_runpod/ not found next to this script."
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONPATH

    # (3) Docker group membership is often not in the current process's
    #     supplementary groups even when /etc/group says it should be. Harbor
    #     reports the resulting EACCES as "Docker daemon is not running",
    #     which sends you to systemctl instead of to `sg`.
    if docker info >/dev/null 2>&1; then
        SG_DOCKER=0
    elif sg docker -c "docker info" >/dev/null 2>&1; then
        SG_DOCKER=1
        echo "note: re-entering the docker group via sg(1)."
    else
        die "Cannot reach the Docker daemon." \
            "RunPod has no build service, so the task image must be built" \
            "locally and pushed. Start Docker, or add yourself to the" \
            "'docker' group."
    fi

    [ -f "$ROOT/.env" ] && . "$ROOT/.env"
    [ -n "${RUNPOD_API_KEY:-}" ] || die "RUNPOD_API_KEY is not set (.env)."

    # The image registry is only required for tasks that build from a
    # Dockerfile -- which is all four of ours. Checked here rather than letting
    # the backend raise, so the failure costs nothing and names the variable.
    if [ -z "${HARBOR_RUNPOD_IMAGE:-}" ] && [ -z "${HARBOR_RUNPOD_IMAGE_REGISTRY:-}" ]; then
        die "HARBOR_RUNPOD_IMAGE_REGISTRY is not set." \
            "" \
            "RunPod cannot build images, only pull them, and every task here" \
            "has an environment/Dockerfile. Use GHCR, not Docker Hub: the" \
            "verifier image carries the hidden labels (tests/test_alerts.json)" \
            "and shares one repo with the agent image, so the repo must be" \
            "private -- Docker Hub's free tier allows exactly one." \
            "" \
            "  export CR_PAT=<GitHub PAT with write:packages>" \
            "  echo \$CR_PAT | docker login ghcr.io -u <user> --password-stdin" \
            "  export HARBOR_RUNPOD_IMAGE_REGISTRY=ghcr.io/<user>" \
            "  export HARBOR_RUNPOD_REGISTRY_USERNAME=<user>" \
            "  export HARBOR_RUNPOD_REGISTRY_PASSWORD=\$CR_PAT"
    fi

    export RUNPOD_API_KEY
    [ -n "${KIMI_API_KEY:-}" ] && export KIMI_API_KEY
    for v in HARBOR_RUNPOD_IMAGE HARBOR_RUNPOD_IMAGE_REGISTRY \
             HARBOR_RUNPOD_REGISTRY_USERNAME HARBOR_RUNPOD_REGISTRY_PASSWORD \
             HARBOR_RUNPOD_START_TIMEOUT_SEC HARBOR_RUNPOD_CLOUD_TYPE; do
        eval "[ -n \"\${$v:-}\" ]" && export "$v" || :
    done
}

preflight

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

cat <<EOF
tasks    : $TASK_COUNT -$(for d in $TASK_DIRS; do printf ' %s' "$(basename "$d")"; done)
agent    : $(echo "$@" | sed -n 's/.*--agent \([^ ]*\).*/\1/p')
env      : harbor_runpod:RunPodEnvironment
registry : ${HARBOR_RUNPOD_IMAGE_REGISTRY:-<pinned image: ${HARBOR_RUNPOD_IMAGE:-none}>}
pods     : named harbor-<session-id>, visible at $CONSOLE_URL

Reminder: pods bill from creation until termination. --no-delete does NOT stop
billing; it keeps the disk, which is billed too.

EOF

# Pin the results directory to the repo. Harbor's -o defaults to the relative
# path "jobs", so invoking this script from anywhere else silently scatters
# results into <that cwd>/jobs -- which is how one codex run ended up in
# /data/projects/jobs while every other run was in the repo. Only supplied when
# the caller did not choose their own.
case " $* " in
    *" -o "*|*" --jobs-dir "*) ;;
    *) set -- "$@" -o "$ROOT/jobs" ;;
esac

# Build the command in a file so `sg -c` needs no quoting gymnastics.
CMD_FILE=$(mktemp)
trap 'rm -f "$JOB_CONFIG" "$CMD_FILE"' EXIT
{
    printf 'export PYTHONPATH=%s\n' "$PYTHONPATH"
    printf 'exec %s/harbor run -c %s --env harbor_runpod:RunPodEnvironment' \
        "$VENV" "$JOB_CONFIG"
    for arg in "$@"; do
        printf " '%s'" "$(printf '%s' "$arg" | sed "s/'/'\\\\''/g")"
    done
    printf '\n'
} > "$CMD_FILE"
chmod +x "$CMD_FILE"

set +e
if [ "$SG_DOCKER" = "1" ]; then
    sg docker -c "$CMD_FILE"
else
    sh "$CMD_FILE"
fi
STATUS=$?
set -e

# A non-zero exit is exactly when a pod is most likely to have outlived its
# trial, so say this on the failure path rather than burying it in a log.
if [ "$STATUS" -ne 0 ]; then
    printf '\nharbor exited %s. Check for surviving pods at %s\n' \
        "$STATUS" "$CONSOLE_URL" >&2
fi
exit "$STATUS"
