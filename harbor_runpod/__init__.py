"""RunPod execution backend for the Harbor agent-evaluation framework.

Harbor's built-in environments are selected from a closed ``EnvironmentType``
enum, so an out-of-tree provider is referenced by import path instead::

    harbor run --env harbor_runpod:RunPodEnvironment ...

See ``README.md`` for the full registration story, required credentials, and
the compose / image-build limitations.
"""

from harbor_runpod.api import (
    RunPodAPIError,
    RunPodClient,
    RunPodTransport,
    pod_ssh_endpoint,
)
from harbor_runpod.runpod_env import (
    ENVIRONMENT_TYPE,
    RunPodConfigurationError,
    RunPodEnvironment,
    RunPodSSHError,
)

__all__ = [
    "ENVIRONMENT_TYPE",
    "RunPodAPIError",
    "RunPodClient",
    "RunPodConfigurationError",
    "RunPodEnvironment",
    "RunPodSSHError",
    "RunPodTransport",
    "pod_ssh_endpoint",
]

__version__ = "0.1.0"
