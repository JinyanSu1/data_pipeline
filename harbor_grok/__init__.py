"""Grok Build with subscription (grok.com login) auth instead of an API key.

    harbor run --agent harbor_grok:GrokBuildAuthJson --model grok-4.5 ...

Harbor's own ``grok-build`` agent only supports ``XAI_API_KEY``. Its docstring
says so outright, and ``run()`` raises before doing anything if the variable is
missing. But the grok CLI itself authenticates perfectly well from
``~/.grok/auth.json`` written by ``grok login`` -- the same shape of credential
Harbor's *codex* agent already knows how to transplant. This class closes that
gap for grok, and nothing else.

Verified locally before this was written, because each of these could have gone
the other way:

  * ``grok --single ... --output-format streaming-json`` (the exact invocation
    Harbor builds) succeeds with ``XAI_API_KEY`` unset, using only the
    logged-in credentials.
  * A junk ``XAI_API_KEY`` does *not* override those credentials -- grok still
    answers correctly. That is what makes the placeholder below safe: the
    parent's mandatory key check can be satisfied without hijacking auth.

Uploading a subscription credential to a rented machine is a real exposure, not
a formality. ``~/.grok/auth.json`` holds an OIDC access token, a refresh token,
and account PII (email, name, user id). It travels inside the SSH channel and
never enters RunPod's PodCreateInput, so it stays out of the console -- but it
does sit on the pod's disk until the pod is deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.grok_build import GrokBuild
from harbor.environments.base import BaseEnvironment

__all__ = ["GrokBuildAuthJson"]

# Grok reads auth.json ahead of the environment, so this is inert. It exists
# only so GrokBuild.run()'s `if not api_key: raise` does not fire. Anything
# truthy works; the value is deliberately self-describing in case it ever
# surfaces in a log.
_PLACEHOLDER_KEY = "unused-auth-json-mode"

_STAGING_PATH = "/tmp/.grok-auth-upload.json"


class GrokBuildAuthJson(GrokBuild):
    """``grok-build`` that authenticates from a host ``auth.json``.

    Resolution order for the credential:
      1. ``GROK_AUTH_JSON_PATH=<path>``  (kwarg or env)
      2. ``~/.grok/auth.json``           (the default ``grok login`` location)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._auth_json_path = self._resolve_auth_json_path()

    def _resolve_auth_json_path(self) -> Path:
        explicit = self._get_env("GROK_AUTH_JSON_PATH")
        path = Path(explicit) if explicit else Path.home() / ".grok" / "auth.json"
        if not path.is_file():
            raise ValueError(
                f"{type(self).__name__} needs a grok credential file and "
                f"{path} does not exist. Run `grok login` on this host, or "
                "point GROK_AUTH_JSON_PATH at an existing auth.json. To use an "
                "API key instead, use Harbor's stock `--agent grok-build`."
            )
        try:
            json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            # Cheap to check here, and the alternative is a pod that boots,
            # installs grok, and only then fails on a malformed credential.
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        return path

    @override
    def _get_env(self, key: str) -> str | None:
        # GrokBuild.run() aborts unless this resolves. Satisfy it without
        # letting the value reach anything that would use it as a real key:
        # only the xAI variable is faked, and only when it is otherwise unset,
        # so an explicitly supplied key still wins.
        value = super()._get_env(key)
        if value is None and key == "XAI_API_KEY":
            return _PLACEHOLDER_KEY
        return value

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)

        # Two steps rather than one: upload_file lands as root, and grok runs
        # as the agent user. Staging in /tmp and copying as the agent avoids
        # having to resolve the agent's $HOME on the host side at all.
        await environment.upload_file(self._auth_json_path, _STAGING_PATH)
        result = await self.exec_as_agent(
            environment,
            command=(
                'mkdir -p "$HOME/.grok" && '
                f'cp {_STAGING_PATH} "$HOME/.grok/auth.json" && '
                'chmod 600 "$HOME/.grok/auth.json"'
            ),
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to install grok auth.json in the environment "
                f"(exit {result.return_code}): {result.stderr}"
            )
        # Best effort: the staged copy is root-owned and readable, so drop it
        # rather than leave a second copy of the credential lying around.
        await self.exec_as_root(environment, command=f"rm -f {_STAGING_PATH}")
