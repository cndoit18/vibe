from __future__ import annotations

import os
import shlex
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path, PurePosixPath
from typing import Any, TYPE_CHECKING

try:
    from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
except ImportError:

    class BaseInstalledAgent:
        def __init__(
            self,
            *args: Any,
            logs_dir: str | Path | None = None,
            model_name: str | None = None,
            extra_env: dict[str, str] | None = None,
            **kwargs: Any,
        ) -> None:
            self.logs_dir = Path(logs_dir) if logs_dir is not None else None
            self.model_name = model_name
            self._extra_env = extra_env or {}

        async def exec_as_root(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("harbor is required to execute VibeInstalledAgent")

        async def exec_as_agent(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("harbor is required to execute VibeInstalledAgent")

    def with_prompt_template(func: Any) -> Any:
        return func

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
else:
    BaseEnvironment = Any
    AgentContext = Any

REMOTE_REPO_DIR = "/installed-agent/vibe"
VENV_DIR = "/opt/vibe-venv"
LOG_PATH = "/logs/agent/vibe.txt"


class VibeInstalledAgent(BaseInstalledAgent):
    def __init__(
        self,
        *args: Any,
        source_dir: str | Path | None = None,
        install_dir: str = REMOTE_REPO_DIR,
        venv_dir: str = VENV_DIR,
        log_path: str = LOG_PATH,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_dir = _resolve_source_dir(source_dir)
        self.install_dir = _posix_path(install_dir)
        self.venv_dir = _posix_path(venv_dir)
        self.vibe_bin = _posix_path(PurePosixPath(self.venv_dir) / "bin" / "vibe")
        self.venv_python = _posix_path(PurePosixPath(self.venv_dir) / "bin" / "python")
        self.log_path = _posix_path(log_path)
        self.log_dir = _posix_path(PurePosixPath(self.log_path).parent)

    @staticmethod
    def name() -> str:
        return "vibe"

    def version(self) -> str | None:
        try:
            return package_version("vibe")
        except PackageNotFoundError:
            return None

    async def install(self, environment: BaseEnvironment) -> None:
        install_parent = _posix_path(PurePosixPath(self.install_dir).parent)
        await self.exec_as_root(
            environment,
            command=(
                f"rm -rf {_quote(self.install_dir)} {_quote(self.venv_dir)} && "
                f"mkdir -p {_quote(install_parent)} {_quote('/opt')} {_quote(self.log_dir)}"
            ),
        )
        await environment.upload_dir(self.source_dir, self.install_dir)
        await self.exec_as_root(
            environment,
            command=(
                f"python3 -m venv {_quote(self.venv_dir)} && "
                f"{_quote(self.venv_python)} -m pip install --upgrade pip && "
                f"{_quote(self.venv_python)} -m pip install -e {_quote(self.install_dir)} && "
                f"{_quote(self.vibe_bin)} --help >/dev/null"
            ),
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {_quote(self.log_dir)} && "
                f"{_quote(self.vibe_bin)} --print -- {shlex.quote(instruction)} "
                f"2>&1 | tee {_quote(self.log_path)}"
            ),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        return None


def _resolve_source_dir(source_dir: str | Path | None) -> Path:
    if source_dir is not None:
        return Path(source_dir).expanduser().resolve()

    env_source_dir = os.environ.get("VIBE_HARBOR_SOURCE_DIR")
    if env_source_dir:
        return Path(env_source_dir).expanduser().resolve()

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "vibe").is_dir():
            return candidate
    raise ValueError("Set source_dir or VIBE_HARBOR_SOURCE_DIR to the Vibe source checkout")


def _posix_path(path: str | PurePosixPath) -> str:
    return PurePosixPath(path).as_posix()


def _quote(value: str) -> str:
    return shlex.quote(value)
