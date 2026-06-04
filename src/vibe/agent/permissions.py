from dataclasses import dataclass
from fnmatch import fnmatchcase
from glob import escape as glob_escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolPermissionRequest:
    name: str
    args: str
    target: str
    grant_pattern: str
    workspace_pattern: str | None = None
    raw_path: str | None = None


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    grant_pattern: str | None = None


def build_permission_request(tool_name: str, args: Any, workspace_root: str) -> ToolPermissionRequest:
    raw_path = _path_arg(args)
    if raw_path is None:
        return ToolPermissionRequest(
            name=tool_name,
            args=str(args),
            target=f"{tool_name}:*",
            grant_pattern=f"{tool_name}:*",
        )

    root = Path(workspace_root).resolve()
    try:
        resolved = _resolve_path(raw_path, root)
    except ValueError as error:
        return ToolPermissionRequest(
            name=tool_name,
            args=str(args),
            target=f"{tool_name}:<invalid-path>",
            grant_pattern=f"{tool_name}:*",
            workspace_pattern=f"error:{error}",
            raw_path=raw_path,
        )
    return ToolPermissionRequest(
        name=tool_name,
        args=str(args),
        target=f"{tool_name}:{_permission_path(resolved)}",
        grant_pattern=f"{tool_name}:*",
        workspace_pattern=f"{tool_name}:{glob_escape(_permission_path(root))}/*",
        raw_path=raw_path,
    )


def permission_matches(patterns: list[str], target: str) -> bool:
    return any(fnmatchcase(target, pattern) for pattern in patterns)


def normalize_decision(decision: PermissionDecision | bool) -> PermissionDecision:
    if isinstance(decision, PermissionDecision):
        return decision
    return PermissionDecision(allowed=decision)


def add_permission_grant(patterns: list[str], grant_pattern: str | None) -> list[str]:
    if grant_pattern is None or permission_matches(patterns, grant_pattern):
        return patterns
    return [*patterns, grant_pattern]


def workspace_permission_allows(request: ToolPermissionRequest) -> bool:
    if request.workspace_pattern is None:
        return True
    if request.workspace_pattern.startswith("error:"):
        return False
    if permission_matches([request.workspace_pattern], request.target):
        return True
    if request.workspace_pattern.endswith("/*"):
        return permission_matches([request.workspace_pattern[:-2]], request.target)
    return False


def workspace_permission_error(request: ToolPermissionRequest) -> str:
    if request.workspace_pattern and request.workspace_pattern.startswith("error:"):
        return f"Error: {request.workspace_pattern.removeprefix('error:')}"
    return f"Error: '{request.raw_path}' is outside the working directory"


def _path_arg(args: Any) -> str | None:
    if isinstance(args, dict):
        value = args.get("path")
        if isinstance(value, str):
            return value
    return None


def _resolve_path(raw_path: str, workspace_root: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _permission_path(path: Path) -> str:
    return path.as_posix()
