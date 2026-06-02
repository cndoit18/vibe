from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from vibe.tools.runtime import Hook, RuntimeTool, ToolContext

MAX_OUTPUT_CHARS = 30_000
MAX_FILE_SIZE = 1_000_000


@dataclass(frozen=True)
class WorkspacePathValue:
    raw: str
    resolved: Path
    max_bytes: int = MAX_FILE_SIZE
    __tool_external_annotation__ = str

    @property
    def display(self) -> str:
        return self.raw

    def read_text(self) -> str:
        return self.resolved.read_text(encoding="utf-8", errors="replace")

    def write_text(self, content: str) -> None:
        size = len(content.encode("utf-8"))
        if size > self.max_bytes:
            raise ValueError(f"updated content exceeds {self.max_bytes} byte limit")
        self.resolved.write_text(content, encoding="utf-8", errors="replace")


class DynamicHooks(Hook):
    def __init__(self, resolver: Callable[[ToolContext], Iterable[Hook]]) -> None:
        self.resolver = resolver

    def prepare(self, ctx: ToolContext) -> None:
        ctx.hooks.extend(self.resolver(ctx))


class WorkspacePath(Hook):
    def __init__(self, name: str, *, require_file: bool = False, create_parents: bool = False) -> None:
        self.name = name
        self.require_file = require_file
        self.create_parents = create_parents

    def prepare(self, ctx: ToolContext) -> None:
        raw = ctx.args[self.name]
        resolved = Path(raw).resolve()
        try:
            resolved.relative_to(Path.cwd())
        except ValueError:
            raise ValueError(f"Path '{raw}' is outside the working directory")

        if self.require_file and not resolved.is_file():
            raise ValueError(f"'{raw}' is not a file")
        if resolved.exists() and not resolved.is_file():
            raise ValueError(f"'{raw}' is not a file")
        if self.create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)

        ctx.args[self.name] = WorkspacePathValue(raw=raw, resolved=resolved)


class MaxFileBytes(Hook):
    def __init__(self, name: str, *, max_bytes: int = MAX_FILE_SIZE) -> None:
        self.name = name
        self.max_bytes = max_bytes

    def prepare(self, ctx: ToolContext) -> None:
        path = ctx.args[self.name]
        resolved = path.resolved if isinstance(path, WorkspacePathValue) else Path(path).resolve()
        size = resolved.stat().st_size
        if size > self.max_bytes:
            raise ValueError(f"File '{resolved}' is {size} bytes, exceeds {self.max_bytes} byte limit")


class MaxTextBytes(Hook):
    def __init__(self, name: str, *, max_bytes: int = MAX_FILE_SIZE) -> None:
        self.name = name
        self.max_bytes = max_bytes

    def prepare(self, ctx: ToolContext) -> None:
        size = len(ctx.args[self.name].encode("utf-8"))
        if size > self.max_bytes:
            raise ValueError(f"{self.name} exceeds {self.max_bytes} byte limit")


class IntRange(Hook):
    def __init__(self, name: str, *, min: int | None = None, max: int | None = None) -> None:
        self.name = name
        self.min = min
        self.max = max

    def prepare(self, ctx: ToolContext) -> None:
        value = ctx.args[self.name]
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.name} must be >= {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.name} must be <= {self.max}")


class NonEmptyText(Hook):
    def __init__(self, name: str) -> None:
        self.name = name

    def prepare(self, ctx: ToolContext) -> None:
        if ctx.args[self.name] == "":
            raise ValueError(f"{self.name} must not be empty")


class TruncateResult(Hook):
    def __init__(self, *, max_chars: int = MAX_OUTPUT_CHARS) -> None:
        self.max_chars = max_chars

    def after(self, ctx: ToolContext) -> None:
        if not isinstance(ctx.result, str) or len(ctx.result) <= self.max_chars:
            return
        ctx.result = ctx.result[: self.max_chars] + f"\n... [truncated, {len(ctx.result) - self.max_chars} more chars]"


class ReturnValueError(Hook):
    def on_error(self, ctx: ToolContext, error: Exception) -> str | None:
        if isinstance(error, ValueError):
            return f"Error: {error}"
        return None


def tool(*hooks: Hook):
    def decorate(func: Callable[..., str]) -> RuntimeTool:
        runtime_hooks = (ReturnValueError(), *hooks)
        return RuntimeTool(func=func, hooks=runtime_hooks)

    return decorate


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"
