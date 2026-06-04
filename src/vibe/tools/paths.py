from dataclasses import dataclass
from pathlib import Path

MAX_FILE_SIZE = 1_000_000


@dataclass(frozen=True)
class WorkspacePath:
    raw: str
    resolved: Path
    max_bytes: int = MAX_FILE_SIZE

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


def workspace_path(raw: str, *, create_parents: bool = False, max_bytes: int = MAX_FILE_SIZE) -> WorkspacePath:
    resolved = Path(raw).resolve()
    try:
        resolved.relative_to(Path.cwd())
    except ValueError:
        raise ValueError(f"'{raw}' is outside the working directory")

    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"'{raw}' is not a file")
    if create_parents:
        if resolved.parent.exists() and not resolved.parent.is_dir():
            raise ValueError(f"parent path for '{raw}' is not a directory")
        resolved.parent.mkdir(parents=True, exist_ok=True)

    return WorkspacePath(raw=raw, resolved=resolved, max_bytes=max_bytes)


def workspace_file(raw: str, *, max_bytes: int = MAX_FILE_SIZE) -> WorkspacePath:
    path = workspace_path(raw, max_bytes=max_bytes)
    if not path.resolved.is_file():
        raise ValueError(f"'{raw}' is not a file")

    return path
