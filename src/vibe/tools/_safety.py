"""Shared safety utilities for file tools."""

import functools
import os
from pathlib import Path

from langchain_core.tools import tool as lc_tool

MAX_OUTPUT_CHARS = 30_000
MAX_FILE_SIZE = 1_000_000  # 1 MB


def tool(func):
    """Drop-in replacement for @tool that catches ValueError and returns it as an error string."""
    original = func

    @functools.wraps(func)
    def _safe(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ValueError as e:
            return f"Error: {e}"

    return lc_tool(_safe)


def validate_path(path: str) -> Path:
    """Resolve and validate a file path. Rejects paths outside the working directory tree."""
    resolved = Path(path).resolve()
    cwd = Path.cwd()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise ValueError(f"Path '{path}' is outside the working directory")
    return resolved


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate text if it exceeds max_chars, appending a notice."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"


def check_file_size(path: Path, max_bytes: int = MAX_FILE_SIZE) -> None:
    """Raise if file exceeds the size limit."""
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"File '{path}' is {size} bytes, exceeds {max_bytes} byte limit")
