from vibe.tools.hooks import IntRange, MaxFileBytes, TruncateResult, WorkspacePath, WorkspacePathValue, tool


@tool(
    WorkspacePath("path", require_file=True),
    MaxFileBytes("path"),
    IntRange("offset", min=0),
    IntRange("limit", min=1),
    TruncateResult(),
)
def read(path: WorkspacePathValue, offset: int = 0, limit: int = 2000) -> str:
    """Read file contents. Returns lines from offset (0-based) up to limit lines.

    Args:
        path: File path relative to working directory.
        offset: Line number to start reading from (0-based). Defaults to 0.
        limit: Maximum number of lines to read. Defaults to 2000.
    """
    lines = path.read_text().splitlines()
    selected = lines[offset : offset + limit]
    if not selected:
        return "(no lines in range)"

    header = ""
    if offset > 0:
        header = f"(showing lines {offset}-{offset + len(selected) - 1} of {len(lines)})\n"

    return header + "\n".join(selected)
