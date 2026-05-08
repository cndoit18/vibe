from vibe.tools._safety import check_file_size, tool, truncate_output, validate_path


@tool
def read(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read file contents. Returns lines from offset (0-based) up to limit lines.

    Args:
        path: File path relative to working directory.
        offset: Line number to start reading from (0-based). Defaults to 0.
        limit: Maximum number of lines to read. Defaults to 2000.
    """
    resolved = validate_path(path)
    if not resolved.is_file():
        raise ValueError(f"'{path}' is not a file")
    check_file_size(resolved)

    lines = resolved.read_text(errors="replace").splitlines()
    selected = lines[offset : offset + limit]
    if not selected:
        return "(no lines in range)"

    header = ""
    if offset > 0:
        header = f"(showing lines {offset}-{offset + len(selected) - 1} of {len(lines)})\n"

    return truncate_output(header + "\n".join(selected))
