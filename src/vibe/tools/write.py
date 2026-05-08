from vibe.tools._safety import MAX_FILE_SIZE, tool, validate_path


@tool
def write(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Args:
        path: File path relative to working directory.
        content: The text content to write.
    """
    if len(content.encode("utf-8")) > MAX_FILE_SIZE:
        raise ValueError(f"content exceeds {MAX_FILE_SIZE} byte limit")

    resolved = validate_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return f"Wrote {len(content)} chars to '{path}'"
