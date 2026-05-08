from vibe.tools._safety import check_file_size, tool, validate_path


@tool
def edit(path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file. Fails if old_string is not found or appears more than once.

    Args:
        path: File path relative to working directory.
        old_string: The exact text to find and replace.
        new_string: The replacement text.
    """
    resolved = validate_path(path)
    if not resolved.is_file():
        raise ValueError(f"'{path}' is not a file")
    check_file_size(resolved)

    content = resolved.read_text()
    count = content.count(old_string)

    if count == 0:
        raise ValueError(f"old_string not found in '{path}'")
    if count > 1:
        raise ValueError(f"old_string appears {count} times in '{path}' — must be unique")

    updated = content.replace(old_string, new_string, 1)
    resolved.write_text(updated)
    return f"Edited '{path}': replaced {len(old_string)} chars with {len(new_string)} chars"
