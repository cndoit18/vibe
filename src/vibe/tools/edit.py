from vibe.tools.hooks import MaxFileBytes, MaxTextBytes, NonEmptyText, WorkspacePath, WorkspacePathValue, tool


@tool(
    WorkspacePath("path", require_file=True),
    MaxFileBytes("path"),
    NonEmptyText("old_string"),
    MaxTextBytes("new_string"),
)
def edit(path: WorkspacePathValue, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file. Fails if old_string is not found or appears more than once.

    Args:
        path: File path relative to working directory.
        old_string: The exact text to find and replace.
        new_string: The replacement text.
    """
    content = path.read_text()
    count = content.count(old_string)

    if count == 0:
        raise ValueError(f"old_string not found in '{path.display}'")
    if count > 1:
        raise ValueError(f"old_string appears {count} times in '{path.display}' — must be unique")

    updated = content.replace(old_string, new_string, 1)
    path.write_text(updated)
    return f"Edited '{path.display}': replaced {len(old_string)} chars with {len(new_string)} chars"
