from vibe.tools.hooks import MaxTextBytes, WorkspacePath, WorkspacePathValue, tool


@tool(
    WorkspacePath("path", create_parents=True),
    MaxTextBytes("content"),
)
def write(path: WorkspacePathValue, content: str) -> str:
    """Create or overwrite a file with the given content.

    Args:
        path: File path relative to working directory.
        content: The text content to write.
    """
    path.write_text(content)
    return f"Wrote {len(content)} chars to '{path.display}'"
