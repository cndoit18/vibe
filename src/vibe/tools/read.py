from typing import Annotated

from pydantic import Field

from vibe.tools.paths import workspace_file
from vibe.tools.runtime import tool


@tool
def read(
    path: Annotated[str, Field(description="File path inside the current workspace.")],
    offset: Annotated[int, Field(ge=0, description="Zero-based line offset to start reading from.")] = 0,
    limit: Annotated[int, Field(ge=1, le=2000, description="Maximum number of lines to return.")] = 2000,
) -> str:
    """Read a workspace file.

    Use this before editing or when you need exact file contents. Use offset and limit to inspect large files in
    smaller chunks.
    """
    lines = workspace_file(path).read_text().splitlines()
    selected = lines[offset : offset + limit]
    if not selected:
        return "(no lines in range)"

    header = ""
    if offset > 0:
        header = f"(showing lines {offset}-{offset + len(selected) - 1} of {len(lines)})\n"

    return header + "\n".join(selected)
