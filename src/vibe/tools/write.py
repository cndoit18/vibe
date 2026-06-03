from typing import Annotated

from pydantic import Field

from vibe.tools.paths import workspace_path
from vibe.tools.runtime import tool


@tool
def write(
    path: Annotated[str, Field(description="File path inside the current workspace.")],
    content: Annotated[str, Field(description="Text content to write to the file.")],
) -> str:
    """Create or overwrite a workspace file.

    Use this when you need to write complete file contents. Parent directories are created automatically.
    """
    target = workspace_path(path, create_parents=True)
    target.write_text(content)
    return f"Wrote {len(content)} chars to '{target.display}'"
