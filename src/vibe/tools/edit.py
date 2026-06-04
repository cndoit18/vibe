from typing import Annotated

from pydantic import Field

from vibe.tools.paths import workspace_file
from vibe.tools.runtime import tool


@tool
def edit(
    path: Annotated[str, Field(description="File path inside the current workspace.")],
    old_string: Annotated[str, Field(min_length=1, description="Exact text to replace.")],
    new_string: Annotated[str, Field(description="Replacement text.")],
    replace_all: Annotated[bool, Field(description="Replace every occurrence instead of requiring a unique match.")] = False,
) -> str:
    """Replace text in a workspace file.

    By default old_string must match exactly once; if it appears multiple times, include more surrounding context.
    """
    target = workspace_file(path)
    content = target.read_text()
    count = content.count(old_string)

    if count == 0:
        raise ValueError(f"old_string not found in '{target.display}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string appears {count} times in '{target.display}'; provide a larger old_string to make the match unique"
        )

    updated = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
    target.write_text(updated)
    return f"Edited '{target.display}': replaced {count if replace_all else 1} occurrence(s)"
