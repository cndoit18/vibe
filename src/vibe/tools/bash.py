import subprocess
from typing import Annotated

from pydantic import Field

from vibe.tools.runtime import tool


@tool
def bash(
    command: Annotated[str, Field(description="Shell command to execute.")],
    timeout: Annotated[int, Field(ge=1, description="Maximum execution time in seconds.")] = 30,
) -> str:
    """Execute a shell command and return its output.

    Use this for commands that inspect or operate on the workspace. Prefer narrower commands when possible.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
