import subprocess

from langchain_core.tools import tool


@tool
def bash(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds. Defaults to 30.
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
