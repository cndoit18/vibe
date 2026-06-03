from importlib.resources import files

_PROMPTS_PKG = "vibe.prompts"


def load_prompt(name: str) -> str:
    """Load a prompt markdown file by name (without extension)."""
    return files(_PROMPTS_PKG).joinpath(f"{name}.md").read_text(encoding="utf-8")
