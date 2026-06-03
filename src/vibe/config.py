import json
from dataclasses import dataclass
from pathlib import Path

SETTINGS_PATH = Path.home() / ".vibe" / "settings.json"


@dataclass(frozen=True)
class Settings:
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


def load_settings() -> Settings:
    try:
        if not SETTINGS_PATH.exists():
            return Settings()
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return Settings()
        return Settings(
            model=data.get("model"),
            base_url=data.get("base_url"),
            api_key=data.get("api_key"),
        )
    except (json.JSONDecodeError, OSError):
        return Settings()
