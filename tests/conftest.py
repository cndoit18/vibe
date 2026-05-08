import pytest
from pathlib import Path

from vibe.session.store import SessionStore


@pytest.fixture
def tmp_session_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def store(tmp_session_dir: Path) -> SessionStore:
    return SessionStore(tmp_session_dir)
