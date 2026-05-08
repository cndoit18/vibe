import json
import uuid
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

DEFAULT_SESSION_DIR = Path.home() / ".vibe" / "sessions"


class SessionStore:
    """JSONL 会话持久化。每条消息一行，追加写入。"""

    def __init__(self, session_dir: Path = DEFAULT_SESSION_DIR):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def new_session(self) -> str:
        session_id = uuid.uuid4().hex[:8]
        (self.session_dir / f"{session_id}.jsonl").touch()
        return session_id

    def append(self, session_id: str, message: BaseMessage):
        path = self.session_dir / f"{session_id}.jsonl"
        entry = messages_to_dict([message])[0]
        with open(path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self, session_id: str) -> list[BaseMessage]:
        path = self.session_dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        if not entries:
            return []
        return messages_from_dict(entries)

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.session_dir.glob("*.jsonl"))
