import pytest
from langchain_core.messages import AIMessage, HumanMessage

from vibe.session.store import SessionStore


class TestNewSession:
    def test_returns_hex_session_id(self, store: SessionStore):
        session_id = store.new_session()
        assert len(session_id) == 8
        assert session_id.isalnum()

    def test_creates_jsonl_file(self, store: SessionStore, tmp_session_dir):
        session_id = store.new_session()
        assert (tmp_session_dir / f"{session_id}.jsonl").exists()

    def test_creates_session_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "sessions"
        SessionStore(nested).new_session()
        assert nested.exists()


@pytest.mark.parametrize("msg_cls,content,expected_type", [
    (HumanMessage, "hello", "human"),
    (AIMessage, "world", "ai"),
    (HumanMessage, "你好世界", "human"),
])
def test_append_and_load(store: SessionStore, msg_cls, content, expected_type):
    session_id = store.new_session()
    store.append(session_id, msg_cls(content=content))

    loaded = store.load(session_id)
    assert len(loaded) == 1
    assert loaded[0].content == content
    assert loaded[0].type == expected_type


def test_multiple_messages(store: SessionStore):
    session_id = store.new_session()
    store.append(session_id, HumanMessage(content="q1"))
    store.append(session_id, AIMessage(content="a1"))
    store.append(session_id, HumanMessage(content="q2"))

    loaded = store.load(session_id)
    assert [m.content for m in loaded] == ["q1", "a1", "q2"]


class TestLoad:
    def test_nonexistent_session_returns_empty(self, store: SessionStore):
        assert store.load("nonexistent") == []

    def test_empty_session_returns_empty(self, store: SessionStore):
        session_id = store.new_session()
        assert store.load(session_id) == []


class TestListSessions:
    def test_empty(self, store: SessionStore):
        assert store.list_sessions() == []

    def test_lists_sessions_sorted(self, store: SessionStore):
        store.new_session()
        store.new_session()
        store.new_session()

        sessions = store.list_sessions()
        assert len(sessions) == 3
        assert sessions == sorted(sessions)
