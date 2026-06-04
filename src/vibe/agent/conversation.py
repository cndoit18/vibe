from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import vibe  # noqa: F401 — ensures deprecation-warning filter runs before langchain imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from vibe.agent.graph import build_graph
from vibe.agent.permissions import PermissionDecision, ToolPermissionRequest
from vibe.session.store import SessionStore
from vibe.tools import bash, edit, read, write


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class AgentConversation:
    def __init__(
        self,
        session_id: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        permission_callback: Callable[[ToolPermissionRequest], PermissionDecision | bool] | None = None,
    ):
        self.store = SessionStore()
        self.session_id = session_id or self.store.new_session()
        self.permission_grants: list[str] = []
        llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)
        self.agent = build_graph(llm, [bash, read, write, edit], permission_callback)

    def load_history(self) -> list[BaseMessage]:
        return self.store.load(self.session_id)

    def send(self, prompt: str) -> Iterator[AgentEvent]:
        history = self.load_history()
        history.append(HumanMessage(content=prompt))
        self.store.append(self.session_id, history[-1])

        for update in self.agent.stream(
            {
                "messages": history,
                "permission_grants": self.permission_grants,
                "workspace_root": str(Path.cwd().resolve()),
            },
            stream_mode="updates",
        ):
            for node_update in update.values():
                if "permission_grants" in node_update:
                    self.permission_grants = node_update["permission_grants"]
                messages = node_update.get("messages", [])
                if not isinstance(messages, list):
                    messages = [messages]
                for msg in messages:
                    self.store.append(self.session_id, msg)
                    yield from self._events_from_message(msg)

    def _events_from_message(self, msg) -> Iterator[AgentEvent]:
        if getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                yield AgentEvent(
                    "tool_call", str(tool_call["args"]), tool_call["name"], tool_call_id=tool_call.get("id")
                )
        elif msg.type == "tool":
            yield AgentEvent(
                "tool_result",
                str(msg.content),
                getattr(msg, "name", None),
                tool_call_id=getattr(msg, "tool_call_id", None),
            )
        elif msg.type == "ai" and msg.content:
            yield AgentEvent("assistant", str(msg.content))
