from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from vibe.agent.graph import build_graph, should_continue, SYSTEM_PROMPT
from vibe.agent.state import AgentState
from vibe.tools import bash, edit, read, write


@pytest.fixture
def compiled_graph():
    llm = MagicMock()
    return build_graph(llm, [bash, read, write, edit])


class TestBuildGraph:
    def test_compiles_and_is_callable(self, compiled_graph):
        assert compiled_graph is not None
        assert callable(compiled_graph.invoke)

    def test_system_prompt_non_empty(self):
        assert SYSTEM_PROMPT.strip()


class TestShouldContinue:
    def test_returns_tools_when_ai_has_tool_calls(self):
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "1"}])]
        )
        assert should_continue(state) == "tools"

    def test_returns_end_when_ai_no_tool_calls(self):
        state = AgentState(messages=[AIMessage(content="done")])
        assert should_continue(state) == "end"

    def test_returns_end_for_human_message(self):
        state = AgentState(messages=[HumanMessage(content="hi")])
        assert should_continue(state) == "end"
