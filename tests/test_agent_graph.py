from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from vibe.agent.graph import SYSTEM_PROMPT, build_graph, make_tool_executor, should_continue
from vibe.agent.permissions import (
    PermissionDecision,
    build_permission_request,
    permission_matches,
    workspace_permission_allows,
)
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


class TestPermissionPatterns:
    def test_path_request_normalizes_relative_path_to_absolute_target(self, tmp_path):
        request = build_permission_request("read", {"path": "file.txt"}, str(tmp_path))

        assert request.target == f"read:{tmp_path / 'file.txt'}"
        assert request.workspace_pattern == f"read:{tmp_path}/*"
        assert request.grant_pattern == "read:*"

    def test_wildcard_permissions_match_tool_targets(self, tmp_path):
        assert permission_matches(["read:*"], f"read:{tmp_path / 'file.txt'}")
        assert permission_matches([f"read:{tmp_path}/*"], f"read:{tmp_path / 'file.txt'}")
        assert not permission_matches(["write:*"], f"read:{tmp_path / 'file.txt'}")

    def test_workspace_permission_allows_workspace_root_itself(self, tmp_path):
        request = build_permission_request("read", {"path": str(tmp_path)}, str(tmp_path))

        assert workspace_permission_allows(request)


class TestToolExecutorPermissions:
    def test_workspace_pattern_rejects_path_outside_cwd_before_user_permission(self, tmp_path):
        permission_callback = MagicMock(return_value=True)
        tool = MagicMock(name="read_tool")
        tool.name = "read"
        executor = make_tool_executor([tool], permission_callback)
        outside = tmp_path.parent / "outside.txt"
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": str(outside)}, "id": "1"}])],
            workspace_root=str(tmp_path),
            permission_grants=[],
        )

        result = executor(state)

        assert "outside the working directory" in result["messages"][0].content
        assert result["permission_denied"] is False
        assert result["permission_grants"] == []
        permission_callback.assert_not_called()
        tool.invoke.assert_not_called()

    def test_allow_tool_grant_skips_later_prompt_for_same_tool(self, tmp_path):
        permission_callback = MagicMock(return_value=PermissionDecision(True, "read:*"))
        tool = MagicMock(name="read_tool")
        tool.name = "read"
        tool.invoke.return_value = "ok"
        executor = make_tool_executor([tool], permission_callback)
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": "file.txt"}, "id": "1"}])],
            workspace_root=str(tmp_path),
            permission_grants=["read:*"],
        )

        result = executor(state)

        assert result["messages"][0].content == "ok"
        assert result["permission_grants"] == ["read:*"]
        permission_callback.assert_not_called()
        tool.invoke.assert_called_once_with({"path": "file.txt"})

    def test_user_allow_adds_grant_pattern_to_state(self, tmp_path):
        permission_callback = MagicMock(return_value=PermissionDecision(True, "read:*"))
        tool = MagicMock(name="read_tool")
        tool.name = "read"
        tool.invoke.return_value = "ok"
        executor = make_tool_executor([tool], permission_callback)
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": "file.txt"}, "id": "1"}])],
            workspace_root=str(tmp_path),
            permission_grants=[],
        )

        result = executor(state)

        assert result["permission_grants"] == ["read:*"]
        permission_callback.assert_called_once()

    def test_user_deny_sets_permission_denied(self, tmp_path):
        permission_callback = MagicMock(return_value=False)
        tool = MagicMock(name="read_tool")
        tool.name = "read"
        executor = make_tool_executor([tool], permission_callback)
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": "file.txt"}, "id": "1"}])],
            workspace_root=str(tmp_path),
            permission_grants=[],
        )

        result = executor(state)

        assert result["messages"][0].content == "Permission denied by user."
        assert result["permission_denied"] is True
        tool.invoke.assert_not_called()

    def test_malformed_path_returns_tool_message_error_before_permission(self, tmp_path):
        permission_callback = MagicMock(return_value=True)
        tool = MagicMock(name="read_tool")
        tool.name = "read"
        executor = make_tool_executor([tool], permission_callback)
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": "bad\x00name"}, "id": "1"}])],
            workspace_root=str(tmp_path),
            permission_grants=[],
        )

        result = executor(state)

        assert "embedded null" in result["messages"][0].content
        permission_callback.assert_not_called()
        tool.invoke.assert_not_called()

    def test_unknown_tool_returns_tool_message_error_without_prompt(self):
        permission_callback = MagicMock(return_value=True)
        executor = make_tool_executor([], permission_callback)
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "missing", "args": {}, "id": "1"}])],
            permission_grants=[],
        )

        result = executor(state)

        assert "not a valid tool" in result["messages"][0].content
        permission_callback.assert_not_called()

    def test_tool_exception_returns_tool_message_error(self, tmp_path):
        permission_callback = MagicMock(return_value=PermissionDecision(True, "read:*"))
        tool = MagicMock(name="read_tool")
        tool.name = "read"
        tool.invoke.side_effect = PermissionError("denied")
        executor = make_tool_executor([tool], permission_callback)
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": "file.txt"}, "id": "1"}])],
            workspace_root=str(tmp_path),
            permission_grants=[],
        )

        result = executor(state)

        assert result["messages"][0].content == "Error: denied"
        assert result["permission_denied"] is False

    def test_non_path_tool_uses_tool_wildcard_target(self):
        permission_callback = MagicMock(return_value=PermissionDecision(True, "bash:*"))
        tool = MagicMock(name="bash_tool")
        tool.name = "bash"
        tool.invoke.return_value = "ok"
        executor = make_tool_executor([tool], permission_callback)
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "pwd"}, "id": "1"}])],
            permission_grants=[],
        )

        result = executor(state)

        request = permission_callback.call_args.args[0]
        assert request.target == "bash:*"
        assert request.grant_pattern == "bash:*"
        assert result["permission_grants"] == ["bash:*"]
