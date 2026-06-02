from vibe.tools import ALL_TOOLS, RUNTIME_TOOLS
from vibe.tools.hooks import (
    MAX_OUTPUT_CHARS,
    DynamicHooks,
    IntRange,
    MaxFileBytes,
    MaxTextBytes,
    TruncateResult,
    WorkspacePath,
    WorkspacePathValue,
    tool,
    truncate_output,
)
from vibe.tools.runtime import Hook, ToolContext


class TestRuntimeTool:
    def test_invoke_applies_prepare_before_body_after(self):
        events = []

        class RecordingHook(Hook):
            def prepare(self, ctx: ToolContext) -> None:
                events.append("prepare")
                ctx.args["value"] += 1

            def after(self, ctx: ToolContext) -> None:
                events.append("after")
                ctx.result += " after"

        @tool(RecordingHook())
        def sample(value: int) -> str:
            events.append("body")
            return f"value={value}"

        assert sample.invoke({"value": 1}) == "value=2 after"
        assert events == ["prepare", "body", "after"]

    def test_dynamic_hooks_run_in_same_invocation(self):
        class UppercaseHook(Hook):
            def after(self, ctx: ToolContext) -> None:
                ctx.result = ctx.result.upper()

        @tool(DynamicHooks(lambda ctx: [UppercaseHook()]))
        def sample() -> str:
            return "hello"

        assert sample.invoke({}) == "HELLO"

    def test_value_error_returns_error_string(self):
        @tool()
        def sample() -> str:
            raise ValueError("bad")

        assert sample.invoke({}) == "Error: bad"


class TestWorkspacePath:
    def test_valid_path_is_converted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("hello")

        @tool(WorkspacePath("path", require_file=True))
        def sample(path: WorkspacePathValue) -> str:
            return path.read_text()

        assert sample.invoke({"path": "file.txt"}) == "hello"

    def test_rejects_absolute_outside(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        @tool(WorkspacePath("path"))
        def sample(path: WorkspacePathValue) -> str:
            return path.display

        assert "outside" in sample.invoke({"path": "/etc/passwd"})

    def test_rejects_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        @tool(WorkspacePath("path"))
        def sample(path: WorkspacePathValue) -> str:
            return path.display

        assert "outside" in sample.invoke({"path": "../../etc/passwd"})

    def test_rejects_non_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "dir").mkdir()

        @tool(WorkspacePath("path", require_file=True))
        def sample(path: WorkspacePathValue) -> str:
            return path.display

        assert "not a file" in sample.invoke({"path": "dir"})


class TestRuntimeLimits:
    def test_large_file_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 1_000_001)

        @tool(WorkspacePath("path", require_file=True), MaxFileBytes("path"))
        def sample(path: WorkspacePathValue) -> str:
            return path.display

        assert "exceeds" in sample.invoke({"path": "big.txt"})

    def test_large_text_rejected(self):
        @tool(MaxTextBytes("content"))
        def sample(content: str) -> str:
            return content

        assert "exceeds" in sample.invoke({"content": "x" * 1_000_001})

    def test_int_range_rejected(self):
        @tool(IntRange("limit", min=1))
        def sample(limit: int) -> str:
            return str(limit)

        assert "limit must be >= 1" in sample.invoke({"limit": 0})


class TestTruncateOutput:
    def test_short_text_unchanged(self):
        assert truncate_output("hello") == "hello"

    def test_long_text_truncated(self):
        text = "x" * (MAX_OUTPUT_CHARS + 100)
        result = truncate_output(text)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_hook_truncates_result(self):
        @tool(TruncateResult(max_chars=5))
        def sample() -> str:
            return "x" * 10

        assert "truncated" in sample.invoke({})


class TestRuntimeToolExports:
    def test_all_tools_are_langchain_tools(self):
        assert [tool.name for tool in ALL_TOOLS] == ["bash", "read", "write", "edit"]
        assert [tool.name for tool in RUNTIME_TOOLS] == ["bash", "read", "write", "edit"]

    def test_runtime_tool_keeps_string_path_schema(self):
        read_tool = next(tool for tool in ALL_TOOLS if tool.name == "read")
        schema = read_tool.args_schema.model_json_schema()
        assert schema["properties"]["path"]["type"] == "string"
