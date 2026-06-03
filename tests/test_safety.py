from vibe.tools import read
from vibe.tools.paths import MAX_FILE_SIZE, workspace_file, workspace_path
from vibe.tools.runtime import MAX_INLINE_OUTPUT_CHARS, Tool, tool


class TestTool:
    def test_value_error_returns_prompt_readable_error(self):
        def sample() -> str:
            """Sample command."""
            raise ValueError("bad")

        assert Tool(sample).invoke({}) == "Error: bad"

    def test_validation_error_returns_prompt_readable_error(self):
        from typing import Annotated

        from pydantic import Field

        def sample(limit: Annotated[int, Field(ge=1)]) -> str:
            """Sample command."""
            return str(limit)

        assert "Error: limit" in Tool(sample).invoke({"limit": 0})

    def test_large_output_is_saved_to_workspace_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def sample() -> str:
            """Sample command."""
            return "x" * (MAX_INLINE_OUTPUT_CHARS + 1)

        result = Tool(sample).invoke({})
        assert "saved to '.vibe/tmp/tool-output-sample-" in result
        output_file = next((tmp_path / ".vibe" / "tmp").glob("tool-output-sample-*.txt"))
        assert output_file.read_text() == "x" * (MAX_INLINE_OUTPUT_CHARS + 1)

    def test_tool_decorator_returns_tool(self):
        @tool
        def sample() -> str:
            """Sample command."""
            return "ok"

        assert isinstance(sample, Tool)
        assert sample.invoke({}) == "ok"


class TestWorkspacePath:
    def test_workspace_file_reads_valid_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("hello")

        assert workspace_file("file.txt").read_text() == "hello"

    def test_workspace_path_rejects_absolute_outside(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        try:
            workspace_path("/etc/passwd")
        except ValueError as error:
            assert "outside" in str(error)
        else:
            raise AssertionError("expected ValueError")

    def test_workspace_path_rejects_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        try:
            workspace_path("../../etc/passwd")
        except ValueError as error:
            assert "outside" in str(error)
        else:
            raise AssertionError("expected ValueError")

    def test_workspace_file_rejects_non_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "dir").mkdir()

        try:
            workspace_file("dir")
        except ValueError as error:
            assert "not a file" in str(error)
        else:
            raise AssertionError("expected ValueError")

    def test_workspace_file_rejects_large_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (MAX_FILE_SIZE + 1))

        try:
            workspace_file("big.txt")
        except ValueError as error:
            assert "exceeds" in str(error)
        else:
            raise AssertionError("expected ValueError")

    def test_workspace_path_write_text_rejects_large_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = workspace_path("big.txt")

        try:
            target.write_text("x" * (MAX_FILE_SIZE + 1))
        except ValueError as error:
            assert "exceeds" in str(error)
        else:
            raise AssertionError("expected ValueError")


class TestRuntimeToolExports:
    def test_runtime_tool_keeps_string_path_schema(self):
        schema = read.args_schema.model_json_schema()
        assert schema["properties"]["path"]["type"] == "string"
        assert schema["properties"]["limit"]["maximum"] == 2000
