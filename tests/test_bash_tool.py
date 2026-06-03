from vibe.tools import bash


class TestBashTool:
    def test_simple_command(self):
        result = bash.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_command_with_stderr(self):
        result = bash.invoke({"command": "echo error >&2"})
        assert "error" in result

    def test_failed_command(self):
        result = bash.invoke({"command": "exit 1"})
        assert "Exit code: 1" in result

    def test_timeout(self):
        result = bash.invoke({"command": "sleep 60", "timeout": 1})
        assert "timed out" in result

    def test_no_output(self):
        result = bash.invoke({"command": "true"})
        assert "(no output)" in result

    def test_rejects_non_positive_timeout(self):
        result = bash.invoke({"command": "true", "timeout": 0})
        assert "timeout must be >= 1" in result
