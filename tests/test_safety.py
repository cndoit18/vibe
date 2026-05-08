from vibe.tools._safety import validate_path, truncate_output, check_file_size, MAX_OUTPUT_CHARS

import pytest


class TestValidatePath:
    def test_valid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = validate_path("some/file.txt")
        assert result == (tmp_path / "some" / "file.txt").resolve()

    def test_rejects_absolute_outside(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside"):
            validate_path("/etc/passwd")

    def test_rejects_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside"):
            validate_path("../../etc/passwd")


class TestTruncateOutput:
    def test_short_text_unchanged(self):
        assert truncate_output("hello") == "hello"

    def test_long_text_truncated(self):
        text = "x" * (MAX_OUTPUT_CHARS + 100)
        result = truncate_output(text)
        assert len(result) < len(text)
        assert "truncated" in result


class TestCheckFileSize:
    def test_small_file_ok(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hi")
        check_file_size(f)

    def test_large_file_rejected(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (1_000_001))
        with pytest.raises(ValueError, match="exceeds"):
            check_file_size(f)
