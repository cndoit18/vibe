from vibe.tools import edit


class TestEditTool:
    def test_edit_replace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("hello world")
        result = edit.invoke({"path": "file.txt", "old_string": "world", "new_string": "vibe"})
        assert "Edited" in result
        assert f.read_text() == "hello vibe"

    def test_edit_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = edit.invoke({"path": "file.txt", "old_string": "missing", "new_string": "x"})
        assert "not found" in result

    def test_edit_multiple_matches(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("aaa aaa aaa")
        result = edit.invoke({"path": "file.txt", "old_string": "aaa", "new_string": "bbb"})
        assert "3 times" in result

    def test_edit_nonexistent_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = edit.invoke({"path": "nope.txt", "old_string": "x", "new_string": "y"})
        assert "not a file" in result

    def test_edit_outside_working_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = edit.invoke({"path": "/etc/hosts", "old_string": "x", "new_string": "y"})
        assert "outside" in result.lower()

    def test_edit_rejects_empty_old_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = edit.invoke({"path": "file.txt", "old_string": "", "new_string": "x"})
        assert "old_string must not be empty" in result

    def test_edit_rejects_oversized_updated_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = edit.invoke({"path": "file.txt", "old_string": "x", "new_string": "y" * 1_000_001})
        assert "exceeds" in result.lower()
