from vibe.tools import read


class TestReadTool:
    def test_read_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3")
        result = read.invoke({"path": "hello.txt"})
        assert "line1" in result
        assert "line3" in result

    def test_read_with_offset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3")
        result = read.invoke({"path": "hello.txt", "offset": 1})
        assert "line1" not in result
        assert "line2" in result

    def test_read_with_limit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3")
        result = read.invoke({"path": "hello.txt", "limit": 1})
        assert "line1" in result
        assert "line2" not in result

    def test_read_nonexistent_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = read.invoke({"path": "nope.txt"})
        assert "not a file" in result

    def test_read_outside_working_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = read.invoke({"path": "/etc/passwd"})
        assert "outside" in result.lower()

    def test_read_empty_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "hello.txt"
        f.write_text("line1")
        result = read.invoke({"path": "hello.txt", "offset": 100})
        assert "no lines" in result

    def test_read_rejects_negative_offset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "hello.txt"
        f.write_text("line1")
        result = read.invoke({"path": "hello.txt", "offset": -1})
        assert "offset must be >= 0" in result

    def test_read_rejects_non_positive_limit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "hello.txt"
        f.write_text("line1")
        result = read.invoke({"path": "hello.txt", "limit": 0})
        assert "limit must be >= 1" in result
