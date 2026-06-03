from vibe.tools import write


class TestWriteTool:
    def test_write_new_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write.invoke({"path": "new.txt", "content": "hello"})
        assert "Wrote" in result
        assert (tmp_path / "new.txt").read_text() == "hello"

    def test_write_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "existing.txt"
        f.write_text("old")
        write.invoke({"path": "existing.txt", "content": "new"})
        assert f.read_text() == "new"

    def test_write_creates_parent_dirs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write.invoke({"path": "sub/dir/file.txt", "content": "deep"})
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "deep"

    def test_write_outside_working_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write.invoke({"path": "/tmp/evil.txt", "content": "nope"})
        assert "outside" in result.lower()

    def test_write_rejects_file_as_parent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "file.txt").write_text("parent")
        result = write.invoke({"path": "file.txt/child.txt", "content": "nope"})
        assert "parent path" in result

    def test_write_oversized_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        big = "x" * (1_000_001)
        result = write.invoke({"path": "big.txt", "content": big})
        assert "exceeds" in result.lower()
