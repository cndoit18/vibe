import json
from unittest.mock import patch

from vibe.config import Settings, load_settings

SETTINGS_PATH_ATTR = "vibe.config.SETTINGS_PATH"


def _write(path, data):
    path.write_text(json.dumps(data))


class TestLoadSettings:
    def test_file_missing(self, tmp_path):
        with patch(SETTINGS_PATH_ATTR, tmp_path / "settings.json"):
            assert load_settings() == Settings()

    def test_valid_json(self, tmp_path):
        p = tmp_path / "settings.json"
        _write(p, {"model": "gpt-4o-mini", "base_url": "https://api.example.com"})
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings(model="gpt-4o-mini", base_url="https://api.example.com")

    def test_partial_json(self, tmp_path):
        p = tmp_path / "settings.json"
        _write(p, {"model": "gpt-4o-mini"})
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings(model="gpt-4o-mini")

    def test_empty_json(self, tmp_path):
        p = tmp_path / "settings.json"
        _write(p, {})
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings()

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "settings.json"
        p.write_text("{bad json")
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings()

    def test_not_a_dict(self, tmp_path):
        p = tmp_path / "settings.json"
        _write(p, [1, 2, 3])
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings()

    def test_null_values(self, tmp_path):
        p = tmp_path / "settings.json"
        _write(p, {"model": None, "base_url": None})
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings()

    def test_extra_keys_ignored(self, tmp_path):
        p = tmp_path / "settings.json"
        _write(p, {"model": "gpt-4o", "temperature": 0.5})
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings(model="gpt-4o")

    def test_empty_string_values(self, tmp_path):
        p = tmp_path / "settings.json"
        _write(p, {"model": "", "base_url": ""})
        with patch(SETTINGS_PATH_ATTR, p):
            assert load_settings() == Settings(model="", base_url="")
