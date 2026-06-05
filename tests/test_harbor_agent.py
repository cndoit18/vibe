import asyncio
import importlib
import json
import shlex
from unittest.mock import patch

from vibe.config import Settings
from vibe.harbor_agent import LOG_PATH, REMOTE_REPO_DIR, SETTINGS_PATH, VENV_DIR, VibeInstalledAgent


class RecordingAgent(VibeInstalledAgent):
    def __init__(self, *args, extra_env=None, **kwargs):
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self.root_calls = []
        self.agent_calls = []

    async def exec_as_root(self, environment, command, **kwargs):
        self.root_calls.append({"environment": environment, "command": command, "kwargs": kwargs})

    async def exec_as_agent(self, environment, command, **kwargs):
        self.agent_calls.append({"environment": environment, "command": command, "kwargs": kwargs})


class FakeEnvironment:
    def __init__(self):
        self.uploads = []
        self.uploaded_files = []

    async def upload_dir(self, source_dir, target_dir):
        self.uploads.append((source_dir, target_dir))

    async def upload_file(self, source_path, target_path):
        self.uploaded_files.append((source_path, target_path, source_path.read_text()))


def test_import_path_contract_loads_vibe_agent():
    for import_path in ("vibe.harbor_agent", "src.vibe.harbor_agent"):
        module = importlib.import_module(import_path)

        agent_cls = getattr(module, "VibeInstalledAgent")

        assert agent_cls.name() == "vibe"


def test_install_uploads_source_and_installs_cli(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    environment = FakeEnvironment()
    agent = RecordingAgent(logs_dir=tmp_path / "logs", source_dir=source_dir)

    asyncio.run(agent.install(environment))

    assert environment.uploads == [(source_dir.resolve(), REMOTE_REPO_DIR)]
    assert len(agent.root_calls) == 2
    assert f"rm -rf {shlex.quote(REMOTE_REPO_DIR)} {shlex.quote(VENV_DIR)}" in agent.root_calls[0]["command"]
    assert "curl -LsSf https://astral.sh/uv/install.sh" in agent.root_calls[0]["command"]
    install_command = agent.root_calls[1]["command"]
    assert f"uv venv {shlex.quote(VENV_DIR)} --python 3.12 --seed" in install_command
    assert f"{VENV_DIR}/bin/python -m pip install -e {REMOTE_REPO_DIR}" in install_command
    assert f"{VENV_DIR}/bin/vibe --help" in install_command


def test_install_writes_vibe_settings_for_agent_user(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    environment = FakeEnvironment()
    agent = RecordingAgent(
        logs_dir=tmp_path / "logs",
        source_dir=source_dir,
        model_name="openai/gpt-4o-mini",
    )

    with patch(
        "vibe.harbor_agent.load_settings",
        return_value=Settings(base_url="https://api.example.com/v1", api_key="sk-test-key"),
    ):
        asyncio.run(agent.install(environment))

    assert len(environment.uploaded_files) == 1
    _, target_path, content = environment.uploaded_files[0]
    assert target_path == SETTINGS_PATH
    assert json.loads(content) == {
        "model": "gpt-4o-mini",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test-key",
    }
    settings_command = agent.agent_calls[0]["command"]
    assert 'mkdir -p "$HOME/.vibe"' in settings_command
    assert 'cp /tmp/vibe-settings.json "$HOME/.vibe/settings.json"' in settings_command
    assert "sk-test-key" not in settings_command


def test_install_reads_settings_only_for_api_configuration(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    environment = FakeEnvironment()
    agent = RecordingAgent(
        logs_dir=tmp_path / "logs",
        source_dir=source_dir,
        extra_env={"OPENAI_API_KEY": "agent-env-key", "OPENAI_BASE_URL": "https://agent-env.example.com/v1"},
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")

    with patch(
        "vibe.harbor_agent.load_settings",
        return_value=Settings(base_url="https://api.z.ai/api/coding/paas/v4", api_key="settings-key"),
    ):
        asyncio.run(agent.install(environment))

    _, _, content = environment.uploaded_files[0]
    assert json.loads(content) == {
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_key": "settings-key",
    }


def test_run_executes_vibe_print_without_overriding_cwd(tmp_path):
    environment = FakeEnvironment()
    agent = RecordingAgent(logs_dir=tmp_path / "logs", source_dir=tmp_path)
    instruction = "fix 'quoted' bug"

    asyncio.run(agent.run(instruction, environment, object()))

    assert len(agent.agent_calls) == 1
    call = agent.agent_calls[0]
    command = call["command"]
    assert f"{VENV_DIR}/bin/vibe --print -- {shlex.quote(instruction)}" in command
    assert f"tee {LOG_PATH}" in command
    assert "cwd" not in call["kwargs"]
    assert "env" not in call["kwargs"]
