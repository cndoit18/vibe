import asyncio
import importlib
import shlex

from vibe.harbor_agent import LOG_PATH, REMOTE_REPO_DIR, VENV_DIR, VibeInstalledAgent


class RecordingAgent(VibeInstalledAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_calls = []
        self.agent_calls = []

    async def exec_as_root(self, environment, command, **kwargs):
        self.root_calls.append({"environment": environment, "command": command, "kwargs": kwargs})

    async def exec_as_agent(self, environment, command, **kwargs):
        self.agent_calls.append({"environment": environment, "command": command, "kwargs": kwargs})


class FakeEnvironment:
    def __init__(self):
        self.uploads = []

    async def upload_dir(self, source_dir, target_dir):
        self.uploads.append((source_dir, target_dir))


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
    install_command = agent.root_calls[1]["command"]
    assert f"python3 -m venv {shlex.quote(VENV_DIR)}" in install_command
    assert f"{VENV_DIR}/bin/python -m pip install -e {REMOTE_REPO_DIR}" in install_command
    assert f"{VENV_DIR}/bin/vibe --help" in install_command


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
