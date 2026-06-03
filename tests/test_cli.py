from unittest.mock import Mock, patch

import pytest

from vibe import cli
from vibe.config import Settings


class TestMainDispatch:
    def test_no_prompt_opens_tui(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe"]),
            patch("vibe.cli.VibeTUI") as tui_cls,
            patch("vibe.cli.load_settings", return_value=Settings()),
        ):
            cli.main()

        tui_cls.assert_called_once_with(None, "gpt-4o", None, None)
        tui_cls.return_value.run.assert_called_once_with(None)

    def test_prompt_opens_tui_with_initial_prompt(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello"]),
            patch("vibe.cli.VibeTUI") as tui_cls,
            patch("vibe.cli.load_settings", return_value=Settings()),
        ):
            cli.main()

        tui_cls.assert_called_once_with(None, "gpt-4o", None, None)
        tui_cls.return_value.run.assert_called_once_with("hello")

    def test_print_requires_prompt(self):
        with (
            patch("sys.argv", ["vibe", "--print"]),
            patch("vibe.cli.load_settings", return_value=Settings()),
            pytest.raises(SystemExit),
        ):
            cli.main()

    def test_print_runs_plain_output(self):
        conversation = Mock()
        conversation.session_id = "abc12345"
        conversation.send.return_value = []
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello", "--print"]),
            patch("vibe.cli.AgentConversation", return_value=conversation) as conversation_cls,
            patch("vibe.cli.load_settings", return_value=Settings()),
        ):
            cli.main()

        conversation_cls.assert_called_once_with(session_id=None, model="gpt-4o", base_url=None, api_key=None)
        conversation.send.assert_called_once_with("hello")


class TestSettingsIntegration:
    def test_model_from_settings(self):
        conversation = Mock()
        conversation.session_id = "abc12345"
        conversation.send.return_value = []
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello", "--print"]),
            patch("vibe.cli.AgentConversation", return_value=conversation) as conversation_cls,
            patch("vibe.cli.load_settings", return_value=Settings(model="gpt-4o-mini")),
        ):
            cli.main()

        conversation_cls.assert_called_once_with(session_id=None, model="gpt-4o-mini", base_url=None, api_key=None)

    def test_model_cli_overrides_settings(self):
        conversation = Mock()
        conversation.session_id = "abc12345"
        conversation.send.return_value = []
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello", "--print", "-m", "gpt-4-turbo"]),
            patch("vibe.cli.AgentConversation", return_value=conversation) as conversation_cls,
            patch("vibe.cli.load_settings", return_value=Settings(model="gpt-4o-mini")),
        ):
            cli.main()

        conversation_cls.assert_called_once_with(session_id=None, model="gpt-4-turbo", base_url=None, api_key=None)

    def test_base_url_from_settings(self):
        conversation = Mock()
        conversation.session_id = "abc12345"
        conversation.send.return_value = []
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello", "--print"]),
            patch("vibe.cli.AgentConversation", return_value=conversation) as conversation_cls,
            patch("vibe.cli.load_settings", return_value=Settings(base_url="https://settings.example.com")),
        ):
            cli.main()

        conversation_cls.assert_called_once_with(
            session_id=None, model="gpt-4o", base_url="https://settings.example.com", api_key=None
        )

    def test_base_url_cli_overrides_settings(self):
        conversation = Mock()
        conversation.session_id = "abc12345"
        conversation.send.return_value = []
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello", "--print", "--base-url", "https://cli.example.com"]),
            patch("vibe.cli.AgentConversation", return_value=conversation) as conversation_cls,
            patch("vibe.cli.load_settings", return_value=Settings(base_url="https://settings.example.com")),
        ):
            cli.main()

        conversation_cls.assert_called_once_with(
            session_id=None, model="gpt-4o", base_url="https://cli.example.com", api_key=None
        )

    def test_api_key_from_settings(self):
        conversation = Mock()
        conversation.session_id = "abc12345"
        conversation.send.return_value = []
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello", "--print"]),
            patch("vibe.cli.AgentConversation", return_value=conversation) as conversation_cls,
            patch("vibe.cli.load_settings", return_value=Settings(api_key="sk-settings-key")),
        ):
            cli.main()

        conversation_cls.assert_called_once_with(
            session_id=None, model="gpt-4o", base_url=None, api_key="sk-settings-key"
        )
