from unittest.mock import Mock, patch

import pytest

from vibe import cli


class TestMainDispatch:
    def test_no_prompt_opens_tui(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe"]),
            patch("vibe.cli.VibeTUI") as tui_cls,
        ):
            cli.main()

        tui_cls.assert_called_once_with(None, "gpt-4o", None)
        tui_cls.return_value.run.assert_called_once_with(None)

    def test_prompt_opens_tui_with_initial_prompt(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello"]),
            patch("vibe.cli.VibeTUI") as tui_cls,
        ):
            cli.main()

        tui_cls.assert_called_once_with(None, "gpt-4o", None)
        tui_cls.return_value.run.assert_called_once_with("hello")

    def test_print_requires_prompt(self):
        with patch("sys.argv", ["vibe", "--print"]), pytest.raises(SystemExit):
            cli.main()

    def test_print_runs_plain_output(self):
        conversation = Mock()
        conversation.session_id = "abc12345"
        conversation.send.return_value = []
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.argv", ["vibe", "hello", "--print"]),
            patch("vibe.cli.AgentConversation", return_value=conversation) as conversation_cls,
        ):
            cli.main()

        conversation_cls.assert_called_once_with(session_id=None, model="gpt-4o", base_url=None)
        conversation.send.assert_called_once_with("hello")
