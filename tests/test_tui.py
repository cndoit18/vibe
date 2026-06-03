from io import StringIO
from unittest.mock import Mock, patch

from rich.console import Console

from vibe.agent.conversation import AgentEvent
from vibe.tui import VibeTUI


def make_tui() -> VibeTUI:
    tui = VibeTUI.__new__(VibeTUI)
    tui.console = Console(file=StringIO(), force_terminal=False, height=24)
    tui.conversation = Mock(session_id="abc12345")
    tui.conversation.send.return_value = []
    return tui


def test_tui_renders_raw_brackets_without_markup_errors():
    tui = make_tui()

    tui._submit("[/]")
    tui._render_event(AgentEvent("tool_call", "{'command': '[/]'}", "bash"))
    tui._render_event(AgentEvent("tool_result", "[/]", "bash"))


def test_user_prompt_is_rendered_as_panel_without_user_title():
    tui = make_tui()

    tui.console.print(tui._user_message("hello"))
    output = tui.console.file.getvalue()

    assert "hello" in output
    assert "翰" not in output


def test_tool_call_is_rendered_as_dim_line():
    tui = make_tui()

    first = tui._tool_call(AgentEvent("tool_call", "{'command': 'ls'}", "bash"))
    second = tui._tool_call(AgentEvent("tool_call", "{'path': 'README.md'}", "read"))

    assert first.plain == "* bash {'command': 'ls'}"
    assert second.plain == "* read {'path': 'README.md'}"


def test_run_uses_bottom_input_prompt_for_piped_input():
    tui = make_tui()
    tui.console.input = Mock(side_effect=["/exit"])

    tui.run()

    prompt = tui.console.input.call_args.args[0]
    assert prompt.plain == "› "
    assert "vibe" in tui.console.file.getvalue()


def test_draw_input_block_does_not_pad_to_terminal_height():
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=40, height=24)

    tui._draw_input_block()

    output = tui.console.file.getvalue()
    assert output.count("\n") == 3
    assert "? for shortcuts" in output


def test_terminal_prompt_handles_cjk_backspace_without_input_decoder():
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=40, height=24)

    fake_stdin = FakeStdin("你好".encode() + b"\x7f\r")
    with (
        patch("sys.stdin", fake_stdin),
        patch("vibe.tui.termios.tcgetattr", return_value="settings"),
        patch("vibe.tui.termios.tcsetattr"),
        patch("vibe.tui.tty.setraw"),
    ):
        prompt = tui._read_terminal_prompt()

    assert prompt == "你"


class FakeStdin:
    def __init__(self, payload: bytes):
        self.buffer = BytesReader(payload)

    def fileno(self):
        return 0


class BytesReader:
    def __init__(self, payload: bytes):
        self.payload = bytearray(payload)

    def read(self, size: int):
        if not self.payload:
            return b""
        chunk = self.payload[:size]
        del self.payload[:size]
        return bytes(chunk)


def test_user_message_uses_terminal_cell_width_for_cjk():
    tui = make_tui()
    tui.console.width = 12

    message = tui._user_message("你好")

    assert message.cell_len == 11


def test_input_help_contains_shortcuts_and_exit_hint():
    tui = make_tui()

    assert "? for shortcuts" in tui._input_help().plain
    assert "/exit to quit" in tui._input_help().plain


def test_initial_prompt_is_not_read_from_input_before_submit():
    tui = make_tui()
    tui.console.input = Mock(side_effect=["/exit"])

    with patch.object(tui, "_submit") as submit:
        tui.run("hello")

    submit.assert_called_once_with("hello")
