from collections import deque
from io import StringIO
from queue import Queue
from threading import Thread
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.syntax import Syntax

from vibe.agent.conversation import AgentConversation, AgentEvent
from vibe.agent.permissions import ToolPermissionRequest
from vibe.tui import KeyPress, PermissionRequest, VibeTUI, read_key


def make_tui() -> VibeTUI:
    tui = VibeTUI.__new__(VibeTUI)
    tui.console = Console(file=StringIO(), force_terminal=False, height=24)
    tui.conversation = Mock(session_id="abc12345")
    tui.conversation.send.return_value = []
    tui._ensure_state()
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


def test_edit_tool_result_is_rendered_as_plain_tool_result():
    tui = make_tui()

    tui._render_event(
        AgentEvent(
            "tool_call",
            "{'path': 'file.txt', 'old_string': 'hello world', 'new_string': 'hello vibe'}",
            "edit",
            "tc1",
        )
    )
    tui._render_event(AgentEvent("tool_result", "Edited 'file.txt': replaced 5 chars with 4 chars", "edit", "tc1"))

    assert not isinstance(tui._history[-1], Syntax)
    assert "Edited 'file.txt'" in tui.console.file.getvalue()


def test_conversation_events_include_assistant_text_before_tool_call():
    conversation = AgentConversation.__new__(AgentConversation)
    message = AIMessage(
        content="I'll run ls first.",
        tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "tc1"}],
    )

    events = list(conversation._events_from_message(message))

    assert [(event.kind, event.content, event.name) for event in events] == [
        ("assistant", "I'll run ls first.", None),
        ("tool_call", "{'command': 'ls'}", "bash"),
    ]


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


def test_permission_key_scrolls_preview_without_changing_selection():
    tui = make_tui()
    tui._permission = PermissionRequest("read", "{'path': 'file.txt'}")

    tui._handle_permission_key(KeyPress("scroll_down"))
    tui._handle_permission_key(KeyPress("scroll_down"))
    tui._handle_permission_key(KeyPress("scroll_up"))

    assert tui._permission.scroll_offset == 3
    assert tui._permission.selected == 0


def test_read_key_maps_xterm_mouse_wheel():
    up_stdin = FakeStdin(b"\x1b[<64;1;1M")
    down_stdin = FakeStdin(b"\x1b[<65;1;1M")
    release_stdin = FakeStdin(b"\x1b[<0;1;1m")

    with patch("sys.stdin", up_stdin):
        assert read_key().name == "scroll_up"
    with patch("sys.stdin", down_stdin):
        assert read_key().name == "scroll_down"
    with patch("sys.stdin", release_stdin):
        assert read_key().name == "mouse"


def test_mouse_release_does_not_reject_permission():
    tui = make_tui()
    tui._permission = PermissionRequest("read", "{'path': 'file.txt'}")

    tui._handle_permission_key(KeyPress("mouse"))

    assert tui._permission is not None


def test_finish_permission_choice_disables_permission_mouse():
    tui = make_tui()
    tui._permission = PermissionRequest("read", "{'path': 'file.txt'}")
    tui._permission_mouse_enabled = True
    tui._write_terminal_control = Mock()
    reply_queue = Queue()
    tui._permission_reply_queue = reply_queue

    tui._finish_permission_choice("y")

    tui._write_terminal_control.assert_called_once_with("\x1b[?1006l\x1b[?1000l")
    assert reply_queue.get_nowait() == "y"


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


def test_permission_prompt_allows_current_tool_pattern():
    tui = make_tui()
    tui._read_permission_choice = Mock(return_value="a")
    request = ToolPermissionRequest(
        name="read", args="{'path': 'file.txt'}", target="read:/repo/file.txt", grant_pattern="read:*"
    )

    decision = tui._permission_prompt(request)

    assert decision.allowed is True
    assert decision.grant_pattern == "read:*"


def test_permission_prompt_single_allow_does_not_add_grant_pattern():
    tui = make_tui()
    tui._read_permission_choice = Mock(return_value="y")
    request = ToolPermissionRequest(
        name="read", args="{'path': 'file.txt'}", target="read:/repo/file.txt", grant_pattern="read:*"
    )

    decision = tui._permission_prompt(request)

    assert decision.allowed is True
    assert decision.grant_pattern is None


def test_permission_panel_allows_tool_not_all_tools():
    tui = make_tui()
    tui.console.print(tui._permission_panel(PermissionRequest("read", "{'path': 'file.txt'}")))
    output = tui.console.file.getvalue()

    assert "allow read during this session" in output
    assert "allow all tools" not in output


def test_edit_permission_preview_shows_file_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file.txt").write_text("before\nhello world\nafter\n", encoding="utf-8")
    tui = make_tui()
    tui.console.print(
        tui._permission_panel(
            PermissionRequest(
                "edit",
                "{'path': 'file.txt', 'old_string': 'hello world', 'new_string': 'hello vibe'}",
            )
        )
    )
    output = tui.console.file.getvalue()

    assert "Update(file.txt)" in output
    assert "Modified 1 lines" in output
    assert "before" in output
    assert "after" in output
    assert "-   2  hello world" in output
    assert "+   2  hello vibe" in output


def test_edit_permission_preview_splits_modified_and_added_lines():
    tui = make_tui()
    diff = tui._content_diff("README.md", "## 使用\n", "## 🚀 使用\nextra\n")

    assert "Modified 1 lines, Added 1 lines" in diff
    assert "Modified 2 lines" not in diff


def test_edit_permission_preview_shows_diff_before_choices():
    tui = make_tui()
    tui.console.print(
        tui._permission_panel(
            PermissionRequest(
                "edit",
                "{'path': 'README.md', 'old_string': '## 配置', 'new_string': '## ⚙ 配置'}",
            )
        )
    )
    output = tui.console.file.getvalue()

    assert output.index("Update(README.md)") < output.index("Do you want to proceed?")
    assert (
        output.count("\n│                                                                              │\n│ • Update")
        == 0
    )
    assert "Modified 1 lines" in output
    assert "Added 1 lines" not in output


def test_edit_permission_preview_replace_all_shows_all_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file.txt").write_text("first\ntarget\nmiddle\ntarget\nlast\n", encoding="utf-8")
    tui = make_tui()
    tui.console.print(
        tui._permission_panel(
            PermissionRequest(
                "edit",
                "{'path': 'file.txt', 'old_string': 'target', 'new_string': 'value', 'replace_all': True}",
            )
        )
    )
    output = tui.console.file.getvalue()

    assert output.count("Modified 2 lines") == 1
    assert "-   2  target" in output
    assert "+   2  value" in output
    assert "-   4  target" in output
    assert "+   4  value" in output


def test_edit_permission_preview_scrolls_to_keep_choices_visible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lines = [f"line {index}" for index in range(30)]
    lines[3] = "target"
    lines[24] = "target"
    (tmp_path / "file.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=80, height=20)
    request = PermissionRequest(
        "edit",
        "{'path': 'file.txt', 'old_string': 'target', 'new_string': 'value', 'replace_all': True}",
    )

    tui.console.print(tui._permission_panel(request))
    first_output = tui.console.file.getvalue()
    tui.console.file = StringIO()
    request.scroll_offset = 23
    tui.console.print(tui._permission_panel(request))
    second_output = tui.console.file.getvalue()

    assert "line 2" in first_output
    assert "line 23" in second_output
    assert "Do you want to proceed?" in second_output
    assert "Enter to confirm" in second_output


def test_edit_permission_preview_clamps_stored_scroll_offset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lines = [f"line {index}" for index in range(30)]
    lines[3] = "target"
    lines[24] = "target"
    (tmp_path / "file.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=80, height=20)
    request = PermissionRequest(
        "edit",
        "{'path': 'file.txt', 'old_string': 'target', 'new_string': 'value', 'replace_all': True}",
        scroll_offset=999,
    )

    tui._permission = request
    tui.console.print(tui._screen())

    assert request.scroll_offset == 25


def test_permission_call_preview_clamps_offset_from_full_diff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lines = [f"line {index}" for index in range(30)]
    lines[3] = "target"
    lines[24] = "target"
    (tmp_path / "file.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=80, height=20)
    tui._permission = PermissionRequest(
        "edit",
        "{'path': 'file.txt', 'old_string': 'target', 'new_string': 'value', 'replace_all': True}",
        scroll_offset=999,
    )

    tui._permission_call_preview(tui._permission.name, tui._permission.args, tui._permission.scroll_offset)

    assert tui._permission.scroll_offset == 25


def test_write_permission_preview_scrolls_to_keep_choices_visible():
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=80, height=20)
    request = PermissionRequest(
        "write",
        {"path": "file.txt", "content": "\n".join(f"line {index}" for index in range(30)) + "\n"}.__repr__(),
    )

    tui.console.print(tui._permission_panel(request))
    first_output = tui.console.file.getvalue()
    tui.console.file = StringIO()
    request.scroll_offset = 23
    tui.console.print(tui._permission_panel(request))
    second_output = tui.console.file.getvalue()

    assert "Create(file.txt)" in first_output
    assert "line 2" in first_output
    assert "line 23" in second_output
    assert "Do you want to proceed?" in second_output
    assert "Enter to confirm" in second_output


def test_load_and_render_history_restores_session_messages():
    tui = make_tui()
    tui.conversation.load_history.return_value = [
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
    ]

    tui._load_and_render_history()
    output = tui.console.file.getvalue()

    assert "hello" in output
    assert "hi there" in output


def test_load_and_render_history_renders_tool_calls_and_results():
    tui = make_tui()
    tui.conversation.load_history.return_value = [
        HumanMessage(content="run ls"),
        AIMessage(
            content="I'll run ls first.",
            tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "tc1"}],
        ),
        ToolMessage(content="file.txt", name="bash", tool_call_id="tc1"),
        AIMessage(content="done"),
    ]

    tui._load_and_render_history()
    output = tui.console.file.getvalue()

    assert "run ls" in output
    assert "I'll run ls first." in output
    assert "bash" in output
    assert "file.txt" in output
    assert "done" in output


def test_enqueue_prompt_keeps_prompt_pending_without_sending():
    tui = make_tui()

    tui._enqueue_prompt("hello")

    assert list(tui._prompt_queue) == ["hello"]
    assert tui._input == ""
    tui.conversation.send.assert_not_called()


def test_start_next_prompt_runs_pending_prompt_in_worker():
    tui = make_tui()
    tui.conversation.send.return_value = [AgentEvent("assistant", "hi")]
    tui._prompt_queue.append("hello")

    tui._start_next_prompt_if_idle()
    assert tui._agent_thread is not None
    tui._agent_thread.join(timeout=2)
    tui._drain_agent_events()

    tui.conversation.send.assert_called_once_with("hello")
    assert not tui._agent_busy
    assert not tui._prompt_queue
    assert "hello" in tui.console.file.getvalue()
    assert "hi" in tui.console.file.getvalue()


def test_agent_done_starts_next_queued_prompt_automatically():
    tui = make_tui()
    tui.conversation.send.side_effect = [[], []]
    tui._prompt_queue.extend(["one", "two"])

    tui._start_next_prompt_if_idle()
    first_thread = tui._agent_thread
    assert first_thread is not None
    first_thread.join(timeout=2)
    tui._drain_agent_events()
    if tui._agent_thread is not None:
        tui._agent_thread.join(timeout=2)
        tui._drain_agent_events()

    assert tui.conversation.send.call_args_list[0].args == ("one",)
    assert tui.conversation.send.call_args_list[1].args == ("two",)
    assert not tui._agent_busy
    assert not tui._prompt_queue


def test_escape_removes_last_queued_prompt_without_canceling_active_prompt():
    tui = make_tui()
    tui._agent_busy = True
    tui._prompt_queue = deque(["one", "two"])

    tui._handle_prompt_key(KeyPress("escape"), Mock())

    assert list(tui._prompt_queue) == ["one"]
    assert tui._agent_busy is True


def test_queue_status_renders_pending_count_prompts_and_escape_hint():
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=80, height=24)
    tui._agent_busy = True
    tui._prompt_queue.extend(["one", "two"])

    tui.console.print(tui._screen())
    output = tui.console.file.getvalue()

    assert "thinking" in output
    assert "Queue · 2 pending" in output
    assert "1." in output
    assert "one" in output
    assert "2." in output
    assert "two" in output
    assert "Esc removes the last queued item" in output


def test_queue_status_compacts_long_or_multiline_prompts():
    tui = make_tui()
    tui.console = Console(file=StringIO(), force_terminal=True, width=100, height=24)
    tui._prompt_queue.extend(["first\nsecond", "x" * 50, "third", "fourth"])

    tui.console.print(tui._screen())
    output = tui.console.file.getvalue()

    assert "Queue · 4 pending" in output
    assert "1." in output
    assert "first second" in output
    assert "2." in output
    assert "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx..." in output
    assert "3." in output
    assert "third" in output
    assert "… 1 more" in output


def test_live_permission_prompt_uses_thread_message_queue():
    tui = make_tui()
    tui._live = object()
    request = ToolPermissionRequest(
        name="read", args="{'path': 'file.txt'}", target="read:/repo/file.txt", grant_pattern="read:*"
    )
    result_queue = Queue()

    thread = Thread(target=lambda: result_queue.put(tui._permission_prompt(request)))
    thread.start()
    message = tui._agent_event_queue.get(timeout=2)
    assert message.kind == "permission_request"
    _, reply_queue = message.payload
    reply_queue.put("a")
    thread.join(timeout=2)

    decision = result_queue.get(timeout=2)
    assert decision.allowed is True
    assert decision.grant_pattern == "read:*"


def test_permission_key_finishes_request_from_main_thread():
    tui = make_tui()
    reply_queue = Queue()
    tui._permission = PermissionRequest("read", "{'path': 'file.txt'}")
    tui._permission_reply_queue = reply_queue

    tui._handle_permission_key(KeyPress("char", "n"))

    assert tui._permission is None
    assert reply_queue.get_nowait() == "n"
