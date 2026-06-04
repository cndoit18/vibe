import ast
import codecs
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
import os
from pathlib import Path
import queue
import select
import sys
import termios
import threading
from typing import Any
import tty

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from vibe.agent.conversation import AgentConversation, AgentEvent
from vibe.agent.permissions import PermissionDecision, ToolPermissionRequest

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", ":q"}


@dataclass
class KeyPress:
    name: str
    value: str = ""


@dataclass
class PermissionRequest:
    name: str
    args: str
    selected: int = 0
    scroll_offset: int = 0


@dataclass(frozen=True)
class AgentThreadMessage:
    kind: str
    payload: Any = None


@dataclass(frozen=True)
class TranscriptViewEvent:
    kind: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class PromptLiveViewModel:
    input_text: str
    status: str | None
    agent_busy: bool
    exit_after_active: bool
    queued_prompts: tuple[str, ...]


@dataclass(frozen=True)
class PermissionDialogViewModel:
    name: str
    args: str
    selected: int
    scroll_offset: int
    terminal_height: int


class TerminalView:
    def __init__(self, console: Console):
        self.console = console

    def render_header(self, session_id: str) -> RenderableType:
        return Text.assemble(("▸ vibe", "bold cyan"), (f"  session {session_id} · /exit to quit", "dim"))

    def render_transcript_event(self, event: TranscriptViewEvent) -> RenderableType:
        if event.kind == "user":
            return self._input_line(event.content)
        if event.kind == "assistant":
            return Markdown(event.content.strip("\n"))
        if event.kind == "tool_call":
            return Text.assemble(("* ", "yellow"), (event.name or "tool", "cyan"), (f" {event.content}", "dim"))
        if event.kind == "tool_result":
            lines = event.content.split("\n")
            first = f"  → {lines[0]}"
            rest = [f"    {line}" for line in lines[1:]]
            return Text("\n".join([first, *rest]), style="dim")
        return Text(event.content)

    def render_live_prompt(self, model: PromptLiveViewModel) -> RenderableType:
        renderables = [
            renderable
            for renderable in (
                self._status_line(model.status, model.agent_busy, model.exit_after_active),
                self._queue_panel(model.queued_prompts),
                self._input_panel(model.input_text),
            )
            if renderable
        ]
        return Group(*renderables) if len(renderables) > 1 else renderables[0]

    def render_permission_dialog(self, model: PermissionDialogViewModel) -> RenderableType:
        options = [
            "Yes",
            f"Yes, allow {model.name} during this session",
            "No",
        ]
        preview = self._permission_call_preview(model.name, model.args, model.scroll_offset, model.terminal_height)
        header = Text.assemble((self._permission_title(model.name), "bold cyan"))
        if isinstance(preview, str):
            header.append("\n")
            header.append(preview, style="dim")

        footer = Text("\nDo you want to proceed?\n", style="bright_black")
        for index, option in enumerate(options):
            prefix = "❯" if index == model.selected else " "
            style = "white" if index == model.selected else "bright_black"
            footer.append(f"{prefix} {index + 1}. {option}\n", style=style)
        footer.append("\nEsc to cancel · ↑/↓ to select · Enter to confirm", style="bright_black")
        if not isinstance(preview, str):
            return Panel(Group(header, preview, footer), border_style="cyan", padding=(0, 1))
        return Panel(Group(header, footer), border_style="cyan", padding=(0, 1))

    def render_input_prompt(self) -> Text:
        return Text.assemble(("› ", "bold bright_black"))

    def render_input_block(self) -> list[RenderableType]:
        return [Rule(style="bright_black"), self._input_line(""), Rule(style="bright_black"), self._input_help()]

    def _status_line(self, status: str | None, agent_busy: bool, exit_after_active: bool) -> Text | None:
        parts: list[tuple[str, str]] = []
        if status:
            parts.append((status, "cyan"))
        elif agent_busy:
            parts.append(("thinking", "cyan"))
        if exit_after_active:
            parts.append(("exiting after current turn", "bright_black"))
        if not parts:
            return None
        chunks = []
        for index, (text, style) in enumerate(parts):
            if index:
                chunks.append((" · ", "bright_black"))
            chunks.append((text, style))
        return Text.assemble(*chunks)

    def _queue_panel(self, queued_prompts: tuple[str, ...]) -> Panel | None:
        if not queued_prompts:
            return None
        body = Text()
        for index, prompt in enumerate(queued_prompts[:3], start=1):
            body.append(f"{index}. ", style="bright_black")
            body.append(self._compact_prompt(prompt), style="white")
            body.append("\n")
        if len(queued_prompts) > 3:
            body.append(f"… {len(queued_prompts) - 3} more\n", style="bright_black")
        body.append("Esc removes the last queued item", style="bright_black")
        title = f"Queue · {len(queued_prompts)} pending"
        return Panel(body, title=title, border_style="yellow", padding=(0, 1))

    def _compact_prompt(self, prompt: str) -> str:
        compact = " ".join(prompt.split())
        if len(compact) <= 40:
            return compact
        return f"{compact[:37]}..."

    def _input_panel(self, input_text: str) -> Panel:
        line = Text.assemble(("› ", "bold bright_black"), (input_text, "white"), ("█", "white"))
        return Panel(line, border_style="bright_black", padding=(0, 1))

    def _permission_call_preview(
        self, name: str, args: str, scroll_offset: int = 0, terminal_height: int | None = None
    ) -> str | Syntax:
        parsed_args = self._parse_tool_args(args)
        if isinstance(parsed_args, dict) and parsed_args:
            if name == "edit":
                edit_preview = self._edit_preview(parsed_args, scroll_offset, terminal_height)
                if edit_preview:
                    return edit_preview
            if name == "write":
                write_preview = self._write_preview(parsed_args, scroll_offset, terminal_height)
                if write_preview:
                    return write_preview
            first_arg = next(iter(parsed_args.values()))
            return f"{name}({first_arg})"
        return f"{name}({args})"

    def _edit_preview(self, args: dict, scroll_offset: int = 0, terminal_height: int | None = None) -> Syntax | None:
        path = args.get("path")
        old_string = args.get("old_string")
        new_string = args.get("new_string")
        if not isinstance(path, str) or not isinstance(old_string, str) or not isinstance(new_string, str):
            return None
        replace_all = args.get("replace_all") is True
        return self._scrollable_diff_preview(
            self._edit_diff(path, old_string, new_string, replace_all), scroll_offset, terminal_height
        )

    def _write_preview(self, args: dict, scroll_offset: int = 0, terminal_height: int | None = None) -> Syntax | None:
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return None
        before = self._read_edit_preview_source(path)
        if before is None:
            diff = self._content_diff(path, "", content)
            diff = diff.replace(f"• Update({path})", f"• Create({path})", 1)
        else:
            diff = self._content_diff(path, before, content, replace_all=True)
        return self._scrollable_diff_preview(diff, scroll_offset, terminal_height)

    def _scrollable_diff_preview(self, diff: str, offset: int, terminal_height: int | None = None) -> Syntax:
        max_lines = max((terminal_height or self.console.height) - 11, 3)
        lines = diff.splitlines()
        offset = max(0, min(offset, max(len(lines) - max_lines, 0)))
        visible_lines = lines[offset : offset + max_lines]
        return Syntax("\n".join(visible_lines), "diff", theme="monokai", word_wrap=False)

    def _edit_diff(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        before = self._read_edit_preview_source(path)
        if before is not None and old_string in before:
            preview = self._replacement_diff(path, before, old_string, new_string, replace_all)
            if preview:
                return preview
            after = before.replace(old_string, new_string) if replace_all else before.replace(old_string, new_string, 1)
            return self._content_diff(path, before, after, replace_all)
        return self._content_diff(path, old_string, new_string, replace_all)

    def _read_edit_preview_source(self, path: str) -> str | None:
        target = Path(path).resolve()
        try:
            target.relative_to(Path.cwd())
        except ValueError:
            return None
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _replacement_diff(
        self, path: str, content: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str | None:
        old_lines = content.splitlines()
        old_match_lines = old_string.splitlines()
        new_match_lines = new_string.splitlines()
        match_starts = self._match_line_starts(old_lines, old_match_lines)
        if not old_match_lines or not new_match_lines or not match_starts:
            return None
        if not replace_all:
            match_starts = match_starts[:1]

        diff_lines = [
            f"• Update({path})",
            f"└ {self._edit_summary_for_replacements(len(old_match_lines), len(new_match_lines), len(match_starts))}",
        ]
        match_start_set = set(match_starts)
        index = 0
        while index < len(old_lines):
            if index in match_start_set:
                for line_number, line in enumerate(old_match_lines, start=index + 1):
                    diff_lines.append(f"-{line_number:>4}  {line}")
                for line_number, line in enumerate(new_match_lines, start=index + 1):
                    diff_lines.append(f"+{line_number:>4}  {line}")
                index += len(old_match_lines)
            else:
                diff_lines.append(f" {index + 1:>4}  {old_lines[index]}")
                index += 1
        return "\n".join(diff_lines)

    def _match_line_starts(self, lines: list[str], match_lines: list[str]) -> list[int]:
        if not match_lines:
            return []
        width = len(match_lines)
        return [index for index in range(len(lines) - width + 1) if lines[index : index + width] == match_lines]

    def _edit_summary_for_replacements(self, removed_per_match: int, added_per_match: int, count: int) -> str:
        modified = min(removed_per_match, added_per_match) * count
        added = max(added_per_match - removed_per_match, 0) * count
        removed = max(removed_per_match - added_per_match, 0) * count
        parts = []
        if modified:
            parts.append(f"Modified {modified} lines")
        if added:
            parts.append(f"Added {added} lines")
        if removed:
            parts.append(f"Removed {removed} lines")
        return ", ".join(parts) if parts else "No changes"

    def _content_diff(self, path: str, before: str, after: str, replace_all: bool = False) -> str:
        old_lines = before.splitlines()
        new_lines = after.splitlines()
        matcher = SequenceMatcher(None, old_lines, new_lines)
        changes = [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]
        visible_changes = changes if replace_all else changes[:1]
        summary = self._edit_summary_for_changes(visible_changes)
        diff_lines = [f"• Update({path})", f"└ {summary}"]
        change_by_start = {i1: (tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in visible_changes}
        index = 0
        while index <= len(old_lines):
            change = change_by_start.get(index)
            if change:
                _, i1, i2, j1, j2 = change
                for line_number, line in enumerate(old_lines[i1:i2], start=i1 + 1):
                    diff_lines.append(f"-{line_number:>4}  {line}")
                for line_number, line in enumerate(new_lines[j1:j2], start=j1 + 1):
                    diff_lines.append(f"+{line_number:>4}  {line}")
                if i2 == i1:
                    change_by_start.pop(index)
                    if index == len(old_lines):
                        break
                else:
                    index = i2
                    continue
            if index == len(old_lines):
                break
            diff_lines.append(f" {index + 1:>4}  {old_lines[index]}")
            index += 1
        return "\n".join(diff_lines)

    def _edit_summary_for_changes(self, changes) -> str:
        modified = 0
        added = 0
        removed = 0
        for tag, i1, i2, j1, j2 in changes:
            if tag == "replace":
                changed_old = i2 - i1
                changed_new = j2 - j1
                modified += min(changed_old, changed_new)
                added += max(changed_new - changed_old, 0)
                removed += max(changed_old - changed_new, 0)
            elif tag == "insert":
                added += j2 - j1
            elif tag == "delete":
                removed += i2 - i1
        parts = []
        if modified:
            parts.append(f"Modified {modified} lines")
        if added:
            parts.append(f"Added {added} lines")
        if removed:
            parts.append(f"Removed {removed} lines")
        return ", ".join(parts) if parts else "No changes"

    def _parse_tool_args(self, args: str):
        try:
            return ast.literal_eval(args)
        except (SyntaxError, ValueError):
            return None

    def _permission_title(self, name: str) -> str:
        titles = {"read": "Read file", "write": "Write file", "edit": "Edit file", "bash": "Run command"}
        return titles.get(name, name.replace("_", " ").title())

    def _input_line(self, prompt: str) -> Text:
        line = Text.assemble(("› ", "bold bright_black"), (prompt, "white"))
        line.truncate(max(1, self.console.width - 1), overflow="crop", pad=True)
        line.stylize("on grey19")
        return line

    def _input_help(self) -> Text:
        left = Text("  ? for shortcuts", style="dim")
        right = Text("/exit to quit", style="dim")
        padding = max(1, self.console.width - 1 - left.cell_len - right.cell_len)
        return Text.assemble(left, (" " * padding, "dim"), right)


class VibeTUI:
    def __init__(
        self,
        session_id: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.console = Console()
        self.view = TerminalView(self.console)
        self._history: list[RenderableType] = []
        self._input = ""
        self._status: Text | None = None
        self._permission: PermissionRequest | None = None
        self._live: Live | None = None
        self._pending_tool_calls: list[AgentEvent] = []
        self._prompt_queue: deque[str] = deque()
        self._agent_event_queue: queue.Queue[AgentThreadMessage] = queue.Queue()
        self._agent_thread: threading.Thread | None = None
        self._agent_busy = False
        self._active_prompt: str | None = None
        self._active_first_event = False
        self._permission_reply_queue: queue.Queue[str] | None = None
        self._permission_mouse_enabled = False
        self._exit_after_active = False
        self.conversation = AgentConversation(
            session_id=session_id,
            model=model,
            base_url=base_url,
            api_key=api_key,
            permission_callback=self._permission_prompt,
        )

    def _ensure_state(self):
        if not hasattr(self, "view"):
            self.view = TerminalView(self.console)
        if not hasattr(self, "_history"):
            self._history = []
        if not hasattr(self, "_input"):
            self._input = ""
        if not hasattr(self, "_status"):
            self._status = None
        if not hasattr(self, "_permission"):
            self._permission = None
        if not hasattr(self, "_live"):
            self._live = None
        if not hasattr(self, "_pending_tool_calls"):
            self._pending_tool_calls = []
        if not hasattr(self, "_prompt_queue"):
            self._prompt_queue = deque()
        if not hasattr(self, "_agent_event_queue"):
            self._agent_event_queue = queue.Queue()
        if not hasattr(self, "_agent_thread"):
            self._agent_thread = None
        if not hasattr(self, "_agent_busy"):
            self._agent_busy = False
        if not hasattr(self, "_active_prompt"):
            self._active_prompt = None
        if not hasattr(self, "_active_first_event"):
            self._active_first_event = False
        if not hasattr(self, "_permission_reply_queue"):
            self._permission_reply_queue = None
        if not hasattr(self, "_permission_mouse_enabled"):
            self._permission_mouse_enabled = False
        if not hasattr(self, "_exit_after_active"):
            self._exit_after_active = False

    def run(self, initial_prompt: str | None = None):
        self._ensure_state()
        if not self.console.is_terminal or not sys.stdin.isatty():
            self._run_plain(initial_prompt)
            return

        self._append(self._header())
        self._load_and_render_history()
        with (
            cbreak_terminal(),
            Live(self._screen(), console=self.console, screen=False, auto_refresh=False, transient=False) as live,
        ):
            self._live = live
            decoder = codecs.getincrementaldecoder("utf-8")("ignore")
            if initial_prompt:
                self._enqueue_prompt(initial_prompt)
            self._refresh()
            while True:
                self._drain_agent_events()
                self._start_next_prompt_if_idle()
                if self._exit_after_active and not self._agent_busy:
                    break
                try:
                    key = read_key(decoder, timeout=0.05)
                except (EOFError, KeyboardInterrupt):
                    break
                if key.name == "timeout":
                    continue
                try:
                    if self._permission:
                        self._handle_permission_key(key)
                    elif self._handle_prompt_key(key, decoder):
                        break
                except (EOFError, KeyboardInterrupt):
                    break
                self._refresh()
        self._disable_permission_mouse()
        self._live = None

    def _run_plain(self, initial_prompt: str | None):
        self._print(self._header())
        if initial_prompt:
            self._submit(initial_prompt)
        while True:
            try:
                prompt = self.console.input(self._input_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            if not prompt:
                continue
            if prompt.lower() in EXIT_COMMANDS:
                break
            self._submit(prompt)

    def _load_and_render_history(self):
        self._ensure_state()
        for msg in self.conversation.load_history():
            if msg.type == "human":
                self._append(self._user_message(str(msg.content)))
            else:
                for event in self._events_from_message(msg):
                    self._render_event(event)
        for call in self._pending_tool_calls:
            self._append(self._tool_call(call))
        self._pending_tool_calls.clear()

    def _events_from_message(self, msg):
        if msg.type == "ai" and msg.content:
            yield AgentEvent("assistant", str(msg.content))
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                yield AgentEvent("tool_call", str(tc["args"]), tc["name"], tc.get("id"))
        elif msg.type == "tool":
            yield AgentEvent(
                "tool_result", str(msg.content), getattr(msg, "name", None), getattr(msg, "tool_call_id", None)
            )

    def _submit(self, prompt: str):
        self._ensure_state()
        self._input = ""
        self._append(self._user_message(prompt))
        self._status = Text("thinking", style="cyan")
        self._pending_tool_calls = []
        self._refresh()
        first = True
        for event in self.conversation.send(prompt):
            if first:
                self._status = None
                first = False
            self._render_event(event)
            self._refresh()
        for call in self._pending_tool_calls:
            self._append(self._tool_call(call))
        self._pending_tool_calls.clear()
        self._status = None
        self._refresh()

    def _enqueue_prompt(self, prompt: str):
        self._ensure_state()
        self._prompt_queue.append(prompt)
        self._input = ""
        self._refresh()

    def _start_next_prompt_if_idle(self):
        self._ensure_state()
        if self._agent_busy or not self._prompt_queue:
            return
        prompt = self._prompt_queue.popleft()
        self._append(self._user_message(prompt))
        self._status = Text("thinking", style="cyan")
        self._pending_tool_calls = []
        self._agent_busy = True
        self._active_prompt = prompt
        self._active_first_event = True
        self._agent_thread = threading.Thread(target=self._run_agent_worker, args=(prompt,), daemon=True)
        self._agent_thread.start()
        self._refresh()

    def _run_agent_worker(self, prompt: str):
        try:
            for event in self.conversation.send(prompt):
                self._agent_event_queue.put(AgentThreadMessage("agent_event", event))
        except Exception as error:
            self._agent_event_queue.put(AgentThreadMessage("agent_error", error))
        finally:
            self._agent_event_queue.put(AgentThreadMessage("agent_done"))

    def _drain_agent_events(self):
        self._ensure_state()
        changed = False
        while True:
            try:
                message = self._agent_event_queue.get_nowait()
            except queue.Empty:
                break
            changed = True
            self._handle_agent_thread_message(message)
        if changed:
            self._refresh()

    def _handle_agent_thread_message(self, message: AgentThreadMessage):
        if message.kind == "agent_event":
            if self._active_first_event:
                self._status = None
                self._active_first_event = False
            self._render_event(message.payload)
        elif message.kind == "agent_error":
            self._append(Text(f"Error: {message.payload}", style="red"))
        elif message.kind == "permission_request":
            request, reply_queue = message.payload
            self._permission = PermissionRequest(request.name, request.args)
            self._permission_reply_queue = reply_queue
            self._enable_permission_mouse()
        elif message.kind == "agent_done":
            for call in self._pending_tool_calls:
                self._append(self._tool_call(call))
            self._pending_tool_calls.clear()
            self._status = None
            self._agent_busy = False
            self._active_prompt = None
            self._active_first_event = False
            self._agent_thread = None
            self._start_next_prompt_if_idle()

    def _read_terminal_prompt(self) -> str:
        self._ensure_state()
        self._input = ""
        decoder = codecs.getincrementaldecoder("utf-8")("ignore")
        self._refresh()
        while True:
            key = read_key(decoder)
            if key.name == "enter":
                prompt = self._input.strip()
                self._input = ""
                self._refresh()
                return prompt
            if key.name == "ctrl_c":
                raise KeyboardInterrupt
            if key.name == "ctrl_d":
                raise EOFError
            if key.name == "backspace":
                self._input = self._input[:-1]
                decoder.reset()
            elif key.name == "partial":
                pass
            elif key.name == "char" and key.value.isprintable():
                self._input += key.value
            self._refresh()

    def _permission_prompt(self, request: ToolPermissionRequest) -> PermissionDecision:
        self._ensure_state()
        if self._live is None:
            choice = self._read_permission_choice(request.name, request.args)
        else:
            reply_queue: queue.Queue[str] = queue.Queue(maxsize=1)
            self._agent_event_queue.put(AgentThreadMessage("permission_request", (request, reply_queue)))
            choice = reply_queue.get()
        if choice == "a":
            return PermissionDecision(allowed=True, grant_pattern=request.grant_pattern)
        return PermissionDecision(allowed=choice == "y")

    def _read_permission_choice(self, name: str, args: str) -> str:
        if self._live is None:
            return self._read_permission_choice_plain(name, args)

        choices = ["y", "a", "n"]
        self._permission = PermissionRequest(name, args)
        self._enable_permission_mouse()
        self._refresh()
        try:
            while True:
                key = read_key()
                if key.name == "enter":
                    return choices[self._permission.selected]
                if key.name in {"escape", "ctrl_c", "ctrl_d"}:
                    return "n"
                if key.name == "scroll_up":
                    self._permission.scroll_offset = max(0, self._permission.scroll_offset - 3)
                elif key.name == "scroll_down":
                    self._permission.scroll_offset += 3
                elif key.name == "up":
                    self._permission.selected = (self._permission.selected - 1) % len(choices)
                elif key.name == "down":
                    self._permission.selected = (self._permission.selected + 1) % len(choices)
                elif key.name == "char":
                    lower = key.value.lower()
                    if lower in ("1", "y"):
                        return "y"
                    if lower in ("2", "a"):
                        return "a"
                    if lower in ("3", "n"):
                        return "n"
                self._refresh()
        finally:
            self._permission = None
            self._disable_permission_mouse()
            self._refresh()

    def _read_permission_choice_plain(self, name: str, args: str) -> str:
        self._print(self._permission_panel(PermissionRequest(name, args)))
        choice = Prompt.ask(
            Text("Select", style="cyan"),
            console=self.console,
            choices=["1", "2", "3", "y", "a", "n"],
            default="1",
            show_choices=False,
            show_default=False,
        )
        choices = {"1": "y", "y": "y", "2": "a", "a": "a", "3": "n", "n": "n"}
        return choices[choice.lower()]

    def _handle_prompt_key(self, key: KeyPress, decoder) -> bool:
        if key.name == "enter":
            prompt = self._input.strip()
            self._input = ""
            decoder.reset()
            if not prompt:
                return False
            if prompt.lower() in EXIT_COMMANDS:
                if self._agent_busy:
                    self._prompt_queue.clear()
                    self._exit_after_active = True
                    return False
                return True
            self._enqueue_prompt(prompt)
        elif key.name == "ctrl_c":
            raise KeyboardInterrupt
        elif key.name == "ctrl_d":
            raise EOFError
        elif key.name == "backspace":
            self._input = self._input[:-1]
            decoder.reset()
        elif key.name == "escape":
            self._undo_last_queued_prompt()
        elif key.name == "partial":
            pass
        elif key.name == "char" and key.value.isprintable():
            self._input += key.value
        return False

    def _handle_permission_key(self, key: KeyPress):
        if not self._permission:
            return
        choices = ["y", "a", "n"]
        if key.name == "enter":
            self._finish_permission_choice(choices[self._permission.selected])
        elif key.name in {"escape", "ctrl_c", "ctrl_d"}:
            self._finish_permission_choice("n")
        elif key.name == "scroll_up":
            self._permission.scroll_offset = max(0, self._permission.scroll_offset - 3)
        elif key.name == "scroll_down":
            self._permission.scroll_offset += 3
        elif key.name == "up":
            self._permission.selected = (self._permission.selected - 1) % len(choices)
        elif key.name == "down":
            self._permission.selected = (self._permission.selected + 1) % len(choices)
        elif key.name == "char":
            lower = key.value.lower()
            if lower in ("1", "y"):
                self._finish_permission_choice("y")
            elif lower in ("2", "a"):
                self._finish_permission_choice("a")
            elif lower in ("3", "n"):
                self._finish_permission_choice("n")

    def _finish_permission_choice(self, choice: str):
        reply_queue = self._permission_reply_queue
        self._permission = None
        self._permission_reply_queue = None
        self._disable_permission_mouse()
        if reply_queue is not None:
            reply_queue.put(choice)

    def _undo_last_queued_prompt(self):
        if self._prompt_queue:
            self._prompt_queue.pop()

    def _write_terminal_control(self, sequence: str):
        try:
            with Path("/dev/tty").open("w", encoding="utf-8") as terminal:
                terminal.write(sequence)
                terminal.flush()
        except OSError:
            pass

    def _enable_permission_mouse(self):
        if not self._permission_mouse_enabled and self.console.is_terminal:
            self._write_terminal_control("\x1b[?1000h\x1b[?1006h")
            self._permission_mouse_enabled = True

    def _disable_permission_mouse(self):
        if self._permission_mouse_enabled:
            self._write_terminal_control("\x1b[?1006l\x1b[?1000l")
            self._permission_mouse_enabled = False

    def _render_event(self, event: AgentEvent):
        if event.kind == "tool_call":
            self._pending_tool_calls.append(event)
        elif event.kind == "tool_result":
            call = self._pop_tool_call(event.tool_call_id)
            if call:
                self._append(self._tool_call(call))
            self._append(self._tool_result(event))
        elif event.kind == "assistant":
            for call in self._pending_tool_calls:
                self._append(self._tool_call(call))
            self._pending_tool_calls.clear()
            self._append(self._assistant_message(event.content))

    def _pop_tool_call(self, tool_call_id: str | None):
        if tool_call_id is None:
            return self._pending_tool_calls.pop(0) if self._pending_tool_calls else None
        for i, call in enumerate(self._pending_tool_calls):
            if call.tool_call_id == tool_call_id:
                return self._pending_tool_calls.pop(i)
        return self._pending_tool_calls.pop(0) if self._pending_tool_calls else None

    def _view(self) -> TerminalView:
        if not hasattr(self, "view"):
            self.view = TerminalView(self.console)
        self.view.console = self.console
        return self.view

    def _screen(self):
        if self._permission:
            return self._permission_panel(self._permission)
        return self._view().render_live_prompt(self._prompt_live_view_model())

    def _prompt_live_view_model(self) -> PromptLiveViewModel:
        return PromptLiveViewModel(
            input_text=self._input,
            status=self._status.plain if self._status else None,
            agent_busy=self._agent_busy,
            exit_after_active=self._exit_after_active,
            queued_prompts=tuple(self._prompt_queue),
        )

    def _permission_dialog_view_model(self) -> PermissionDialogViewModel:
        assert self._permission is not None
        return PermissionDialogViewModel(
            name=self._permission.name,
            args=self._permission.args,
            selected=self._permission.selected,
            scroll_offset=self._permission.scroll_offset,
            terminal_height=self.console.height,
        )

    def _status_line(self):
        return self._view()._status_line(
            self._status.plain if self._status else None,
            self._agent_busy,
            self._exit_after_active,
        )

    def _queue_panel(self):
        return self._view()._queue_panel(tuple(self._prompt_queue))

    def _compact_prompt(self, prompt: str):
        return self._view()._compact_prompt(prompt)

    def _input_panel(self):
        return self._view()._input_panel(self._input)

    def _permission_panel(self, request: PermissionRequest):
        model = PermissionDialogViewModel(
            name=request.name,
            args=request.args,
            selected=request.selected,
            scroll_offset=request.scroll_offset,
            terminal_height=self.console.height,
        )
        panel = self._view().render_permission_dialog(model)
        request.scroll_offset = self._clamped_permission_scroll_offset(request)
        return panel

    def _clamped_permission_scroll_offset(self, request: PermissionRequest) -> int:
        diff = self._permission_preview_diff(request.name, request.args)
        if diff is None:
            return request.scroll_offset
        max_lines = max(self.console.height - 11, 3)
        return max(0, min(request.scroll_offset, max(len(diff.splitlines()) - max_lines, 0)))

    def _permission_preview_diff(self, name: str, args: str) -> str | None:
        parsed_args = self._parse_tool_args(args)
        if not isinstance(parsed_args, dict) or not parsed_args:
            return None
        if name == "edit":
            path = parsed_args.get("path")
            old_string = parsed_args.get("old_string")
            new_string = parsed_args.get("new_string")
            if not isinstance(path, str) or not isinstance(old_string, str) or not isinstance(new_string, str):
                return None
            return self._edit_diff(path, old_string, new_string, parsed_args.get("replace_all") is True)
        if name == "write":
            path = parsed_args.get("path")
            content = parsed_args.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                return None
            before = self._read_edit_preview_source(path)
            if before is None:
                return self._content_diff(path, "", content).replace(f"• Update({path})", f"• Create({path})", 1)
            return self._content_diff(path, before, content, replace_all=True)
        return None

    def _append(self, renderable: RenderableType):
        self._history.append(renderable)
        self.console.print(renderable)

    def _refresh(self):
        if self._live:
            self._live.update(self._screen(), refresh=True)

    def _print(self, renderable: RenderableType):
        self.console.print(renderable)

    def _header(self):
        return self._view().render_header(self.conversation.session_id)

    def _user_message(self, prompt: str):
        return self._view().render_transcript_event(TranscriptViewEvent("user", prompt))

    def _assistant_message(self, content: str):
        return self._view().render_transcript_event(TranscriptViewEvent("assistant", content))

    def _tool_call(self, event: AgentEvent):
        return self._view().render_transcript_event(
            TranscriptViewEvent("tool_call", event.content, event.name, event.tool_call_id)
        )

    def _tool_result(self, event: AgentEvent):
        return self._view().render_transcript_event(
            TranscriptViewEvent("tool_result", event.content, event.name, event.tool_call_id)
        )

    def _edit_preview(self, args: dict, scroll_offset: int = 0):
        return self._view()._edit_preview(args, scroll_offset, self.console.height)

    def _write_preview(self, args: dict, scroll_offset: int = 0):
        return self._view()._write_preview(args, scroll_offset, self.console.height)

    def _scrollable_diff_preview(self, diff: str, offset: int) -> Syntax:
        max_lines = max(self.console.height - 11, 3)
        lines = diff.splitlines()
        offset = max(0, min(offset, max(len(lines) - max_lines, 0)))
        if self._permission:
            self._permission.scroll_offset = offset
        visible_lines = lines[offset : offset + max_lines]
        return Syntax("\n".join(visible_lines), "diff", theme="monokai", word_wrap=False)

    def _edit_diff(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        return self._view()._edit_diff(path, old_string, new_string, replace_all)

    def _read_edit_preview_source(self, path: str) -> str | None:
        return self._view()._read_edit_preview_source(path)

    def _replacement_diff(
        self, path: str, content: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str | None:
        return self._view()._replacement_diff(path, content, old_string, new_string, replace_all)

    def _match_line_starts(self, lines: list[str], match_lines: list[str]) -> list[int]:
        return self._view()._match_line_starts(lines, match_lines)

    def _edit_summary_for_replacements(self, removed_per_match: int, added_per_match: int, count: int) -> str:
        return self._view()._edit_summary_for_replacements(removed_per_match, added_per_match, count)

    def _content_diff(self, path: str, before: str, after: str, replace_all: bool = False) -> str:
        return self._view()._content_diff(path, before, after, replace_all)

    def _edit_summary_for_changes(self, changes):
        return self._view()._edit_summary_for_changes(changes)

    def _permission_call_preview(self, name: str, args: str, scroll_offset: int = 0) -> str | Syntax:
        preview = self._view()._permission_call_preview(name, args, scroll_offset, self.console.height)
        if self._permission and isinstance(preview, Syntax):
            self._permission.scroll_offset = self._clamped_permission_scroll_offset(self._permission)
        return preview

    def _parse_tool_args(self, args: str):
        return self._view()._parse_tool_args(args)

    def _permission_title(self, name: str):
        return self._view()._permission_title(name)

    def _input_prompt(self):
        return self._view().render_input_prompt()

    def _input_line(self, prompt: str):
        return self._view()._input_line(prompt)

    def _input_help(self):
        return self._view()._input_help()

    def _draw_input_block(self):
        renderables = self._view().render_input_block()
        for renderable in renderables[:-1]:
            self.console.print(renderable)
        self.console.print(renderables[-1], end="")


def read_stdin_byte(fd: int) -> bytes:
    if not hasattr(sys.stdin, "isatty"):
        return sys.stdin.buffer.read(1)
    return os.read(fd, 1)


def read_key(decoder=None, timeout: float | None = None) -> KeyPress:
    fd = sys.stdin.fileno()
    if timeout is not None and not select.select([fd], [], [], timeout)[0]:
        return KeyPress("timeout")
    chunk = read_stdin_byte(fd)
    if not chunk:
        return KeyPress("eof")
    byte = chunk[0]
    if byte in (10, 13):
        return KeyPress("enter")
    if byte == 3:
        return KeyPress("ctrl_c")
    if byte == 4:
        return KeyPress("ctrl_d")
    if byte in (8, 127):
        return KeyPress("backspace")
    if byte == 27:
        sequence = read_escape_sequence(fd)
        if sequence in ("[A", "OA"):
            return KeyPress("up")
        if sequence in ("[B", "OB"):
            return KeyPress("down")
        if sequence.startswith("[<64;") and sequence.endswith("M"):
            return KeyPress("scroll_up")
        if sequence.startswith("[<65;") and sequence.endswith("M"):
            return KeyPress("scroll_down")
        if sequence.startswith("[<") and sequence[-1:] in ("M", "m"):
            return KeyPress("mouse")
        return KeyPress("escape")
    if decoder is None:
        return KeyPress("char", chunk.decode("utf-8", errors="ignore"))
    character = decoder.decode(chunk)
    if not character:
        return KeyPress("partial")
    return KeyPress("char", character)


def read_escape_sequence(fd: int) -> str:
    sequence = ""
    while select.select([fd], [], [], 0.05)[0]:
        chunk = read_stdin_byte(fd)
        if not chunk:
            break
        sequence += chr(chunk[0])
        if sequence.startswith("[<"):
            if chunk[0] in (ord("M"), ord("m")):
                break
        elif len(sequence) > 1 and 0x40 <= chunk[0] <= 0x7E:
            break
    return sequence


@contextmanager
def cbreak_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
