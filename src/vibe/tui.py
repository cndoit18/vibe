import ast
import codecs
import os
import select
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from vibe.agent.conversation import AgentConversation, AgentEvent

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


class VibeTUI:
    def __init__(
        self,
        session_id: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.console = Console()
        self._history: list[RenderableType] = []
        self._input = ""
        self._status: Text | None = None
        self._permission: PermissionRequest | None = None
        self._permission_allow_all = False
        self._live: Live | None = None
        self._pending_tool_calls: list[AgentEvent] = []
        self.conversation = AgentConversation(
            session_id=session_id,
            model=model,
            base_url=base_url,
            api_key=api_key,
            permission_callback=self._permission_prompt,
        )

    def _ensure_state(self):
        if not hasattr(self, "_history"):
            self._history = []
        if not hasattr(self, "_input"):
            self._input = ""
        if not hasattr(self, "_status"):
            self._status = None
        if not hasattr(self, "_permission"):
            self._permission = None
        if not hasattr(self, "_permission_allow_all"):
            self._permission_allow_all = False
        if not hasattr(self, "_live"):
            self._live = None
        if not hasattr(self, "_pending_tool_calls"):
            self._pending_tool_calls = []

    def run(self, initial_prompt: str | None = None):
        self._ensure_state()
        if not self.console.is_terminal or not sys.stdin.isatty():
            self._run_plain(initial_prompt)
            return

        self._append(self._header())
        with (
            cbreak_terminal(),
            Live(self._screen(), console=self.console, screen=False, auto_refresh=False, transient=False) as live,
        ):
            self._live = live
            self._refresh()
            if initial_prompt:
                self._submit(initial_prompt)
            while True:
                try:
                    prompt = self._read_terminal_prompt()
                except (EOFError, KeyboardInterrupt):
                    break
                if not prompt:
                    continue
                if prompt.lower() in EXIT_COMMANDS:
                    break
                self._submit(prompt)
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

    def _permission_prompt(self, name: str, args: str) -> bool:
        if self._permission_allow_all:
            return True
        choice = self._read_permission_choice(name, args)
        if choice == "a":
            self._permission_allow_all = True
            return True
        return choice == "y"

    def _read_permission_choice(self, name: str, args: str) -> str:
        if self._live is None:
            return self._read_permission_choice_plain(name, args)

        choices = ["y", "a", "n"]
        self._permission = PermissionRequest(name, args)
        self._refresh()
        try:
            while True:
                key = read_key()
                if key.name == "enter":
                    return choices[self._permission.selected]
                if key.name in {"escape", "ctrl_c", "ctrl_d"}:
                    return "n"
                if key.name == "up":
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

    def _screen(self):
        return self._permission_panel(self._permission) if self._permission else self._input_panel()

    def _input_panel(self):
        line = Text.assemble(("› ", "bold bright_black"), (self._input, "white"), ("█", "white"))
        return Panel(line, border_style="bright_black", padding=(0, 1))

    def _permission_panel(self, request: PermissionRequest):
        options = [
            "Yes",
            "Yes, allow all tools during this session",
            "No",
        ]
        body = Text.assemble(
            (self._permission_title(request.name), "bold cyan"),
            "\n\n",
            (self._permission_call_preview(request.name, request.args), "dim"),
            "\n\n",
            ("Do you want to proceed?", "bright_black"),
            "\n",
        )
        for index, option in enumerate(options):
            prefix = "❯" if index == request.selected else " "
            style = "white" if index == request.selected else "bright_black"
            body.append(f"{prefix} {index + 1}. {option}\n", style=style)
        body.append("\nEsc to cancel · ↑/↓ to select · Enter to confirm", style="bright_black")
        return Panel(body, border_style="cyan", padding=(0, 1))

    def _append(self, renderable: RenderableType):
        self._history.append(renderable)
        self.console.print(renderable)

    def _refresh(self):
        if self._live:
            self._live.update(self._screen(), refresh=True)

    def _print(self, renderable: RenderableType):
        self.console.print(renderable)

    def _header(self):
        return Text.assemble(
            ("▸ vibe", "bold cyan"),
            (f"  session {self.conversation.session_id} · /exit to quit", "dim"),
        )

    def _user_message(self, prompt: str):
        return self._input_line(prompt)

    def _assistant_message(self, content: str):
        return Markdown(content.strip("\n"))

    def _tool_call(self, event: AgentEvent):
        return Text.assemble(("* ", "yellow"), (event.name or "tool", "cyan"), (f" {event.content}", "dim"))

    def _tool_result(self, event: AgentEvent):
        lines = event.content.split("\n")
        first = f"  → {lines[0]}"
        rest = [f"    {line}" for line in lines[1:]]
        return Text("\n".join([first, *rest]), style="dim")

    def _permission_call_preview(self, name: str, args: str) -> str:
        parsed_args = self._parse_tool_args(args)
        if isinstance(parsed_args, dict) and parsed_args:
            first_arg = next(iter(parsed_args.values()))
            return f"{name}({first_arg})"
        return f"{name}({args})"

    def _parse_tool_args(self, args: str):
        try:
            return ast.literal_eval(args)
        except (SyntaxError, ValueError):
            return None

    def _permission_title(self, name: str):
        titles = {"read": "Read file", "write": "Write file", "edit": "Edit file", "bash": "Run command"}
        return titles.get(name, name.replace("_", " ").title())

    def _input_prompt(self):
        return Text.assemble(("› ", "bold bright_black"))

    def _input_line(self, prompt: str):
        line = Text.assemble(("› ", "bold bright_black"), (prompt, "white"))
        line.truncate(max(1, self.console.width - 1), overflow="crop", pad=True)
        line.stylize("on grey19")
        return line

    def _input_help(self):
        left = Text("  ? for shortcuts", style="dim")
        right = Text("/exit to quit", style="dim")
        padding = max(1, self.console.width - 1 - left.cell_len - right.cell_len)
        return Text.assemble(left, (" " * padding, "dim"), right)

    def _draw_input_block(self):
        self.console.print(Rule(style="bright_black"))
        self.console.print(self._input_line(""))
        self.console.print(Rule(style="bright_black"))
        self.console.print(self._input_help(), end="")


def read_stdin_byte(fd: int) -> bytes:
    if not hasattr(sys.stdin, "isatty"):
        return sys.stdin.buffer.read(1)
    return os.read(fd, 1)


def read_key(decoder=None) -> KeyPress:
    fd = sys.stdin.fileno()
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
        if len(sequence) > 1 and 0x40 <= chunk[0] <= 0x7E:
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
