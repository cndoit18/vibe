import codecs
import select
import sys
import termios
import tty
from contextlib import contextmanager

from rich.console import Console, RenderableType
from rich.control import Control
from rich.markdown import Markdown
from rich.rule import Rule
from rich.segment import ControlType
from rich.text import Text

from vibe.agent.conversation import AgentConversation, AgentEvent

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", ":q"}
ERASE_LINE = Control((ControlType.ERASE_IN_LINE, 2))


class VibeTUI:
    def __init__(
        self,
        session_id: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.console = Console()
        self.conversation = AgentConversation(session_id=session_id, model=model, base_url=base_url, api_key=api_key)

    def run(self, initial_prompt: str | None = None):
        self._print(self._header())
        if initial_prompt:
            self._submit(initial_prompt)

        while True:
            try:
                prompt, prompt_is_visible = self._read_prompt()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            if not prompt:
                continue
            if prompt.lower() in EXIT_COMMANDS:
                break
            self._submit(prompt, render_user=not prompt_is_visible)

    def _submit(self, prompt: str, *, render_user: bool = True):
        if render_user:
            self._print(self._user_message(prompt))
        self._print(Text("thinking", style="cyan"))
        self._pending_tool_calls = []
        first = True
        for event in self.conversation.send(prompt):
            if first:
                self.console.control(Control.move_to_column(1, -1), ERASE_LINE)
                first = False
            self._render_event(event)
        for call in self._pending_tool_calls:
            self._print(self._tool_call(call))
        self._pending_tool_calls.clear()
        if first:
            self.console.control(Control.move_to_column(1, -1), ERASE_LINE)

    def _render_event(self, event: AgentEvent):
        if event.kind == "tool_call":
            self._pending_tool_calls.append(event)
        elif event.kind == "tool_result":
            call = self._pop_tool_call(event.tool_call_id)
            if call:
                self._print(self._tool_call(call))
            self._print(self._tool_result(event))
        elif event.kind == "assistant":
            for call in self._pending_tool_calls:
                self._print(self._tool_call(call))
            self._pending_tool_calls.clear()
            self._print(self._assistant_message(event.content))

    def _pop_tool_call(self, tool_call_id: str | None):
        if tool_call_id is None:
            return self._pending_tool_calls.pop(0) if self._pending_tool_calls else None
        for i, call in enumerate(self._pending_tool_calls):
            if call.tool_call_id == tool_call_id:
                return self._pending_tool_calls.pop(i)
        return self._pending_tool_calls.pop(0) if self._pending_tool_calls else None

    def _read_prompt(self) -> tuple[str, bool]:
        if not self.console.is_terminal or not sys.stdin.isatty():
            return self.console.input(self._input_prompt()).strip(), False

        self._draw_input_block()
        prompt = self._read_terminal_prompt().strip()
        self._finish_input_block()
        return prompt, False

    def _draw_input_block(self):
        self.console.print(self._input_rule())
        self.console.print(self._input_line(""))
        self.console.print(self._input_rule())
        self.console.print(self._input_help(), end="")
        self.console.control(Control.move_to_column(1, -2))

    def _read_terminal_prompt(self) -> str:
        prompt = ""
        decoder = codecs.getincrementaldecoder("utf-8")("ignore")
        with raw_terminal():
            self.console.control(Control.show_cursor(True))
            self._render_prompt(prompt)
            while True:
                chunk = sys.stdin.buffer.read(1)
                if not chunk:
                    raise EOFError

                byte = chunk[0]
                if byte in (10, 13):
                    self._render_prompt(prompt)
                    return prompt
                if byte == 3:
                    raise KeyboardInterrupt
                if byte == 4:
                    raise EOFError
                if byte in (8, 127):
                    prompt = prompt[:-1]
                    decoder.reset()
                    self._render_prompt(prompt)
                    continue
                if byte == 27:
                    self._discard_escape_sequence()
                    continue

                character = decoder.decode(chunk)
                if character and character.isprintable():
                    prompt += character
                    self._render_prompt(prompt)

    def _render_prompt(self, prompt: str):
        line = Text.assemble(self._input_line(prompt))
        self.console.control(Control.move_to_column(1), ERASE_LINE)
        self.console.print(line, end="")
        prompt_cells = Text(prompt).cell_len
        cursor_col = min(2 + prompt_cells + 1, self.console.width)
        self.console.control(Control.move_to_column(cursor_col))

    def _discard_escape_sequence(self):
        fd = sys.stdin.fileno()
        while select.select([fd], [], [], 0.01)[0]:
            chunk = sys.stdin.buffer.read(1)
            if not chunk or 0x40 <= chunk[0] <= 0x7E:
                break

    def _finish_input_block(self):
        self.console.control(
            Control.move_to_column(1, -1),
            ERASE_LINE,
            Control.move_to_column(1, 1),
            ERASE_LINE,
            Control.move_to_column(1, 1),
            ERASE_LINE,
            Control.move_to_column(1, 1),
            ERASE_LINE,
            Control.move_to_column(1, -3),
        )

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

    def _input_prompt(self):
        return Text.assemble(("› ", "bold bright_black"))

    def _input_rule(self):
        return Rule(style="bright_black")

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


@contextmanager
def raw_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
