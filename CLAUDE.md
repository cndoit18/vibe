# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

- Install: `uv sync`
- Run agent: `uv run vibe "prompt"` (supports `-s <session_id>`, `-m <model>`, `--base-url <url>`)
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`

## Environment Variables

- `OPENAI_API_KEY` — required at runtime; unset = OpenAI SDK error on invoke
- `OPENAI_BASE_URL` — optional; used as default for `--base-url` CLI flag (OpenAI-compatible endpoints)

## Code Style

- Line length: 120 (configured in pyproject.toml; differs from ruff default 88)
- Comments and docstrings: English (docs/ markdown files are in Chinese)
- Python: 3.12+ (`.python-version` is 3.13)

## Architecture

LangGraph `StateGraph` ReAct loop: `START → llm_call → should_continue? → tools → llm_call (loop) / END`

- `src/vibe/agent/graph.py` — graph definition, system prompt, node wiring
- `src/vibe/agent/state.py` — `AgentState(MessagesState)`
- `src/vibe/tools/bash.py` — shell execution tool with 30s timeout
- `src/vibe/tools/read.py`, `write.py`, `edit.py` — file tools guarded by hook-based `src/vibe/tools/runtime.py`
- `src/vibe/tools/runtime.py` — hook-based runtime tools that implement LangChain `BaseTool` directly
- `src/vibe/session/store.py` — JSONL append-only session persistence
- `src/vibe/cli.py` — argparse CLI entry point

## Tool Safety & Sessions

- File tools are constrained to the current working directory, reject files over 1 MB, and truncate output at 30,000 chars.
- Sessions are append-only JSONL files under `~/.vibe/sessions`.

## References

- Use `@docs/01-tech-stack.md`, `@docs/02-architecture.md`, and `@docs/03-roadmap.md` for deeper design context when needed.
