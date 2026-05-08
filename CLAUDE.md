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
- `src/vibe/tools/bash.py` — shell execution tool (30s timeout); `read`/`write`/`edit` tools planned
- `src/vibe/session/store.py` — JSONL append-only session persistence
- `src/vibe/cli.py` — argparse CLI entry point
