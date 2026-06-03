import argparse
import os

from vibe.agent.conversation import AgentConversation
from vibe.tui import VibeTUI


def run_print(prompt: str, session_id: str | None = None, model: str = "gpt-4o", base_url: str | None = None):
    conversation = AgentConversation(session_id=session_id, model=model, base_url=base_url)
    if session_id is None:
        print(f"[session: {conversation.session_id}]")

    for event in conversation.send(prompt):
        if event.kind == "tool_call":
            print(f"  🔧 {event.name}({event.content})")
        elif event.kind == "assistant":
            print(event.content)
        elif event.kind == "tool_result":
            print(f"  → {event.content[:500]}")


def main():
    parser = argparse.ArgumentParser(description="Vibe - a coding agent")
    parser.add_argument("prompt", nargs="?", help="What you want the agent to do")
    parser.add_argument("-s", "--session", help="Resume an existing session")
    parser.add_argument("-m", "--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible API endpoint")
    parser.add_argument(
        "--print", action="store_true", help="Run once and print plain output instead of opening the TUI"
    )
    args = parser.parse_args()

    if args.print:
        if args.prompt is None:
            parser.error("--print requires a prompt")
        run_print(args.prompt, args.session, args.model, args.base_url)
    else:
        VibeTUI(args.session, args.model, args.base_url).run(args.prompt)


if __name__ == "__main__":
    main()
