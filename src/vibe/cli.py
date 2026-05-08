import argparse
import os

import vibe  # noqa: F401 — ensures deprecation-warning filter runs before langchain imports

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from vibe.agent.graph import build_graph
from vibe.session.store import SessionStore
from vibe.tools import ALL_TOOLS


def run(prompt: str, session_id: str | None = None, model: str = "gpt-4o", base_url: str | None = None):
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY"))
    agent = build_graph(llm, ALL_TOOLS)
    store = SessionStore()

    if session_id is None:
        session_id = store.new_session()
        print(f"[session: {session_id}]")

    history = store.load(session_id)
    history.append(HumanMessage(content=prompt))
    store.append(session_id, history[-1])

    result = agent.invoke({"messages": history})
    response_messages = result["messages"][len(history) :]

    for msg in response_messages:
        store.append(session_id, msg)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 {tc['name']}({tc['args']})")
        elif msg.content and msg.type == "ai":
            print(msg.content)
        elif msg.type == "tool":
            print(f"  → {msg.content[:500]}")


def main():
    parser = argparse.ArgumentParser(description="Vibe - a coding agent")
    parser.add_argument("prompt", help="What you want the agent to do")
    parser.add_argument("-s", "--session", help="Resume an existing session")
    parser.add_argument("-m", "--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible API endpoint")
    args = parser.parse_args()
    run(args.prompt, args.session, args.model, args.base_url)


if __name__ == "__main__":
    main()
