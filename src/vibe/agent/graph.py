from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from vibe.agent.permissions import (
    PermissionDecision,
    ToolPermissionRequest,
    add_permission_grant,
    build_permission_request,
    normalize_decision,
    permission_matches,
    workspace_permission_allows,
    workspace_permission_error,
)
from vibe.agent.state import AgentState
from vibe.prompts import load_prompt

SYSTEM_PROMPT = load_prompt("system")


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def after_tools(state: AgentState) -> str:
    if state.get("permission_denied", False):
        return "end"
    return "llm_call"


def make_tool_executor(
    tools: list,
    permission_callback: Callable[[ToolPermissionRequest], PermissionDecision | bool] | None,
):
    tools_by_name = {t.name: t for t in tools}

    def tool_node(state: AgentState):
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {"messages": [], "permission_denied": False}

        result_messages = []
        permission_denied = False
        permission_grants = list(state.get("permission_grants", []))
        workspace_root = state.get("workspace_root") or str(Path.cwd().resolve())

        for tc in last.tool_calls:
            if permission_denied:
                result_messages.append(ToolMessage(content="Tool call skipped.", tool_call_id=tc["id"], name=tc["name"]))
                continue

            tool = tools_by_name.get(tc["name"])
            if tool is None:
                available = ", ".join(sorted(tools_by_name))
                result_messages.append(
                    ToolMessage(
                        content=f"Error: {tc['name']} is not a valid tool; available tools: {available}",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
                continue

            request = build_permission_request(tc["name"], tc["args"], workspace_root)
            if not workspace_permission_allows(request):
                result_messages.append(
                    ToolMessage(content=workspace_permission_error(request), tool_call_id=tc["id"], name=tc["name"])
                )
                continue

            if permission_callback and not permission_matches(permission_grants, request.target):
                decision = normalize_decision(permission_callback(request))
                if not decision.allowed:
                    permission_denied = True
                    result_messages.append(
                        ToolMessage(content="Permission denied by user.", tool_call_id=tc["id"], name=tc["name"])
                    )
                    continue
                permission_grants = add_permission_grant(permission_grants, decision.grant_pattern)

            try:
                result = tool.invoke(tc["args"])
            except Exception as error:
                result = f"Error: {error}"
            result_messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))

        return {"messages": result_messages, "permission_denied": permission_denied, "permission_grants": permission_grants}

    return tool_node


def build_graph(
    llm,
    tools: list,
    permission_callback: Callable[[ToolPermissionRequest], PermissionDecision | bool] | None = None,
):
    llm_with_tools = llm.bind_tools(tools)

    def llm_call(state: AgentState):
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tools", make_tool_executor(tools, permission_callback))
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {"tools": "tools", "end": END})
    graph.add_conditional_edges("tools", after_tools, {"llm_call": "llm_call", "end": END})

    return graph.compile()
