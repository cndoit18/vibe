from collections.abc import Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

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


def make_tool_executor(tools: list, permission_callback: Callable[[str, str], bool] | None):
    tools_by_name = {t.name: t for t in tools}

    def tool_node(state: AgentState):
        last = state["messages"][-1]
        result_messages = []
        permission_denied = False
        for tc in last.tool_calls:
            if permission_denied:
                result_messages.append(
                    ToolMessage(content="Tool call skipped.", tool_call_id=tc["id"], name=tc["name"])
                )
            elif permission_callback and not permission_callback(tc["name"], str(tc["args"])):
                permission_denied = True
                result_messages.append(
                    ToolMessage(content="Permission denied by user.", tool_call_id=tc["id"], name=tc["name"])
                )
            else:
                result = tools_by_name[tc["name"]].invoke(tc["args"])
                result_messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": result_messages, "permission_denied": permission_denied}

    return tool_node


def build_graph(llm, tools: list, permission_callback: Callable[[str, str], bool] | None = None):
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
