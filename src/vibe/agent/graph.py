from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from vibe.agent.state import AgentState

SYSTEM_PROMPT = """You are Vibe, a coding agent running in the user's terminal.
You help with software engineering tasks by reading files, running commands, and writing code.
Be concise. Explain what you're doing, not what you could do.
When using tools, prefer the most direct approach."""


def build_graph(llm, tools: list):
    """构建 agent graph：llm_call → should_continue → tool_node → 循环"""
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def llm_call(state: AgentState):
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm_call")

    return graph.compile()
