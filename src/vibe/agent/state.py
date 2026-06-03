from langgraph.graph import MessagesState


class AgentState(MessagesState):
    permission_denied: bool
