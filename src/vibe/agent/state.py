from langgraph.graph import MessagesState


class AgentState(MessagesState):
    permission_denied: bool
    permission_grants: list[str]
    workspace_root: str
