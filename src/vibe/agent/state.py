from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Agent 的状态就是消息列表。MessagesState 提供了 add_messages reducer 自动追加。"""
    pass
