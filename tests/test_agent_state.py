from langchain_core.messages import AIMessage, HumanMessage

from vibe.agent.state import AgentState


class TestAgentState:
    def test_accepts_messages(self):
        state = AgentState(messages=[HumanMessage(content="hi")])
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "hi"

    def test_add_messages_reducer_appends(self):
        state = AgentState(messages=[HumanMessage(content="q")])
        # Simulate the reducer: AgentState with new messages appended
        updated = AgentState(messages=state["messages"] + [AIMessage(content="a")])
        assert len(updated["messages"]) == 2
        assert updated["messages"][1].content == "a"
