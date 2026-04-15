from graph.builder import build_graph
from graph.state import build_initial_state


class AgentOrchestrator:

    def __init__(self):
        self.graph = build_graph()

    def run(
        self,
        query: str,
        user_id: str,
        session_id: str = "",
        system_prompt: str = "",
    ):
        # 传入 thread_id，确保 checkpointer 按 session 维度读写状态。
        result = self.graph.invoke(
            build_initial_state(query, user_id, session_id, system_prompt),
            config={"configurable": {"thread_id": session_id}},
        )

        print("result===========result \n", result, "\n")

        return result.get("final_answer", "")

    def stream(
        self,
        query: str,
        user_id: str,
        session_id: str = "",
        system_prompt: str = "",
    ):
        # stream_mode="messages" 会把图中 LLM 节点产生的消息增量往外透传。
        return self.graph.stream(
            build_initial_state(query, user_id, session_id, system_prompt),
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        )