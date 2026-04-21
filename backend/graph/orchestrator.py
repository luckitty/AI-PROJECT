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

        # print("result===========result \n", result[:200], "\n")

        return result.get("final_answer", "")

    def stream(
        self,
        query: str,
        user_id: str,
        session_id: str = "",
        system_prompt: str = "",
    ):
        # 使用 custom：由 response 节点内 get_stream_writer 显式写入增量，
        # 不依赖 LLM 回调链，避免「整段才出」或中间长时间无包（messages 模式在自定义节点里不稳定）。
        return self.graph.stream(
            build_initial_state(query, user_id, session_id, system_prompt),
            config={"configurable": {"thread_id": session_id}},
            stream_mode="custom",
        )