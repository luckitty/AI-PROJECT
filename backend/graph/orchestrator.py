from langgraph.types import Command

from graph.builder import build_graph
from graph.state import build_initial_state

# 人机断点很多时节点的 interrupt 次数上限，防止异常情况下死循环。
MAX_HITL_RESUME_STEPS = 64


def graph_run_config(session_id: str, user_id: str, run_mode: str) -> dict:
    """
    组装 LangGraph 的 RunnableConfig：保留 thread_id 供 Redis 检查点；
    metadata / tags 供 LangSmith 按用户、会话、同步或流式筛选 Trace。
    """
    thread_id = session_id or ""
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "user_id": user_id or "",
            "session_id": thread_id,
            "run_mode": run_mode,
        },
        "tags": ["travel-guide-agent", run_mode],
    }


class AgentOrchestrator:

    def __init__(self):
        self.graph = build_graph()

    def run(
        self,
        query: str,
        user_id: str,
        session_id: str = "",
        system_prompt: str = "",
        resume=None,
        human_breakpoints: bool = False,
    ):
        """
        执行 ``invoke``。
        - ``human_breakpoints=False``（默认）：遇 ``interrupt()`` 时在服务端自动 ``resume=True`` 直到跑完，兼容未改造的前端。
        - ``human_breakpoints=True``：首次挂起即返回，由客户端传 ``resume`` 逐步恢复。
        """
        config = graph_run_config(session_id, user_id, "invoke")
        if resume is not None:
            result = self.graph.invoke(Command(resume=resume), config=config)
        else:
            result = self.graph.invoke(
                build_initial_state(
                    query, user_id, session_id, system_prompt, stream_sink_active=False
                ),
                config=config,
            )
        if human_breakpoints:
            return result
        steps = 0
        while result.get("__interrupt__"):
            steps += 1
            if steps > MAX_HITL_RESUME_STEPS:
                raise RuntimeError(
                    "人机断点自动恢复次数超过上限，请检查图结构或启用 human_breakpoints 人工逐步 resume"
                )
            result = self.graph.invoke(Command(resume=True), config=config)
        return result

    def stream(
        self,
        query: str,
        user_id: str,
        session_id: str = "",
        system_prompt: str = "",
        resume=None,
    ):
        """
        流式执行：``updates`` 用于透出 ``interrupt()``，``custom`` 用于正文增量。
        产出为元组 ``(mode, payload)``（多模式 stream 的默认形态）。
        """
        config = graph_run_config(session_id, user_id, "stream")
        if resume is not None:
            stream_input = Command(resume=resume)
        else:
            stream_input = build_initial_state(
                query, user_id, session_id, system_prompt, stream_sink_active=True
            )
        return self.graph.stream(
            stream_input,
            config=config,
            stream_mode=["updates", "custom"],
        )
