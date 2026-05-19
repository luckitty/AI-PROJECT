"""
人机可恢复断点：在节点内调用 LangGraph ``interrupt()``，挂起后由调用方
``Command(resume=...)`` 恢复；resume 返回值即 ``interrupt()`` 在此节点内的出参。

约定 resume：
- ``True`` / ``{"action": "continue"}``：继续向下执行；
- ``False`` / ``{"action": "stop", "message": "可选文案"}``：终止本轮（human_halt + 追加一条 assistant 消息）。

若 ``human_halt`` 已为 True（人机拒绝或节点内协作 stop），不再调用 ``interrupt()``，避免二次挂起。
"""
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from graph.chat_messages import last_assistant_text


def normalize_human_resume(decision):
    """
    将 ``Command(resume=...)`` 规范成 ``halt``（人机选择终止）或 ``go``（继续）。
    """
    if decision is False:
        return "halt"
    if isinstance(decision, dict):
        act = decision.get("action")
        if act in ("stop", "halt"):
            return "halt"
        return "go"
    return "go"


def build_human_checkpoint(payload_builder):
    """
    用 payload_builder(state) 生成挂起时下发给前端的 JSON 可序列化 payload。
    """

    def checkpoint(state):
        # response_node 等已在节点内写过中断结果时，只做路由收口，不再弹出断点。
        if state.get("human_halt"):
            return {}
        payload = payload_builder(state)
        decision = interrupt(payload)
        if normalize_human_resume(decision) == "halt":
            msg = "请求已中断"
            if isinstance(decision, dict) and decision.get("message"):
                msg = str(decision["message"])
            return {"human_halt": True, "messages": [AIMessage(content=msg)]}
        return {"human_halt": False}

    return checkpoint


# 各断点节点绑定独立闭包，便于在调试器里区分栈与节点名。
human_checkpoint_start = build_human_checkpoint(
    lambda s: {
        "checkpoint": "before_planner",
        "query": s.get("query"),
        "session_id": s.get("session_id"),
    }
)

human_checkpoint_after_planner = build_human_checkpoint(
    lambda s: {
        "checkpoint": "after_planner",
        "session_id": s.get("session_id"),
        "need_rag": s.get("need_rag"),
        "need_memory": s.get("need_memory"),
        "need_tool": s.get("need_tool"),
        "need_amap_mcp": s.get("need_amap_mcp"),
        "need_history": s.get("need_history"),
    }
)

human_checkpoint_after_rag = build_human_checkpoint(
    lambda s: {
        "checkpoint": "after_rag",
        "session_id": s.get("session_id"),
        "need_rag": s.get("need_rag"),
    }
)

# 长期记忆节点之后：可能继续进本地旅游 RAG，也可能直接去 capability。
human_checkpoint_after_memory = build_human_checkpoint(
    lambda s: {
        "checkpoint": "after_memory",
        "session_id": s.get("session_id"),
        "need_rag": s.get("need_rag"),
        "need_memory": s.get("need_memory"),
    }
)

human_checkpoint_after_capability = build_human_checkpoint(
    lambda s: {
        "checkpoint": "after_capability",
        "session_id": s.get("session_id"),
        "memory_context": (s.get("memory_context") or "")[:500],
        "tool_result": (s.get("tool_result") or "")[:500],
    }
)

human_checkpoint_after_response = build_human_checkpoint(
    lambda s: {
        "checkpoint": "after_response",
        "session_id": s.get("session_id"),
        # 预览最近助手正文（messages；字段名沿用便于调试面板）
        "final_answer_preview": last_assistant_text(s.get("messages"))[:800],
    }
)
