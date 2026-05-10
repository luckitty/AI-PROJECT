from typing import Annotated, TypedDict, Optional, Any, List, NotRequired

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str
    user_id: str
    session_id: str
    system_prompt: str

    # ChatGPT 式多轮：user/assistant 交替由 add_messages 追加；短期记忆以此为准。
    messages: Annotated[list[AnyMessage], add_messages]

    # 超出滑动窗口的早期轮次经 LLM 压入此字段；checkpoint 内 messages 同时被 RemoveMessage 裁剪以控制体积。
    conversation_summary: NotRequired[str]

    need_rag: bool
    need_tool: bool
    need_memory: bool

    # planner 判定终稿是否需要带多轮历史；false 时 response 只喂当前句（仍可有记忆/工具结果）。
    need_history: bool

    # 模型判定为旅游攻略意图，但问题未命中允许走本地旅游 RAG 的六个城市时，由 response 用攻略专用指令单刷 LLM。
    travel_itinerary_in_response: bool

    # True 表示当前请求走 SSE stream，travel 节点可把攻略正文推到 custom；invoke 短路时为 False。
    stream_sink_active: bool

    # rag 节点写入 Document 列表；此前误标成 str 易误导调用方。
    rag_context: Optional[List[Any]]
    tool_result: Optional[str]
    memory_context: Optional[str]

    # True：用户在人机断点选择终止，或节点内因 interrupt_manager.stop（断连/stop）协作结束本轮。
    human_halt: bool


def build_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    system_prompt: str,
    stream_sink_active: bool = False,
) -> AgentState:
    """
    LangGraph invoke 的初始补丁：只写字段会被本轮覆盖的值。
    每轮追加一条用户消息，由 add_messages 与 checkpoint 里已有 messages 合并（ChatGPT 式会话列表）。
    """
    return {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        # 保留与 create_assistant 一致的系统提示词，让 response 节点统一遵循。
        "system_prompt": system_prompt,

        "messages": [HumanMessage(content=query)],

        "need_rag": False,
        "need_tool": False,
        "need_memory": False,

        "need_history": True,

        "travel_itinerary_in_response": False,

        "stream_sink_active": stream_sink_active,

        "rag_context": None,
        "tool_result": None,
        "memory_context": None,
        "human_halt": False,
    }
