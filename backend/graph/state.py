from typing import TypedDict, Optional

class AgentState(TypedDict):
    query: str
    user_id: str
    session_id: str
    system_prompt: str

    need_rag: bool
    need_tool: bool
    need_memory: bool

    rag_context: Optional[str]
    travel_context: Optional[str]
    tool_result: Optional[str]
    memory_context: Optional[str]

    final_answer: Optional[str]


def build_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    system_prompt: str,
) -> AgentState:
    """
    LangGraph invoke 的完整初始状态；planner 会覆盖 need_*，各子节点再填上下文。
    """
    return {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        # 保留与 create_assistant 一致的系统提示词，让 response 节点统一遵循。
        "system_prompt": system_prompt,

        "need_rag": False,
        "need_tool": False,
        "need_memory": False,
        
        "rag_context": None,
        "travel_context": None,
        "tool_result": None,
        "memory_context": None,
        "final_answer": None,
    }