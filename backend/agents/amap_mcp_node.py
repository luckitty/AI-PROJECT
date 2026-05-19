"""
高德 MCP 图节点：planner 判定 need_amap_mcp 时进入，选路与执行委托 mcp_servers。
"""

from graph.chat_messages import format_conversation_for_prompt
from mcp_servers.amap_mcp_registry import amap_mcp_executor
from tools.travel_personal_map_builder import invoke_personal_map_from_guide


def amap_mcp_node(state):
    """
    高德 MCP 图节点：结果写入 tool_result 供 response 使用。
    若用户已确认导入上一份攻略，则直接走专属地图构建，不再经通用工具选择器。
    """
    query = state["query"]

    if state.get("confirm_amap_personal_map"):
        guide_text = (state.get("last_travel_guide_for_map") or "").strip()
        result = invoke_personal_map_from_guide(guide_text, query)
        return {
            **state,
            "tool_result": result,
            "confirm_amap_personal_map": False,
            "pending_amap_personal_map_offer": False,
            "amap_personal_map_ready": True,
        }

    conversation_history = format_conversation_for_prompt(
        state.get("messages"),
        state.get("conversation_summary"),
        max_chars=2600,
        max_messages=10,
    ).strip()

    result = amap_mcp_executor.run(
        query,
        conversation_history=conversation_history if conversation_history else None,
    )

    return {
        **state,
        "tool_result": result,
    }
