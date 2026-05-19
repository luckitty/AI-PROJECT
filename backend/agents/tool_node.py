from graph.chat_messages import format_conversation_for_prompt
from tools.tool_registry import tool_executor


def tool_node(state):
    # planner 已决定需要工具时才会进入此节点，这里只负责执行并回填结果。
    query = state["query"]
    # 多轮对话节选，供选工具与参数抽取（含前文景点与路线指代）。
    conversation_history = format_conversation_for_prompt(
        state.get("messages"),
        state.get("conversation_summary"),
        max_chars=2600,
        max_messages=10,
    ).strip()
    print("tool_node===========conversation_history", conversation_history, "\n")

    result = tool_executor.run(
        query,
        conversation_history=conversation_history if conversation_history else None,
    )

    return {
        **state,
        "tool_result": result
    }