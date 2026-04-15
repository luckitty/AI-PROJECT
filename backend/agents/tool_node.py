from tools.tool_registry import tool_executor


def tool_node(state):
    # planner 已决定需要工具时才会进入此节点，这里只负责执行并回填结果。
    query = state["query"]

    result = tool_executor.run(query)
    print("tool_node===========result \n", result, "\n")

    return {
        **state,
        "tool_result": result
    }