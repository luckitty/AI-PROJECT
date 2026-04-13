def route_by_plan(state):
    """
    根据 planner 决策选择下一步执行哪个节点
    """
    if state.get("need_memory"):
        return "memory"

    if state.get("need_rag"):
        return "rag"

    if state.get("need_tool"):
        return "tool"

    return "response"