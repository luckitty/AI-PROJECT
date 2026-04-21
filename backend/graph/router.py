def route_from_planner(state):
    """
    根据 planner 决策进行首跳路由：
    仅在这里选择 rag / memory / tool / response。
    """
    # 旅游问题优先走 rag，再由 rag 后路由决定是否进入 travel。
    if state.get("need_rag"):
        return "rag"

    if state.get("need_memory"):
        return "memory"

    if state.get("need_tool"):
        return "tool"

    return "response"