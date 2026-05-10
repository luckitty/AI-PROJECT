def route_from_planner(state):
    """
    根据 planner 决策进行首跳路由：
    仅在这里选择 rag / memory / tool / response。
    """
    # 旅游六城问题优先走 rag，召回素材由 response 内 search_travel 一次性生成攻略。
    if state.get("need_rag"):
        return "rag"

    if state.get("need_memory"):
        return "memory"

    if state.get("need_tool"):
        return "tool"

    return "response"