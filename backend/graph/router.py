def route_from_planner(state):
    """
    根据 planner 决策进行首跳路由：
    仅在这里选择 rag / memory / amap_mcp / tool / response。

    当 need_rag 与 need_memory 同时为真时，先走 memory 拉用户长期记忆，再由图编排进入 rag，
    避免仅召回笔记素材却看不到用户偏好。
    """
    if state.get("need_rag") and state.get("need_memory"):
        return "memory"
    if state.get("need_rag"):
        return "rag"

    if state.get("need_memory"):
        return "memory"

    if state.get("need_amap_mcp"):
        return "amap_mcp"

    if state.get("need_tool"):
        return "tool"

    return "response"