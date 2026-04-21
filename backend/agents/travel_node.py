from tools.search_travel_tool import search_travel


def travel_node(state):
    query = state["query"]

    # search_travel 工具内部已经完成“检索+路线初稿生成”，节点层只做状态回填。
    travel_draft = search_travel.invoke({"query": query})

    return {
        **state,
        "travel_context": travel_draft,
    }