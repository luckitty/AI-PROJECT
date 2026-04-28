from tools.search_travel_tool import search_travel


def travel_node(state):
    query = state["query"]
    rag_context = state["rag_context"]

    # search_travel 工具内部已经完成“检索+路线初稿生成”，节点层只做状态回填。
    rag_context = search_travel.invoke(query, rag_context)

    return {
        **state,
        "travel_context": rag_context,
    }