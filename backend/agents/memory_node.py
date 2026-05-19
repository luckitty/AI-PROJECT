from memory.long_memory import search_long_memory


def memory_node(state):
    """
    按当前用户句语义检索长期记忆，并把命中正文拼成一段字符串写入 state，
    供 response 节点在「用户记忆」块中直接使用。
    """
    user_id = state["user_id"]
    query = state["query"]

    docs = search_long_memory(query, user_id)
    memory_texts = [doc.page_content for doc in docs if (doc.page_content or "").strip()]
    memory_context = "\n".join(memory_texts)

    return {
        **state,
        "memory_context": memory_context,
    }