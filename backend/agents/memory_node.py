from memory.long_memory import search_long_memory


def memory_node(state):
    user_id = state["user_id"]
    query = state["query"]

    memory = search_long_memory(query, user_id)

    return {
        **state,
        "memory_context": memory
    }