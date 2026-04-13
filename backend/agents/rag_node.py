from rag.retriever import retriever


def rag_node(state):
    query = state["query"]
    context = retriever(query)

    return {
        **state,
        "rag_context": context
    }