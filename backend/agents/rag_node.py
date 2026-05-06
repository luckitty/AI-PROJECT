import time

from rag.travel_cache_retriever import retrieve_travel_docs


def rag_node(state):
    """
    旅游垂类 RAG 节点：先从旅游缓存里取结构化素材摘要，再交给工具节点生成最终攻略。
    """
    query = state["query"]
    rag_started_at = time.perf_counter()
    docs = retrieve_travel_docs(query, top_k=6)
    rag_seconds = time.perf_counter() - rag_started_at
    print(
        "pipeline_timing===========rag_node retrieve_travel_docs "
        f"耗时: {rag_seconds:.2f}s 命中文档数: {len(docs)}"
    )
    return {
        **state,
        "rag_context": docs,
    }