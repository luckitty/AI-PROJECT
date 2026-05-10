import time

from langgraph.config import get_stream_writer

from rag.travel_cache_retriever import retrieve_travel_docs


def rag_node(state):
    """
    旅游垂类 RAG 节点：先从旅游缓存里取结构化素材摘要，再交给工具节点生成最终攻略。
    """
    query = state["query"]
    # 向量检索可能数十秒且无 token；流式下先推一行提示，避免用户误以为请求挂死。
    if state.get("stream_sink_active"):
        try:
            writer = get_stream_writer()
            writer({"content": "正在检索本地旅游笔记素材，请稍候…\n"})
        except Exception:
            pass
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