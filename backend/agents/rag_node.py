from rag.travel_cache_retriever import retrieve_travel_docs


def rag_node(state):
    """
    旅游垂类 RAG 节点：先从旅游缓存里取结构化素材摘要，再交给工具节点生成最终攻略。
    """
    query = state["query"]
    docs = retrieve_travel_docs(query, top_k=6)
    return {
        **state,
        "rag_context": docs,
    }

    lines = []
    # 这里保持摘要紧凑，避免把超长正文直接塞进上下文导致响应变慢。
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        city = str(metadata.get("city") or "").strip()
        title = str(metadata.get("title") or "").strip() or "(无标题)"
        spots = str(metadata.get("spots_text") or "").strip()
        foods = str(metadata.get("foods_text") or "").strip()
        summary = str(metadata.get("raw_summary") or "").strip()
        lines.append(
            f"- 结果{index} | 城市={city or '未知'} | 标题={title} | 景点={spots or '暂无'} | "
            f"美食={foods or '暂无'} | 摘要={summary or '暂无'}"
        )

    rag_travel_context = "旅游缓存检索摘要：\n" + "\n".join(lines)
    return {
        **state,
        "rag_context": rag_travel_context,
    }
