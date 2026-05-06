from langgraph.config import get_stream_writer

from tools.search_travel_tool import search_travel


def travel_node(state):
    query = state["query"]
    rag_context = state["rag_context"]

    # 仅 SSE stream 请求置 stream_sink_active：才把正文增量推到 custom，并与 response 附录衔接对齐。
    stream_writer = None
    if bool(state.get("stream_sink_active")):
        try:
            stream_writer = get_stream_writer()
        except Exception:
            stream_writer = None

    # search_travel：单次 LLM 生成攻略正文（可流式）；response_node 仅在未流式时补推同一正文。
    # 必须直接函数调用；禁止使用 tool.invoke(query, rag_context)：第二个参数会被当成 RunnableConfig。
    travel_context = search_travel(query, rag_context or [], stream_writer)

    return {
        **state,
        "travel_context": travel_context,
    }