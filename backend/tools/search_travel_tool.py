"""
旅游攻略工具：调用 rag 层旅游缓存检索，按离线写入的 guide_body 组装素材返回给模型。
无 guide_body 的条目不参与拼装（默认建库侧已保证有条可用攻略正文）。
检索逻辑见 rag/travel_cache_retriever.py。
"""
import json
import time
from typing import List

from rag.offlineCache.travel_text_denoise import denoise_xhs_note_text
from tools.travel_itinerary_builder import build_llm_itinerary_bundle


def build_note_block(title: str, guide_body: str) -> str:
    """
    组装单条笔记输出：标题 + 离线攻略正文 guide_body。
    全文不截断，统一经去噪（话题标签、URL 等）；调用方需保证 guide_body 非空。
    """
    clean_title = denoise_xhs_note_text(title.strip()) or "(无标题)"
    clean_body = denoise_xhs_note_text(guide_body.strip())
    parts = [f"【标题】{clean_title}", f"【攻略正文】\n{clean_body}"]
    return "\n\n".join(parts)


def build_travel_body_material(docs) -> str:
    """
    组装检索素材正文（仅含带 guide_body 的条目），不含指令；
    行程规则全部由 travel_itinerary_builder 内唯一指令承担。
    """
    blocks: List[str] = []
    rank = 0
    for doc in docs:
        metadata = doc.metadata or {}
        guide_raw = (metadata.get("guide_body") or "").strip()
        guide_body = denoise_xhs_note_text(guide_raw)
        if not guide_body:
            continue
        rank += 1
        title = (metadata.get("title") or "").strip() or "(无标题)"
        note_block = build_note_block(title, guide_body)
        blocks.append(f"========== 结果 {rank}（相似度排序） ==========\n{note_block}")

    return "【素材明细】\n" + "\n\n".join(blocks)


def compose_travel_tool_payload(
    query: str,
    docs: list,
    stream_writer=None,
    conversation_history: str | None = None,
) -> dict:
    """
    旅游工具出口：攻略仅在一次 LLM 调用中生成（见 build_llm_itinerary_bundle），
    此处只序列化正文与流式标记，供 response 节点解析后追加 AIMessage，不再触发第二遍模型。
    """
    material_started_at = time.perf_counter()
    body_material = build_travel_body_material(docs)
    print("body_material===========body_material \n", body_material[:1000], "\n")
    material_seconds = time.perf_counter() - material_started_at
    bundle_started_at = time.perf_counter()

    itinerary_bundle = build_llm_itinerary_bundle(
        query,
        body_material,
        stream_writer=stream_writer,
        conversation_history=conversation_history,
    )

    bundle_seconds = time.perf_counter() - bundle_started_at
    print(
        "pipeline_timing===========compose_travel_tool_payload "
        f"组装素材耗时: {material_seconds:.2f}s 攻略bundle(含LLM): {bundle_seconds:.2f}s"
    )

    visible_text = str(itinerary_bundle.get("visible_answer") or "").strip()
    # 流式接口下 travel 节点已把正文增量推到 SSE；response 在未流式时补推全文。
    visible_guide_streamed = stream_writer is not None and bool(visible_text)
    return {
        "visible_answer_draft": visible_text,
        "visible_guide_streamed": visible_guide_streamed,
    }


def search_travel(
    query: str,
    rag_context: list,
    stream_writer=None,
    conversation_history: str | None = None,
) -> str:
    """
    旅游行程专用入口：接收 rag 文档列表，拼接检索素材并单次大模型流式/非流式生成攻略正文。
    由 LangGraph 的 response 节点直接函数调用；LangChain 的 StructuredTool.invoke 第二个参数是 RunnableConfig，
    若误写成 invoke(query, docs) 会把文档列表当成 config 解析，触发 AttributeError。
    """
    docs = rag_context or []
    print("search_travel 命中文档数:", len(docs))

    search_started_at = time.perf_counter()

    travel_output_payload = compose_travel_tool_payload(
        query, docs, stream_writer, conversation_history=conversation_history
    )

    search_seconds = time.perf_counter() - search_started_at
    print(f"\n pipeline_timing===========search_travel 整段耗时: {search_seconds:.2f}s \n")

    draft = str(travel_output_payload.get("visible_answer_draft") or "")
    print(
        "search_travel===========visible_draft_len \n",
        len(draft),
        "\n",
    )
    return json.dumps(travel_output_payload, ensure_ascii=False)
