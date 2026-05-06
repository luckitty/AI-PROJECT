"""
旅游攻略工具：调用 rag 层旅游缓存检索 + 本地配图 OCR，组装返回给模型。
检索逻辑见 rag/travel_cache_retriever.py。
"""
import json
import time
from typing import List

from tools.travel_itinerary_builder import build_llm_itinerary_bundle

# 用于从正文/OCR 里截取「和吃喝更相关」的一小段，供美食类紧凑摘要使用（非穷举，覆盖常见口语）。
FOOD_SNIPPET_HINTS = (
    "美食",
    "小吃",
    "餐厅",
    "饭馆",
    "火锅",
    "烤鸭",
    "奶茶",
    "咖啡",
    "早茶",
    "早餐",
    "午餐",
    "晚餐",
    "夜宵",
    "必吃",
    "好吃",
    "铜锅",
    "涮肉",
    "豆汁",
    "米其林",
)

# 单条笔记写入攻略 prompt 的长度上限：过长会拉高输入 token 与 get_docs 侧处理成本。
NOTE_DESC_MAX_CHARS = 1400
NOTE_OCR_MAX_CHARS = 600


def pick_food_related_snippet(desc: str, ocr_text: str, max_len: int = 220) -> str:
    """
    从正文 desc 与配图 OCR 中优先截取包含餐饮线索的片段；若无命中则退回正文前若干字。
    """
    blob = (desc or "").strip().replace("\n", " ")
    for hint in FOOD_SNIPPET_HINTS:
        if hint in blob:
            idx = blob.index(hint)
            start = max(0, idx - 36)
            end = min(len(blob), start + max_len)
            piece = blob[start:end]
            return piece + ("..." if end < len(blob) else "")
    ocr = (ocr_text or "").strip().replace("\n", " ")
    for hint in FOOD_SNIPPET_HINTS:
        if hint in ocr:
            idx = ocr.index(hint)
            start = max(0, idx - 24)
            end = min(len(ocr), start + min(max_len, 180))
            piece = ocr[start:end]
            return piece + ("..." if end < len(ocr) else "")
    if blob:
        return blob[:max_len] + ("..." if len(blob) > max_len else "")
    if ocr:
        return ocr[:max_len] + ("..." if len(ocr) > max_len else "")
    return "（暂无正文与配图文字摘要）"

def build_note_block(note: dict) -> str:
    """
    组装单条笔记输出：标题、链接、正文 desc、配图 OCR 汇总。
    """
    title = (note.get("title") or "").strip() or "(无标题)"
    # url = (note.get("note_url") or "").strip()
    desc = (note.get("desc") or "").strip()
    if len(desc) > NOTE_DESC_MAX_CHARS:
        desc = desc[:NOTE_DESC_MAX_CHARS] + "..."

    parts = [f"【标题】{title}"]
    # if url:
    #     parts.append(f"【链接】{url}")

    prebuilt_ocr_text = (note.get("ocr_text") or "").strip()
    if len(prebuilt_ocr_text) > NOTE_OCR_MAX_CHARS:
        prebuilt_ocr_text = prebuilt_ocr_text[:NOTE_OCR_MAX_CHARS] + "..."
    ocr_chunks: List[str] = [prebuilt_ocr_text] if prebuilt_ocr_text else []
    # 按用户要求融合素材：
    # - OCR 成功：desc + OCR 合并，给模型更多可用事实
    # - OCR 失败：仅使用 desc，避免空 OCR 噪声影响路线生成
    merged_content_parts: List[str] = []
    if desc:
        merged_content_parts.append(f"【正文 desc】\n{desc}")
    if ocr_chunks:
        merged_content_parts.append("【配图 OCR 文字】\n" + "\n\n".join(ocr_chunks))
    if not merged_content_parts:
        merged_content_parts.append("【正文素材】（该条暂无可用正文与 OCR 文字）")
    parts.append("\n\n".join(merged_content_parts))

    return "\n\n".join(parts)


def build_travel_body_material(docs) -> str:
    """
    组装检索素材正文（笔记 desc + OCR 等），不含指令；行程规则全部由 travel_itinerary_builder 内唯一指令承担。
    """
    blocks: List[str] = []
    for rank, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        title = (metadata.get("title") or "").strip() or "(无标题)"
        desc = (metadata.get("desc") or "").strip().replace("\n", " ")
        note_id = str(metadata.get("note_id") or "")
        note = {
            "note_id": note_id,
            "title": title,
            "desc": desc,
            "note_url": metadata.get("note_url"),
            "ocr_text": metadata.get("ocr_text"),
        }
        note_block = build_note_block(note)

        blocks.append(f"========== 结果 {rank}（相似度排序） ==========\n{note_block}")

    return "【素材明细】\n" + "\n\n".join(blocks)


def compose_travel_tool_payload(query: str, docs: list, stream_writer=None) -> dict:
    """
    旅游工具出口：攻略仅在一次 LLM 调用中生成（见 build_llm_itinerary_bundle），
    此处只序列化正文与流式标记，供 response 节点拼最终 state，不再触发第二遍模型。
    """
    material_started_at = time.perf_counter()
    body_material = build_travel_body_material(docs)
    material_seconds = time.perf_counter() - material_started_at
    bundle_started_at = time.perf_counter()
    itinerary_bundle = build_llm_itinerary_bundle(
        query, body_material, stream_writer=stream_writer
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


def search_travel(query: str, rag_context: list, stream_writer=None) -> str:
    """
    旅游行程专用入口：接收 rag 文档列表，拼接检索素材并单次大模型流式/非流式生成攻略正文。
    说明：仅由 LangGraph 的 travel 节点直接调用；LangChain 的 StructuredTool.invoke 第二个参数是 RunnableConfig，
    若误写成 invoke(query, docs) 会把文档列表当成 config 解析，触发 AttributeError，表现为流式接口长时间无输出或报错。
    """
    docs = rag_context or []
    print("search_travel 命中文档数:", len(docs))

    search_started_at = time.perf_counter()
    travel_output_payload = compose_travel_tool_payload(query, docs, stream_writer)
    search_seconds = time.perf_counter() - search_started_at
    print(f"pipeline_timing===========search_travel 整段耗时: {search_seconds:.2f}s")
    draft = str(travel_output_payload.get("visible_answer_draft") or "")
    print(
        "search_travel===========visible_draft_len \n",
        len(draft),
        "\n",
    )
    return json.dumps(travel_output_payload, ensure_ascii=False)
