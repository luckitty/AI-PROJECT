"""
旅游攻略工具：调用 rag 层旅游缓存检索 + 本地配图 OCR，组装返回给模型。
检索逻辑见 rag/travel_cache_retriever.py。
"""
import json
from typing import List

from langchain.tools import tool

from rag.travel_cache_retriever import retrieve_travel_docs
from tools.travel_itinerary_builder import (
    build_itinerary_format_instruction,
    build_llm_itinerary_bundle,
)

# 普通旅游问题需要足够的素材覆盖景点集合；4 条容易漏掉城市核心景点。
TOP_K_NOTES = 10

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

    parts = [f"【标题】{title}"]
    # if url:
    #     parts.append(f"【链接】{url}")

    prebuilt_ocr_text = (note.get("ocr_text") or "").strip()
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


def build_travel_material(query: str, docs) -> str:
    """
    统一组装“结构约束+风格建议+素材明细”，作为旅游路线生成唯一输入，
    让结构规则与文风建议分层，避免多处拼接导致约束漂移。
    """
    # 构建行程格式要求 规则
    format_instruction = build_itinerary_format_instruction(query)
    # 单次遍历构建素材明细，避免重复遍历 docs。
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

    return (
        format_instruction
        + "\n\n【风格建议】\n"
        "- 优先使用下面的事实信息，不够再做合理补充，但不要偏离用户问题。\n"
        "- 语言可以稍微幽默风趣一点，不要过于正式。\n"
        "- 回答时可以适当使用表情或图标，提升可读性。\n"
        "- 表达形式可以多元化，但不要破坏上方结构约束。\n\n"
        + "【素材明细（优先使用）】\n"
        + "\n\n".join(blocks)
    )


@tool
def search_travel(query: str, rag_context: str) -> str:
    """旅游行程专用工具：从本地 data/cache 语义检索笔记，组装规则与素材供最终流式回答使用。"""
    docs = rag_context
    print("search_travel 命中文档数:", len(docs))

    travel_material = build_travel_material(query, docs)
    # 单次调用大模型同时生成两份结果：
    # - visible_answer：给用户看的攻略草稿
    # - itinerary_structured：给高德路线补交通用的隐藏结构
    itinerary_bundle = build_llm_itinerary_bundle(docs, query)
    skeleton_json = str(itinerary_bundle.get("itinerary_structured") or "").strip()
    visible_answer_draft = str(itinerary_bundle.get("visible_answer") or "").strip()
    print("search_travel===========skeleton_json \n", skeleton_json, "\n")
    travel_output_payload = {
        "itinerary_structured": skeleton_json,
        "visible_answer_draft": visible_answer_draft,
        # 给 response_node 的最终旅游攻略素材：规则 + RAG 摘要 + 素材明细。
        "response_material": travel_material,
    }
    return json.dumps(travel_output_payload, ensure_ascii=False)
