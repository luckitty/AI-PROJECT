"""
旅游攻略工具：调用 rag 层旅游缓存检索 + 本地配图 OCR，组装返回给模型。
检索逻辑见 rag/travel_cache_retriever.py。
"""
import json
from typing import List

from langchain.tools import tool

from rag.travel_cache_retriever import is_food_focus_query, retrieve_travel_docs
from tools.travel_itinerary_builder import (
    build_itinerary_format_instruction,
    generate_travel_draft,
)

TOP_K_NOTES = 4
# 美食类问题需要覆盖同城多条笔记；具体上限仍受 retrieve_travel_docs 与数据量约束。
TOP_K_FOOD = 10

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


def build_compact_material_summary(docs, query: str = "") -> str:
    """
    把检索文档压成紧凑摘要，确保模型在长文本被截断时仍能拿到核心事实。
    美食类问题时优先展示结构化餐饮字段（foods_text、标签）与正文/OCR 中与吃喝相关的片段，避免只显示标题+泛化攻略前几句。
    """
    food_focus = is_food_focus_query((query or "").strip())
    lines: List[str] = []
    for rank, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        title = (metadata.get("title") or "").strip() or "(无标题)"
        desc = (metadata.get("desc") or "").strip().replace("\n", " ")
        note_id = str(metadata.get("note_id") or "")
        if not food_focus:
            short_desc = desc[:180] + ("..." if len(desc) > 180 else "")
            lines.append(f"- 结果{rank} | note_id={note_id} | 标题={title} | 摘要={short_desc}")
            continue
        foods_text = (metadata.get("foods_text") or "").strip()
        tags_text = (metadata.get("tags_text") or "").strip()
        ocr_text = (metadata.get("ocr_text") or "").strip()
        foods_short = (
            foods_text[:100] + ("..." if len(foods_text) > 100 else "")
            if foods_text
            else "（建库未抽到餐饮关键词，请看下方正文/配图摘要）"
        )
        tags_short = tags_text[:72] + ("..." if len(tags_text) > 72 else "") if tags_text else ""
        snippet = pick_food_related_snippet(desc, ocr_text, max_len=220)
        extra_tags = f" | 标签={tags_short}" if tags_short else ""
        lines.append(
            f"- 结果{rank} | note_id={note_id} | 标题={title} | 餐饮关键词={foods_short}{extra_tags}"
            f" | 正文或配图摘要={snippet}"
        )
    return "\n".join(lines)


def build_travel_material(query: str, docs) -> str:
    """
    统一组装“规则+摘要+素材明细”，作为旅游路线生成唯一输入，避免多处拼接导致约束漂移。
    """
    # 构建行程格式要求 规则
    format_instruction = build_itinerary_format_instruction(query)
    # 构建素材紧凑摘要
    compact_summary = build_compact_material_summary(docs, query)
    # 构建素材明细
    blocks = []
    for rank, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        note = {
            "note_id": metadata.get("note_id"),
            "note_url": metadata.get("note_url"),
            "title": metadata.get("title"),
            "desc": metadata.get("desc"),
            "ocr_text": metadata.get("ocr_text"),
        }
        try:
            note_block = build_note_block(note)
        except Exception as error:
            # 单条素材构建失败时降级为简版，避免整次工具返回失败。
            note_block = (
                f"【标题】{(note.get('title') or '').strip() or '(无标题)'}\n\n"
                f"【正文 desc】\n{(note.get('desc') or '').strip()}\n\n"
                f"【配图 OCR 文字】（该条素材处理失败：{error}）"
            )
        blocks.append(f"========== 结果 {rank}（相似度排序） ==========\n{note_block}")

    return (
        format_instruction
        + "\n\n【执行提醒】\n"
        "- 优先使用下面的参考材料事实，不够再做合理补充，但不要偏离用户问题。\n"
        "- 语言可以稍微幽默风趣一点，不要过于正式。\n"
        "- 回答的时候可以适当使用表情或图标，提升可读性。\n"
        "- 回答格式可以多元化，但不要破坏规则要求的结构。\n\n"
        + "【素材紧凑摘要（优先使用）】但不要出现“素材”、“参考材料”等字眼\n"
        + compact_summary
        + "\n\n"
        + "\n\n".join(blocks)
    )


@tool
def search_travel(query: str) -> str:
    """旅游行程专用工具：从本地 data/cache 语义检索笔记，组装规则与素材并直接生成路线初稿。"""
    q = (query or "").strip()
    if not q:
        return "请提供具体的旅游攻略检索问题。"

    top_k = TOP_K_FOOD if is_food_focus_query(q) else TOP_K_NOTES
    docs = retrieve_travel_docs(q, top_k=top_k)
    print("search_travel 命中文档数:", len(docs))
    if not docs:
        return "本地暂无缓存笔记（data/cache 为空或无法读取），请先通过其它方式导入缓存数据。"

    travel_material = build_travel_material(q, docs)
    # 工具层直接产出路线初稿，节点层只做编排，避免规则模板泄漏到 response 节点日志与用户输出。
    travel_draft = generate_travel_draft(q, travel_material, docs)
    travel_output_payload = {
        # 先产出结构化路线草稿，供 response 节点作为最终攻略骨架。
        "itinerary_structured": travel_draft or "",
        # 给 response_node 的最终旅游攻略素材：规则 + RAG 摘要 + 素材明细。
        "response_material": travel_material,
    }
    return json.dumps(travel_output_payload, ensure_ascii=False)
