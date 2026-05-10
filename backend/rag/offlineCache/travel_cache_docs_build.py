"""
旅游缓存笔记 → LangChain Document 的离线构建链路。

含 OCR 缓存读写、规则/大模型去噪、攻略正文缓存、结构化画像与向量正文长度裁剪。
在线加载入口仍通过 rag.travel_loader.load_travel_cache_docs，仅把「每条笔记的重处理」集中在本模块。
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from rag.offlineCache.travel_ocr import get_or_build_note_ocr_text, normalize_text
from rag.offlineCache.travel_text_denoise import denoise_xhs_note_text
from rag.offlineCache.travel_llm_guide_denoise import (
    resolve_guide_body_for_note,
    save_llm_guide_cache,
)
from rag.offlineCache.travel_profile_extractor import build_structured_profile

# 智谱 embedding-3 单条 input 上限约 3072 tokens；中文整篇笔记拼接后易超上限，会返回 1210「参数有误」。
# 仅截断写入向量库的 page_content；metadata 仍保留 desc/ocr 全文，检索命中后攻略由 search_travel 用 metadata 拼装。
TRAVEL_EMBED_PAGE_MAX_CHARS = 2000


def build_travel_cache_documents(
    merged: dict,
    data_root: Path,
    city_name: str | None,
    ocr_cache: dict,
    guide_cache: dict,
    guide_cache_path: Path,
    *,
    allow_runtime_ocr: bool,
    use_llm_guide_denoise: bool,
    force_refresh_llm_guide: bool,
) -> tuple[list[Document], bool]:
    """
    对已合并的 note_id → 笔记字典 逐条做 OCR/攻略缓存、去噪、结构化画像与向量正文裁剪。

    返回 (documents, ocr_cache_dirty)；后者为 True 时表示 OCR 内存缓存有变更，调用方需 save_ocr_cache。
    LLM 攻略缓存仅在单条成功时由本函数内部立即 save_llm_guide_cache，避免批量中断丢缓存。
    """
    docs: list[Document] = []
    backend_root = data_root.resolve().parent
    cache_updated = False

    for note in merged.values():
        # 标题与正文先做 normalize，再去小红书噪声；OCR 整段入库不做字符数截断。
        title = denoise_xhs_note_text(normalize_text(note.get("title")))
        desc = denoise_xhs_note_text(normalize_text(note.get("desc")))
        ocr_text, changed = get_or_build_note_ocr_text(
            note,
            backend_root,
            ocr_cache,
            allow_runtime_ocr=allow_runtime_ocr,
        )
        if changed:
            cache_updated = True
        ocr_text_for_embed = denoise_xhs_note_text(normalize_text(ocr_text))
        inferred_city = normalize_text(note.get("city") or city_name)
        guide_body, guide_changed = resolve_guide_body_for_note(
            str(note.get("note_id") or ""),
            title,
            desc,
            ocr_text_for_embed,
            guide_cache,
            invoke_llm_on_miss=use_llm_guide_denoise,
            force_refresh=force_refresh_llm_guide,
        )
        if guide_changed:
            # 每条 LLM 成功写回后立即落盘，避免整城批量耗时长、进程中断时缓存全丢。
            save_llm_guide_cache(guide_cache_path, guide_cache)
        structured_profile = build_structured_profile(
            city_name=inferred_city,
            title=title,
            desc=desc,
            ocr_text=ocr_text_for_embed,
            guide_body=guide_body if guide_body else None,
        )
        profile_text_for_embed = (
            f"城市:{structured_profile['city']}\n"
            f"景点:{' '.join(structured_profile['spots'])}\n"
            f"美食:{' '.join(structured_profile['foods'])}\n"
            f"交通:{' '.join(structured_profile['transport'])}\n"
            f"标签:{' '.join(structured_profile['tags'])}\n"
            f"旅行风格:{structured_profile['travel_style']}\n"
            f"预算:{structured_profile['budget_level']}\n"
            f"时长:{structured_profile['duration']}"
        ).strip()
        # 向量化预算内优先保留标题、正文与结构化摘要；有大模型攻略正文时不再拼接 OCR 以免重复。
        if guide_body:
            head_block = f"{title}\n{guide_body}\n{profile_text_for_embed}".strip()
        else:
            head_block = f"{title}\n{desc}\n{profile_text_for_embed}".strip()
        budget = TRAVEL_EMBED_PAGE_MAX_CHARS
        if len(head_block) >= budget:
            page_content = head_block[:budget]
        else:
            room = budget - len(head_block) - 1
            if guide_body:
                page_content = head_block.strip()[:budget]
            else:
                ocr_head = (ocr_text_for_embed or "")[: max(0, room)]
                page_content = f"{head_block}\n{ocr_head}".strip()[:budget]
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source_type": "travel_cache",
                    "note_id": note.get("note_id"),
                    "note_url": note.get("note_url"),
                    "title": title,
                    "desc": desc,
                    # 与向量正文一致：存去噪后全文，便于攻略素材与调试一致。
                    "ocr_text": ocr_text_for_embed,
                    # 大模型合并去噪后的纯攻略正文；空字符串表示未做或未命中缓存。
                    "guide_body": guide_body,
                    "city": structured_profile["city"],
                    "spots_text": " ".join(structured_profile["spots"]),
                    "foods_text": " ".join(structured_profile["foods"]),
                    "transport_text": " ".join(structured_profile["transport"]),
                    "duration": structured_profile["duration"],
                    "budget_level": structured_profile["budget_level"],
                    "tags_text": " ".join(structured_profile["tags"]),
                    "travel_style": structured_profile["travel_style"],
                    "raw_summary": structured_profile["raw_summary"],
                    "itinerary_json": json.dumps(structured_profile["itinerary"], ensure_ascii=False),
                    "structured_profile_json": json.dumps(structured_profile, ensure_ascii=False),
                    # Milvus metadata 字段不支持 list/dict，这里序列化成 JSON 字符串，使用侧再反序列化。
                    "feed_images_json": json.dumps(note.get("feed_images") or [], ensure_ascii=False),
                },
            )
        )

    return docs, cache_updated
