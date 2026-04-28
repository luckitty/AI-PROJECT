from pathlib import Path
import json

from langchain_core.documents import Document

from rag.travel_ocr import (
    MAX_OCR_CHARS_FOR_EMBED,
    get_or_build_note_ocr_text,
    load_ocr_cache,
    normalize_text,
    save_ocr_cache,
)
from rag.travel_profile_extractor import build_structured_profile, extract_spots


def load_travel_cache_docs(data_path="data", city_name=None, allow_runtime_ocr=False):
    """
    从 data/cache 下读取旅游笔记缓存，按 note_id 去重并生成 Document（整篇不切分）。
    city_name 传值时只读取对应城市文件（如 北京 -> 北京.json），用于缩小检索域。
    allow_runtime_ocr 控制是否在查询阶段补跑 OCR：
    - False：只复用已有 OCR 缓存，不做实时识别（默认，保证查询低时延）
    - True：缓存缺失时允许识别图片并回写缓存（适合离线预处理）
    """
    data_root = Path(data_path)
    cache_dir = data_root / "cache"
    if not cache_dir.is_dir():
        return []

    target_files = sorted(cache_dir.glob("*.json"))
    if city_name:
        city_file = cache_dir / f"{city_name}.json"
        if not city_file.is_file():
            return []
        target_files = [city_file]

    merged = {}
    for path in target_files:
        try:
            with open(path, "r", encoding="utf-8") as file:
                records = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, dict):
                continue
            note_id = str(item.get("note_id", "")).strip()
            if not note_id:
                continue
            # 记录来源城市：优先使用文件名（如 北京.json），便于后续城市检索提权。
            item_city = str(path.stem).strip()
            if item_city:
                item["city"] = item_city
            prev = merged.get(note_id)
            if prev is None:
                merged[note_id] = item
                continue
            has_images = bool(item.get("feed_images"))
            prev_has_images = bool(prev.get("feed_images"))
            if has_images and not prev_has_images:
                merged[note_id] = item

    ocr_cache_path = data_root / "cache_ocr_text.json"
    ocr_cache = load_ocr_cache(ocr_cache_path)
    cache_updated = False

    docs = []
    backend_root = data_root.resolve().parent
    for note in merged.values():
        title = normalize_text(note.get("title"))
        desc = normalize_text(note.get("desc"))
        ocr_text, changed = get_or_build_note_ocr_text(
            note,
            backend_root,
            ocr_cache,
            allow_runtime_ocr=allow_runtime_ocr,
        )
        if changed:
            cache_updated = True
        ocr_text_for_embed = normalize_text(ocr_text)[:MAX_OCR_CHARS_FOR_EMBED]
        inferred_city = normalize_text(note.get("city") or city_name)
        structured_profile = build_structured_profile(
            city_name=inferred_city,
            title=title,
            desc=desc,
            ocr_text=ocr_text,
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
        page_content = f"{title}\n{desc}\n{ocr_text_for_embed}\n{profile_text_for_embed}".strip()
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source_type": "travel_cache",
                    "note_id": note.get("note_id"),
                    "note_url": note.get("note_url"),
                    "title": title,
                    "desc": desc,
                    "ocr_text": ocr_text,
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
    if cache_updated:
        save_ocr_cache(ocr_cache_path, ocr_cache)
    return docs
