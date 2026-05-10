from pathlib import Path
import json

from rag.offlineCache.travel_ocr import load_ocr_cache, save_ocr_cache
from rag.offlineCache.travel_llm_guide_denoise import load_llm_guide_cache
from rag.offlineCache.travel_cache_docs_build import build_travel_cache_documents

# 对外加载接口固定「只读 OCR 缓存、不强制刷新大模型攻略缓存」。
# 离线脚本 tools/offlineTool/rebuild_travel_ocr_and_milvus.py 在调用 load_travel_cache_docs 前按需改为 True，用完再还原。
loader_allow_runtime_ocr = False
loader_force_refresh_llm_guide = False


def load_travel_cache_docs(
    data_path="data",
    city_name=None,
    use_llm_guide_denoise=False,
):
    """
    从 data/cache 下读取旅游笔记缓存，按 note_id 建索引并生成 Document（整篇不切分）。
    约定缓存内 note_id 唯一；若未按 City 过滤而扫多文件，同名后者覆盖。
    city_name 传值时只读取对应城市文件（如 北京 -> 北京.json），用于缩小检索域。
    OCR 仅使用 data/cache_ocr_text.json 已有条目；实时识别由离线重建脚本通过 loader_allow_runtime_ocr 打开。

    use_llm_guide_denoise：为 True 时，对缓存未命中的笔记调用大模型生成「纯攻略正文」并写入
    data/cache_llm_guide_body.json；为 False 时仍会从该文件读取已有结果（离线跑过后在线自动生效）。
    强制忽略已有大模型条目由 loader_force_refresh_llm_guide 控制（仅重建脚本使用）。
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

    print("travel_loader===========target_files \n", target_files)

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
            merged[note_id] = item

    ocr_cache_path = data_root / "cache_ocr_text.json"
    ocr_cache = load_ocr_cache(ocr_cache_path)

    guide_cache_path = data_root / "cache_llm_guide_body.json"
    guide_cache = load_llm_guide_cache(guide_cache_path)

    docs, cache_updated = build_travel_cache_documents(
        merged,
        data_root,
        city_name,
        ocr_cache,
        guide_cache,
        guide_cache_path,
        allow_runtime_ocr=loader_allow_runtime_ocr,
        use_llm_guide_denoise=use_llm_guide_denoise,
        force_refresh_llm_guide=loader_force_refresh_llm_guide,
    )
    if cache_updated:
        save_ocr_cache(ocr_cache_path, ocr_cache)
    return docs
