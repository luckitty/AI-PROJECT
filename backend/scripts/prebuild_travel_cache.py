#!/usr/bin/env python3
"""
离线预构建旅游缓存：对 data/cache 下各城市 JSON 配图跑 OCR、合并回 JSON、按城市写入 Milvus。

在线问答时 loader 使用 allow_runtime_ocr=False，仅复用本脚本产出的 ocr_text / cache_ocr_text.json，
避免每次检索都跑 OCR 与全量融合。

用法（在项目根目录 ai-project 下）::

    python backend/scripts/prebuild_travel_cache.py

可选：已跑过 OCR 时跳过识别；仅合并 JSON；仅写向量库（见 argparse）。
"""
import argparse
import json
import sys
from pathlib import Path

# 与 ``python main.py`` 从 backend 目录启动时一致：path 前部为「项目根」+「backend」，
# 以便 ``from backend.xxx`` 与 ``from rag.xxx`` 同时可用。
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
for path_entry in (str(BACKEND_ROOT), str(PROJECT_ROOT)):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from rag.travel_cache_retriever import (  # noqa: E402
    ensure_travel_vectorstore_by_city,
    list_available_cities,
)
from rag.travel_loader import load_ocr_cache, load_travel_cache_docs  # noqa: E402

DATA_PATH = str(BACKEND_ROOT / "data")
CACHE_DIR = BACKEND_ROOT / "data" / "cache"
OCR_CACHE_PATH = BACKEND_ROOT / "data" / "cache_ocr_text.json"


def merge_ocr_into_city_json_files() -> int:
    """
    将 cache_ocr_text.json 中每条笔记的 ocr_text 写回对应城市 JSON，
    便于单独拷贝数据目录、且加载时优先走 note['ocr_text'] 快速路径。
    返回写回过的文件数量。
    """
    if not OCR_CACHE_PATH.is_file():
        print("未找到 cache_ocr_text.json，跳过写回城市 JSON。")
        return 0
    ocr_cache = load_ocr_cache(OCR_CACHE_PATH)
    if not ocr_cache:
        print("OCR 缓存为空，跳过写回城市 JSON。")
        return 0
    written = 0
    for path in sorted(CACHE_DIR.glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
            records = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            print(f"跳过无法解析的文件 {path.name}: {error}")
            continue
        if not isinstance(records, list):
            continue
        changed = False
        for note in records:
            if not isinstance(note, dict):
                continue
            note_id = str(note.get("note_id") or "").strip()
            if not note_id:
                continue
            entry = ocr_cache.get(note_id)
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("ocr_text") or "").strip()
            if not text:
                continue
            if note.get("ocr_text") != text:
                note["ocr_text"] = text
                changed = True
        if changed:
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已写回 OCR 字段: {path.name}")
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="离线 OCR + 合并城市 JSON + Milvus 向量化")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="跳过 RapidOCR 步骤（已存在 cache_ocr_text.json 时使用）",
    )
    parser.add_argument(
        "--skip-merge-json",
        action="store_true",
        help="跳过将 OCR 写回各城市 JSON",
    )
    parser.add_argument(
        "--skip-milvus",
        action="store_true",
        help="跳过写入 Milvus（仅做 OCR 与/或合并 JSON）",
    )
    args = parser.parse_args()

    if not CACHE_DIR.is_dir():
        print(f"未找到缓存目录: {CACHE_DIR}")
        sys.exit(1)

    cities = list_available_cities()
    print(f"检测到城市缓存文件: {', '.join(cities) if cities else '（无）'}")

    if not args.skip_ocr:
        print("--- 步骤 1/3: 离线 OCR，更新 cache_ocr_text.json ---")
        docs = load_travel_cache_docs(DATA_PATH, city_name=None, allow_runtime_ocr=True)
        print(f"已加载笔记条数（含融合字段）: {len(docs)}")
    else:
        print("--- 步骤 1/3: 已跳过 OCR ---")

    if not args.skip_merge_json:
        print("--- 步骤 2/3: 将 OCR 合并回 data/cache/*.json ---")
        merge_ocr_into_city_json_files()
    else:
        print("--- 步骤 2/3: 已跳过写回城市 JSON ---")

    if not args.skip_milvus:
        print("--- 步骤 3/3: 按城市与全量写入 Milvus ---")
        for city in cities:
            print(f"  向量化: {city} ...")
            ensure_travel_vectorstore_by_city(city)
        print("  向量化: 全量（未命中城市时的降级集合）...")
        ensure_travel_vectorstore_by_city(None)
        print("Milvus 写入完成。")
    else:
        print("--- 步骤 3/3: 已跳过 Milvus ---")

    print("预构建流程结束。重启后端后，检索将直接使用已向量化的内容，无需在线 OCR。")


if __name__ == "__main__":
    main()
