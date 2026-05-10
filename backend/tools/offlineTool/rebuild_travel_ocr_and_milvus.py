"""
离线任务：六城旅游 RAG 白名单内各城市笔记重新 OCR，再按 travel_cache_retriever
既有「cache json 文件 mtime/size 签名」逻辑全量重建 Milvus 集合（含各 city key 与 all）。

用法（在 backend 目录下）:
  python3 tools/offlineTool/rebuild_travel_ocr_and_milvus.py

可选:
  --skip-ocr       仅删签名并重建向量（不删 OCR 缓存、不重跑识别）
  --skip-milvus    仅清六城相关 OCR 缓存并重跑 OCR，不写 Milvus
  --llm-guide-denoise  离线调用大模型合并 desc+OCR 去广告/重复，写入 data/cache_llm_guide_body.json
  --force-llm-guide    与上项配合：忽略已有大模型缓存强制重算（耗 API）

依赖: rapidocr_onnxruntime（OCR）、智谱 embedding 配置、Milvus 与 rag 层一致；大模型去噪需 DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 兼容从仓库根或 backend 目录执行（本文件在 tools/offlineTool 下，backend 再上两级）
backend_root = Path(__file__).resolve().parents[2]
repo_root = backend_root.parent
for p in (str(backend_root), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.planner_node import TRAVEL_RAG_CITY_ALLOWLIST
from rag.travel_cache_retriever import ensure_travel_vectorstore_by_city
import rag.travel_loader as travel_loader_module
from rag.offlineCache.travel_ocr import load_ocr_cache, save_ocr_cache


def collect_note_ids_from_allowlist_cities(data_root: Path) -> set[str]:
    """从六城 cache/*.json 汇总 note_id，用于精准清理 OCR 磁盘缓存。"""
    ids: set[str] = set()
    cache_dir = data_root / "cache"
    for city in TRAVEL_RAG_CITY_ALLOWLIST:
        path = cache_dir / f"{city}.json"
        if not path.is_file():
            print(f"rebuild_travel_ocr===========缺少城市文件跳过: {path}")
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            print(f"rebuild_travel_ocr===========读取失败 {path}: {ex!r}")
            continue
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, dict):
                continue
            nid = str(item.get("note_id") or "").strip()
            if nid:
                ids.add(nid)
    return ids


def clear_ocr_cache_entries(data_root: Path, note_ids: set[str]) -> int:
    """从 cache_ocr_text.json 删除指定笔记条目，强制下次 allow_runtime_ocr 时重新识别。"""
    ocr_path = data_root / "cache_ocr_text.json"
    cache = load_ocr_cache(ocr_path)
    removed = 0
    for nid in note_ids:
        if nid in cache:
            del cache[nid]
            removed += 1
    if removed:
        save_ocr_cache(ocr_path, cache)
    return removed


def delete_travel_milvus_signature_files(data_root: Path) -> list[Path]:
    """删除持久化签名，使 ensure_travel_vectorstore_by_city 走全量删 collection + 重嵌入。"""
    deleted: list[Path] = []
    for path in sorted(data_root.glob(".travel_milvus_sig_*.json")):
        try:
            path.unlink()
            deleted.append(path)
        except OSError as ex:
            print(f"rebuild_travel_ocr===========删签名失败 {path}: {ex!r}")
    return deleted


def run_ocr_for_each_city(
    data_path: str,
    use_llm_guide_denoise: bool = False,
    force_refresh_llm_guide: bool = False,
) -> None:
    """
    逐城加载语料并打开 allow_runtime_ocr，触发 RapidOCR 写回 cache_ocr_text.json。
    可选 use_llm_guide_denoise：对未命中 cache_llm_guide_body 的笔记调用大模型生成纯攻略正文。
    """
    for city in TRAVEL_RAG_CITY_ALLOWLIST:
        print(
            f"rebuild_travel_ocr===========开始 OCR city={city!r}（单城可能数分钟，请等待）",
            flush=True,
        )
        t0 = time.perf_counter()
        prev_ocr = travel_loader_module.loader_allow_runtime_ocr
        prev_force = travel_loader_module.loader_force_refresh_llm_guide
        travel_loader_module.loader_allow_runtime_ocr = True
        travel_loader_module.loader_force_refresh_llm_guide = force_refresh_llm_guide
        try:
            docs = travel_loader_module.load_travel_cache_docs(
                data_path,
                city_name=city,
                use_llm_guide_denoise=use_llm_guide_denoise,
            )
        finally:
            travel_loader_module.loader_allow_runtime_ocr = prev_ocr
            travel_loader_module.loader_force_refresh_llm_guide = prev_force
        dt = time.perf_counter() - t0
        print(
            f"rebuild_travel_ocr===========OCR+加载完成 city={city!r} "
            f"docs={len(docs)} 耗时={dt:.1f}s"
        )


def run_llm_guide_fill_only(
    data_path: str,
    force_refresh_llm_guide: bool = False,
) -> None:
    """
    不重跑 OCR，仅逐城加载缓存并补全 cache_llm_guide_body（未命中或 force 时调大模型）。
    适用于已有 OCR 与图片缓存、只想批量正文去噪的场景。
    """
    for city in TRAVEL_RAG_CITY_ALLOWLIST:
        print(
            f"rebuild_travel_ocr===========仅大模型攻略去噪 city={city!r}",
            flush=True,
        )
        t0 = time.perf_counter()
        prev_force = travel_loader_module.loader_force_refresh_llm_guide
        travel_loader_module.loader_force_refresh_llm_guide = force_refresh_llm_guide
        try:
            docs = travel_loader_module.load_travel_cache_docs(
                data_path,
                city_name=city,
                use_llm_guide_denoise=True,
            )
        finally:
            travel_loader_module.loader_force_refresh_llm_guide = prev_force
        dt = time.perf_counter() - t0
        print(
            f"rebuild_travel_ocr===========LLM 去噪加载完成 city={city!r} "
            f"docs={len(docs)} 耗时={dt:.1f}s"
        )


def run_milvus_rebuild_for_each_key() -> None:
    """对六城各 key 与全量 all 各跑一次 ensure，内部会 drop 旧 collection 并分批写入。"""
    for city in TRAVEL_RAG_CITY_ALLOWLIST:
        t0 = time.perf_counter()
        docs, vs = ensure_travel_vectorstore_by_city(city)
        dt = time.perf_counter() - t0
        print(
            f"rebuild_travel_ocr===========Milvus 重建完成 city={city!r} "
            f"docs={len(docs)} vectorstore={'ok' if vs else 'none'} 耗时={dt:.1f}s"
        )
    t0 = time.perf_counter()
    docs, vs = ensure_travel_vectorstore_by_city(None)
    dt = time.perf_counter() - t0
    print(
        f"rebuild_travel_ocr===========Milvus 重建完成 city=all "
        f"docs={len(docs)} vectorstore={'ok' if vs else 'none'} 耗时={dt:.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="六城 OCR 重跑 + 旅游 Milvus 全量重建")
    parser.add_argument("--skip-ocr", action="store_true", help="不重跑 OCR，只重建 Milvus")
    parser.add_argument("--skip-milvus", action="store_true", help="只重跑 OCR，不写 Milvus")
    parser.add_argument(
        "--llm-guide-denoise",
        action="store_true",
        help="加载每条笔记时对大模型缓存未命中的条目调用 DeepSeek 生成 guide_body",
    )
    parser.add_argument(
        "--force-llm-guide",
        action="store_true",
        help="配合 --llm-guide-denoise：跳过已有大模型缓存强制重调",
    )
    args = parser.parse_args()

    data_root = backend_root / "data"
    data_path = str(data_root)
    print("rebuild_travel_ocr===========任务开始", flush=True)

    if not args.skip_ocr:
        note_ids = collect_note_ids_from_allowlist_cities(data_root)
        print(
            "rebuild_travel_ocr===========六城 note 数量:",
            len(note_ids),
            "城市列表:",
            list(TRAVEL_RAG_CITY_ALLOWLIST),
        )
        removed = clear_ocr_cache_entries(data_root, note_ids)
        print(f"rebuild_travel_ocr===========已清除 OCR 缓存条目数: {removed}")
        run_ocr_for_each_city(
            data_path,
            use_llm_guide_denoise=args.llm_guide_denoise,
            force_refresh_llm_guide=args.force_llm_guide,
        )
    else:
        print("rebuild_travel_ocr===========--skip-ocr 已跳过 OCR 重跑")
        if args.llm_guide_denoise:
            run_llm_guide_fill_only(
                data_path,
                force_refresh_llm_guide=args.force_llm_guide,
            )

    if not args.skip_milvus:
        deleted = delete_travel_milvus_signature_files(data_root)
        print(
            "rebuild_travel_ocr===========已删除 Milvus 签名文件数:",
            len(deleted),
            [p.name for p in deleted],
            flush=True,
        )
        print("rebuild_travel_ocr===========开始 Milvus 全量重建（六城 + all）", flush=True)
        run_milvus_rebuild_for_each_key()
    else:
        print("rebuild_travel_ocr===========--skip-milvus 已跳过 Milvus 重建")

    print("rebuild_travel_ocr===========全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
