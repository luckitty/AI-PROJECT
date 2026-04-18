"""
旅游缓存检索器：travel_cache 独立 Milvus 域，与主知识库解耦。
"""
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

from rag.embedding import get_embeddings
from rag.loader import get_docs
from langchain_milvus import Milvus
from pymilvus import connections, utility
from rag.vectorstores.milvus_client import milvus_connection_args

BACKEND_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = BACKEND_ROOT / "data" / "cache"
TRAVEL_COLLECTION_NAME_PREFIX = "travel_cache_collection"
EMBED_BATCH_SIZE = 32

cached_signatures = {}
cached_docs_by_key = {}
cached_vectorstores = {}

FOOD_FOCUS_KEYWORDS = (
    "美食",
    "好吃的",
    "小吃",
    "必吃",
    "餐厅",
    "美食攻略",
    "吃货",
    "去哪吃",
    "吃啥",
    "打卡美食",
    "特色菜",
)


def is_food_focus_query(query: str) -> bool:
    """是否美食向问题：用于放大召回与同城餐饮笔记合并。调用方保证 query 为有效字符串。"""
    return any(keyword in query for keyword in FOOD_FOCUS_KEYWORDS)


def metadata_suggests_food(metadata: dict) -> bool:
    """元数据是否像餐饮笔记（与向量召回互补）。"""
    if not metadata:
        return False
    if str(metadata.get("foods_text") or "").strip():
        return True
    if "美食" in str(metadata.get("tags_text") or ""):
        return True
    combined = f"{metadata.get('title') or ''}\n{metadata.get('desc') or ''}"
    return any(k in combined for k in FOOD_FOCUS_KEYWORDS)


def list_available_cities() -> List[str]:
    """cache 目录下城市名（文件名），长名优先匹配。"""
    if not CACHE_DIR.is_dir():
        return []
    return sorted([p.stem for p in sorted(CACHE_DIR.glob("*.json"))], key=len, reverse=True)


def detect_city_from_query(query: str) -> Optional[str]:
    """从问题里匹配城市名。调用方保证 query 为有效字符串。"""
    for city in list_available_cities():
        if city and city in query:
            return city
    return None


def build_cache_key(city_name: Optional[str]) -> str:
    if not city_name:
        return "all"
    return f"city_{hashlib.md5(city_name.encode('utf-8')).hexdigest()[:12]}"


def milvus_sig_path(cache_key: str) -> Path:
    return BACKEND_ROOT / "data" / f".travel_milvus_sig_{cache_key}.json"


def load_persisted_travel_signature(cache_key: str) -> Optional[Tuple]:
    path = milvus_sig_path(cache_key)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        s = raw.get("signature")
        if s is None:
            return None
        # 全量：[[name,mtime,size],...]；单城：[name,mtime,size]
        if s and isinstance(s[0], list):
            return tuple(tuple(x) for x in s)
        return tuple(s)
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return None


def save_persisted_travel_signature(cache_key: str, sig: Tuple) -> None:
    path = milvus_sig_path(cache_key)
    if sig and isinstance(sig[0], tuple):
        payload = [list(x) for x in sig]
    else:
        payload = list(sig)
    try:
        path.write_text(json.dumps({"signature": payload}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return


def travel_milvus_sync(collection_name: str, op: str):
    """op 为 has 时返回是否已有 collection；为 drop 时存在则删除。"""
    conn = milvus_connection_args()
    alias = "travel_cache_sync"
    try:
        connections.connect(
            alias=alias,
            host=conn["host"],
            port=conn["port"],
            timeout=conn["timeout"],
            grpc_options=conn.get("grpc_options"),
        )
        if op == "has":
            return bool(utility.has_collection(collection_name, using=alias))
        if utility.has_collection(collection_name, using=alias):
            utility.drop_collection(collection_name, using=alias)
    except Exception:
        if op == "has":
            return False
    finally:
        try:
            connections.disconnect(alias)
        except Exception:
            pass
    if op == "has":
        return False


def ocr_signature_tuple() -> Tuple:
    """
    OCR 聚合缓存文件参与 Milvus 签名。
    仅更新 data/cache_ocr_text.json 而未改城市 JSON 时，也必须触发向量重算，否则会长期沿用无 OCR 的旧向量。
    """
    ocr_path = BACKEND_ROOT / "data" / "cache_ocr_text.json"
    if not ocr_path.is_file():
        return ("", 0.0, 0)
    stat = ocr_path.stat()
    return (ocr_path.name, stat.st_mtime, stat.st_size)


def cache_signature_for_city(city_name: Optional[str]) -> Tuple:
    if not CACHE_DIR.is_dir():
        return tuple()
    if city_name:
        city_file = CACHE_DIR / f"{city_name}.json"
        if not city_file.is_file():
            return tuple()
        stat = city_file.stat()
        # 与城市文件、OCR 缓存三元组一起签名，形状为 (文件签名, OCR 签名)。
        return ((city_file.name, stat.st_mtime, stat.st_size), ocr_signature_tuple())
    files = tuple(
        (p.name, p.stat().st_mtime, p.stat().st_size) for p in sorted(CACHE_DIR.glob("*.json"))
    )
    # 全量：每个城市文件一条签名，最后追加 OCR 缓存文件签名。
    return files + (ocr_signature_tuple(),)


def ensure_travel_vectorstore_by_city(city_name: Optional[str]):
    """按城市建/缓存向量库；无数据时缓存空列表。"""
    cache_key = build_cache_key(city_name)
    current_signature = cache_signature_for_city(city_name)
    cached_signature = cached_signatures.get(cache_key)
    cached_docs = cached_docs_by_key.get(cache_key)
    cached_vectorstore = cached_vectorstores.get(cache_key)
    if cached_docs is not None and cached_signature == current_signature:
        return cached_docs, cached_vectorstore

    docs = get_docs(str(BACKEND_ROOT / "data"), source_type="travel_cache", city_name=city_name)
    if not docs:
        cached_signatures[cache_key] = current_signature
        cached_docs_by_key[cache_key] = []
        cached_vectorstores[cache_key] = None
        return cached_docs_by_key[cache_key], cached_vectorstores[cache_key]

    target_collection_name = f"{TRAVEL_COLLECTION_NAME_PREFIX}_{cache_key}"
    persisted_sig = load_persisted_travel_signature(cache_key)
    if persisted_sig == current_signature and travel_milvus_sync(target_collection_name, "has"):
        try:
            embedding = get_embeddings()
            vectorstore = Milvus(
                embedding_function=embedding,
                connection_args=milvus_connection_args(),
                collection_name=target_collection_name,
            )
            cached_signatures[cache_key] = current_signature
            cached_docs_by_key[cache_key] = docs
            cached_vectorstores[cache_key] = vectorstore
            return cached_docs_by_key[cache_key], cached_vectorstores[cache_key]
        except Exception:
            pass

    embedding = get_embeddings()
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [doc.metadata["doc_id"] for doc in docs]
    first_batch_end = min(EMBED_BATCH_SIZE, len(docs))
    travel_milvus_sync(target_collection_name, "drop")
    vectorstore = Milvus.from_texts(
        texts=texts[:first_batch_end],
        embedding=embedding,
        metadatas=metadatas[:first_batch_end],
        ids=ids[:first_batch_end],
        connection_args=milvus_connection_args(),
        collection_name=target_collection_name,
        drop_old=False,
    )
    for start in range(first_batch_end, len(docs), EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, len(docs))
        vectorstore.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
    save_persisted_travel_signature(cache_key, current_signature)
    cached_signatures[cache_key] = current_signature
    cached_docs_by_key[cache_key] = docs
    cached_vectorstores[cache_key] = vectorstore
    return cached_docs_by_key[cache_key], cached_vectorstores[cache_key]


def ensure_travel_vectorstore(query: str):
    city_name = detect_city_from_query(query)
    docs, vectorstore = ensure_travel_vectorstore_by_city(city_name)
    if docs and vectorstore is not None:
        return docs, vectorstore
    return ensure_travel_vectorstore_by_city(None)


def retrieve_travel_docs(query: str, top_k: int = 4) -> List:
    """向量召回后按城市 + 餐饮信号重排，返回 top_k。"""
    docs, vectorstore = ensure_travel_vectorstore(query)
    if not docs or vectorstore is None:
        return []
    city_name = detect_city_from_query(query) or ""
    food_focus = is_food_focus_query(query)
    print("top_k===========top_k \n", top_k, "\n")
    print("len(docs===========len(docs \n", len(docs), "\n")
    if food_focus:
        effective_top_k = min(max(top_k, 10), len(docs))
        initial_k = min(max(effective_top_k * 4, 32), len(docs))
    else:
        effective_top_k = min(top_k, len(docs))
        initial_k = min(max(top_k * 3, top_k), len(docs))
    candidates = vectorstore.similarity_search(query, k=initial_k)

    if food_focus and city_name:
        seen = set()
        for doc in candidates:
            key = (doc.metadata or {}).get("doc_id") or (doc.metadata or {}).get("note_id")
            if key:
                seen.add(key)
        for doc in docs:
            meta = doc.metadata or {}
            if str(meta.get("city") or "") != city_name or not metadata_suggests_food(meta):
                continue
            key = meta.get("doc_id") or meta.get("note_id")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            candidates.append(doc)

    scored = []
    for rank, doc in enumerate(candidates):
        meta = doc.metadata or {}
        score = 0
        if city_name and str(meta.get("city") or "") == city_name:
            score += 4
        if food_focus and metadata_suggests_food(meta):
            score += 6
        scored.append((score, -rank, doc))
    scored.sort(reverse=True)
    return [x[2] for x in scored[:effective_top_k]]
