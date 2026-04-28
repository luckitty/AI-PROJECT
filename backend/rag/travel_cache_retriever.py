"""
旅游缓存检索器：复用 loader 的 source_type 动态加载能力，
将 travel_cache 语料独立建成 Chroma 检索域，和 txt 主知识库解耦。
"""
import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
INTENT_FIELD_KEYWORDS = {
    "foods_text": ["美食", "吃", "餐厅", "小吃", "早餐", "午餐", "晚餐", "夜宵"],
    "tags_text": ["拍照", "出片", "打卡", "机位", "景点"],
    "transport_text": ["交通", "地铁", "公交", "怎么去", "打车", "路线"],
}
INTENT_KEYWORD_TO_FIELDS = {}
for field_name, keywords in INTENT_FIELD_KEYWORDS.items():
    for keyword in keywords:
        INTENT_KEYWORD_TO_FIELDS.setdefault(keyword, []).append(field_name)

# 城市路由缓存：避免每次请求都重新 glob cache 目录。
cached_city_file_signature = tuple()
cached_city_names = []

def list_available_cities() -> List[str]:
    """列出 cache 目录里可路由的城市名（文件名去掉 .json），并复用进程内缓存。"""

    if not CACHE_DIR.is_dir():
        return []
    global cached_city_file_signature, cached_city_names
    city_files = sorted(CACHE_DIR.glob("*.json"))
    print("city_files===========city_files \n", city_files, cached_city_names,"\n")

    current_signature = tuple((p.name, p.stat().st_mtime, p.stat().st_size) for p in city_files)
    print("current_signature===========current_signature \n", current_signature, "\n")
    if current_signature == cached_city_file_signature and cached_city_names:
        return cached_city_names
    # 命中多个城市时按长度优先匹配，因此这里提前按长度降序缓存，后续 detect 可直接复用。
    cities = [p.stem for p in city_files]
    cached_city_names = sorted(cities, key=len, reverse=True)
    cached_city_file_signature = current_signature
    return cached_city_names


def detect_city_from_query(query: str) -> Optional[str]:
    """从用户问题中识别城市名；命中多个时优先匹配更长名称，减少误命中。"""
    q = (query or "").strip()
    if not q:
        return None
    cities = list_available_cities()
    for city in cities:
        if city and city in q:
            return city
    return None


def build_cache_key(city_name: Optional[str]) -> str:
    """把城市名映射成稳定缓存 key；未命中城市时使用 all。"""
    if not city_name:
        return "all"
    city_hash = hashlib.md5(city_name.encode("utf-8")).hexdigest()[:12]
    return f"city_{city_hash}"


def collection_name_for_key(cache_key: str) -> str:
    """为每个缓存 key 生成独立 collection，避免不同城市互相覆盖。"""
    return f"{TRAVEL_COLLECTION_NAME_PREFIX}_{cache_key}"


def signature_to_jsonable(sig: Tuple) -> Any:
    """把 cache 签名的嵌套 tuple 转成可 json 序列化的结构。"""
    if isinstance(sig, tuple):
        return [signature_to_jsonable(x) for x in sig]
    return sig


def jsonable_to_signature(obj: Any) -> Tuple:
    """从磁盘读回嵌套 list，还原为 tuple（与 cache_signature_for_city 返回值可比）。"""
    if isinstance(obj, list):
        return tuple(jsonable_to_signature(x) for x in obj)
    return obj


def travel_milvus_sig_file_path(cache_key: str) -> Path:
    """磁盘记录「某 cache_key 对应的 Milvus 是否已与数据文件签名一致」，用于进程重启后跳过全量重嵌入。"""
    return BACKEND_ROOT / "data" / f".travel_milvus_sig_{cache_key}.json"


def load_persisted_travel_signature(cache_key: str) -> Optional[Tuple]:
    path = travel_milvus_sig_file_path(cache_key)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return jsonable_to_signature(raw.get("signature"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def save_persisted_travel_signature(cache_key: str, sig: Tuple) -> None:
    """建库成功后写入签名，便于下次冷启动直接连接已有 collection。"""
    path = travel_milvus_sig_file_path(cache_key)
    try:
        path.write_text(
            json.dumps({"signature": signature_to_jsonable(sig)}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        return


def milvus_has_collection_sync(collection_name: str) -> bool:
    """同步探测 Milvus 中是否已有对应 collection；连接失败或异常时返回 False，避免拖死 HTTP 请求。"""
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
        return bool(utility.has_collection(collection_name, using=alias))
    except Exception:
        return False
    finally:
        try:
            connections.disconnect(alias)
        except Exception:
            pass


def drop_travel_collection_if_exists(collection_name: str) -> None:
    """
    使用 pymilvus 同步删除旧 collection，避免 langchain_milvus 在 drop_old=True
    分支里触发 AsyncMilvusClient 未 await 的 RuntimeWarning。
    """
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
        if utility.has_collection(collection_name, using=alias):
            utility.drop_collection(collection_name, using=alias)
    finally:
        try:
            connections.disconnect(alias)
        except Exception:
            pass


def cache_signature_for_city(city_name: Optional[str]) -> Tuple:
    """生成缓存签名：城市模式只看单文件，全量模式看所有 json。"""
    if not CACHE_DIR.is_dir():
        return tuple()
    if city_name:
        city_file = CACHE_DIR / f"{city_name}.json"
        if not city_file.is_file():
            return tuple()
        stat = city_file.stat()
        return (city_file.name, stat.st_mtime, stat.st_size)
    signature = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        stat = p.stat()
        signature.append((p.name, stat.st_mtime, stat.st_size))
    return tuple(signature)


def ensure_travel_vectorstore_by_city(city_name: Optional[str]):
    """
    根据城市构建并缓存旅游笔记向量库。
    命中城市时只向量化该城市文件；未命中时回退到全量。
    """
    cache_key = build_cache_key(city_name)
    current_signature = cache_signature_for_city(city_name)
    cached_signature = cached_signatures.get(cache_key)
    cached_docs = cached_docs_by_key.get(cache_key)
    cached_vectorstore = cached_vectorstores.get(cache_key)
    # 签名一致时直接复用缓存（即使向量库为 None，也能避免每次都重复扫盘加载空数据）。
    if cached_docs is not None and cached_signature == current_signature:
        return cached_docs, cached_vectorstore

    docs = get_docs(str(BACKEND_ROOT / "data"), source_type="travel_cache", city_name=city_name)
    if not docs:
        cached_signatures[cache_key] = current_signature
        cached_docs_by_key[cache_key] = []
        cached_vectorstores[cache_key] = None
        return cached_docs_by_key[cache_key], cached_vectorstores[cache_key]

    target_collection_name = collection_name_for_key(cache_key)
    # 进程重启后内存缓存为空：若磁盘签名与当前数据文件一致且 Milvus 里已有 collection，则只连库不重跑智谱 embedding（冷启动大幅加速）。
    persisted_sig = load_persisted_travel_signature(cache_key)
    if persisted_sig == current_signature and milvus_has_collection_sync(target_collection_name):
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
            # 复用失败（版本/字段不一致等）时回退到下方全量重建，避免 search_travel 整条链路抛错无响应。
            pass

    # 旅游域使用独立 Milvus collection，避免覆盖主知识库 collection。
    # 同时每个城市 key 使用独立 collection，避免切换城市时互相 drop_old。
    # 智谱 embedding 接口单次最多 64 条，这里显式分批入库，避免 from_documents 一次性提交超限。
    embedding = get_embeddings()
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [doc.metadata["doc_id"] for doc in docs]
    first_batch_end = min(EMBED_BATCH_SIZE, len(docs))
    # 每次重建城市向量库前先同步删除旧 collection，避免触发 drop_old 的异步告警。
    drop_travel_collection_if_exists(target_collection_name)
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
    """
    先尝试城市路由构建检索器，未命中城市或该城市无数据时回退到全量检索器。
    """
    city_name = detect_city_from_query(query)
    docs, vectorstore = ensure_travel_vectorstore_by_city(city_name)
    if docs and vectorstore is not None:
        return docs, vectorstore
    return ensure_travel_vectorstore_by_city(None)


def retrieve_travel_docs(query: str, top_k: int = 4) -> List:
    """
    旅游缓存语义检索：先向量召回，再用结构化字段重排，返回 top_k 条。
    """
    docs, vectorstore = ensure_travel_vectorstore(query)
    if not docs or vectorstore is None:
        return []
    city_name = detect_city_from_query(query)
    # 城市旅游问题需要足够覆盖面，否则候选景点会缺「颐和园/鸟巢/水立方」这类核心点。
    effective_top_k = min(top_k, len(docs))
    if city_name:
        effective_top_k = min(max(top_k, 10), len(docs))
    initial_k = max(effective_top_k * 3, effective_top_k, 24)
    initial_k = min(initial_k, len(docs))
    candidates = vectorstore.similarity_search(query, k=initial_k)

    return rerank_docs_by_structured_profile(
        candidates,
        query,
        effective_top_k,
        city_name=city_name,
    )


def infer_query_intent_fields(query: str) -> list[str]:
    """根据用户问题推断应优先匹配的结构化字段。"""
    intent_fields = []
    seen_fields = set()
    for keyword, fields in INTENT_KEYWORD_TO_FIELDS.items():
        if keyword not in query:
            continue
        for field_name in fields:
            if field_name in seen_fields:
                continue
            seen_fields.add(field_name)
            intent_fields.append(field_name)
    return intent_fields


def rerank_docs_by_structured_profile(
    candidates: list,
    query: str,
    top_k: int,
    city_name: Optional[str] = None,
) -> list:
    """结合城市与意图字段进行轻量重排，提升垂类问题命中率。"""
    # 调用方已算出城市时直接复用，避免重复做城市文件扫描。
    city_name = city_name if city_name is not None else (detect_city_from_query(query) or "")
    intent_fields = infer_query_intent_fields(query)
    scored_items = []
    for rank, doc in enumerate(candidates):
        metadata = doc.metadata or {}
        score = 0
        doc_city = str(metadata.get("city") or "")
        if city_name and city_name == doc_city:
            score += 4
        for field_name in intent_fields:
            field_text = str(metadata.get(field_name) or "")
            if not field_text:
                continue
            # 即使 query 没有逐词命中，只要命中对应意图字段，也给一个基础提权。
            score += 2
            for keyword in INTENT_FIELD_KEYWORDS.get(field_name, []):
                if keyword in query and keyword in field_text:
                    score += 2
        scored_items.append((score, -rank, doc))
    scored_items.sort(reverse=True)
    return [item[2] for item in scored_items[:top_k]]
