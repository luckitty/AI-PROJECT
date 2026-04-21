"""
长期记忆：用 Milvus 向量库按用户检索记忆（与 RAG 知识库分 collection，避免混写）。

连接参数与 ``rag.vectorstores.milvus_client.get_milvus_connection_args`` 一致；
collection 名称见本模块常量 ``LONG_MEMORY_COLLECTION``（与 RAG 默认的 ``rag_collection`` 区分）。
"""
import uuid
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_milvus import Milvus

import rag.vectorstores.milvus_langchain_compat  # noqa: F401
from rag.embedding import get_embeddings
from rag.vectorstores.milvus_client import get_milvus_connection_args

# 与 RAG 默认 collection 区分开，长期记忆单独存一份
LONG_MEMORY_COLLECTION = "long_memory"

# 首次用 from_documents 建表时是否 drop 已有同名 collection（开发环境可改为 True 清库重建）
DROP_OLD_ON_BOOTSTRAP = False

# 进程内单例，懒加载；与 RAG 里 skip_ingest 分支类似，只连库、按需写入
vector_store: Milvus | None = None
# 连接不可用时做进程内熔断，避免每次请求都重复打 Milvus 并刷异常日志。
long_memory_disabled = False
# 仅打印一次降级日志，避免控制台被重复错误淹没。
long_memory_warned = False


def disable_long_memory(reason: str) -> None:
    """
    标记长期记忆降级不可用（仅当前进程生效）。
    说明：Milvus 连接失败时，降级为“跳过读写长期记忆”，不影响主聊天流程。
    """
    global long_memory_disabled
    global long_memory_warned
    long_memory_disabled = True
    if not long_memory_warned:
        long_memory_warned = True


def get_vectorstore() -> Milvus:
    """懒加载 Milvus；首次调用时创建连接与 collection 句柄。"""
    global vector_store
    if long_memory_disabled:
        raise RuntimeError("long memory is disabled")
    if vector_store is None:
        try:
            embedding = get_embeddings()
            vector_store = Milvus(
                embedding_function=embedding,
                connection_args=get_milvus_connection_args(),
                collection_name=LONG_MEMORY_COLLECTION,
            )
        except Exception as exc:
            disable_long_memory(f"init failed: {exc}")
            raise
    return vector_store


def bootstrap_vectorstore_with_doc(doc: Document, doc_id: str) -> Milvus:
    """
    collection 尚不存在或尚未初始化时，用单条文档创建（等价于 RAG 里 Milvus.from_documents 建表）。
    """
    embedding = get_embeddings()
    return Milvus.from_documents(
        [doc],
        embedding,
        connection_args=get_milvus_connection_args(),
        collection_name=LONG_MEMORY_COLLECTION,
        drop_old=DROP_OLD_ON_BOOTSTRAP,
        ids=[doc_id],
    )


def save_long_memory(
    text: str,
    user_id: str,
    *,
    extra_metadata: dict | None = None,
) -> str:
    """
    写入一条长期记忆（向量入库，便于语义检索）。

    Returns:
        本条记忆在向量库中的 doc_id（与 Milvus 主键一致）。
    """
    doc_id = str(uuid.uuid4())
    meta: dict = {
        "doc_id": doc_id,
        "user_id": user_id,
        # ISO8601 UTC，便于排序与排查
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        meta.update(extra_metadata)
    doc = Document(page_content=text, metadata=meta)

    # 正常路径：直接向已存在（或可连接）的 collection 写入。
    # 兜底路径：若 collection 尚未初始化，使用 from_documents 完成建表并写入首条文档。
    if long_memory_disabled:
        return doc_id

    try:
        get_vectorstore().add_documents([doc], ids=[doc_id])
    except Exception:
        global vector_store
        try:
            vector_store = bootstrap_vectorstore_with_doc(doc, doc_id)
        except Exception as exc:
            disable_long_memory(f"save failed: {exc}")
    return doc_id


def document_matches_user(doc: Document, user_id: str) -> bool:
    """同一用户仅按 metadata.user_id 判断。"""
    meta = doc.metadata or {}
    return meta.get("user_id") == user_id


def search_long_memory(
    query: str,
    user_id: str,
    *,
    k: int = 4,
) -> list[Document]:
    """
    按语义检索当前用户相关的长期记忆。

    先向量多召回再在内存里按 user_id 过滤（Milvus 侧未建用户标量索引时仍可用）。
    """
    # 防御式处理：非法 k 或空查询时直接返回空结果，避免无意义向量检索。
    if k <= 0 or not query.strip():
        return []

    if long_memory_disabled:
        return []

    try:
        store = get_vectorstore()
    except Exception:
        return []

    # 多用户共享 collection 时，全局 TopK 可能全是其他用户，因此先多取再按 user_id 过滤。
    # 这里把召回上限控制在更合理范围，避免 k 很大时一次拉取过多候选造成额外开销。
    try:
        fetch_k = min(max(k * 20, 40), 200)
        candidates = store.similarity_search(query, k=fetch_k, expr=f'user_id == "{user_id}"')
    except Exception as exc:
        disable_long_memory(f"search failed: {exc}")
        return []

    matched_docs = [doc for doc in candidates if document_matches_user(doc, user_id)]
    return matched_docs[:k]
