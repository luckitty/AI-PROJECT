import os

from langchain.tools import tool
from rag.embedding import get_embeddings
from rag.loader import get_docs
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import NoOpReranker, Reranker
# from rag.vectorstores.chroma_client import get_vectorstore
from rag.vectorstores.milvus_client import get_vectorstore_milvus
from datetime import datetime
# 懒加载：避免 import assistant 时全量建库
rag_ready = False
hybrid_retriever = None
knowledge_reranker = None


def ensure_rag():
    global rag_ready, hybrid_retriever, knowledge_reranker
    print("ensure_rag===========确保RAG \n", rag_ready, "\n\n")
    if rag_ready:
        return
    embedding = get_embeddings()
    docs = get_docs("data", embeddings=embedding)
    # 使用chroma
    # vector_store = get_vectorstore(docs, embedding)
    # 使用milvus
    print("get_vectorstore_milvus:=========\n", datetime.now(), "\n\n")
    vector_store = get_vectorstore_milvus(docs, embedding)
    print("add_documents:=========\n", datetime.now(), "\n\n")
    hybrid_retriever = HybridRetriever(vector_store, docs)
    use_rerank = os.getenv("USE_RERANKER", "1").lower() not in ("0", "false", "no")
    knowledge_reranker = Reranker() if use_rerank else NoOpReranker()
    rag_ready = True


def retriever(user_query: str) -> str:
    """从本地文本知识库检索与问题相关的片段（BM25+向量混合，可能含噪声）。仅当回答需要知识库中的专有内容时再调用；通用知识或闲聊不要调用。返回后请自行判断片段是否与问题相关，仅引用能支撑结论的内容；无关或不足时请向用户说明，勿臆测。"""
    k = 4
    ensure_rag()
    docs = hybrid_retriever.retrieve(user_query, k=k)
    if not docs:
        return "未检索到相关文档片段。"

    reranked_docs = knowledge_reranker.rerank(user_query, docs, top_k=min(k, 5))
    parts = []
    for i, doc in enumerate(reranked_docs, start=1):
        parts.append(f"[片段{i}]\n{doc.page_content}")

    return "\n\n".join(parts)
