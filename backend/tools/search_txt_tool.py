import os

from langchain.tools import tool
from rag.embeddings.hf_embedding import get_embeddings
from rag.ingest.loader import get_docs
from rag.retrievers.hybrid import HybridRetriever
from rag.rerank.reranker import NoOpReranker, Reranker
from rag.vectorstores.chroma_client import get_vectorstore
from datetime import datetime
# 懒加载：避免 import assistant 时全量建库
rag_ready = False
hybrid_retriever = None
knowledge_reranker = None


def ensure_rag():
    print("ensure_rag===========确保RAG \n", datetime.now())
    global rag_ready, hybrid_retriever, knowledge_reranker
    if rag_ready:
        return
    docs = get_docs("data")
    embedding = get_embeddings()
    vector_store = get_vectorstore(docs, embedding)
    hybrid_retriever = HybridRetriever(vector_store, docs)
    use_rerank = os.getenv("USE_RERANKER", "1").lower() not in ("0", "false", "no")
    knowledge_reranker = Reranker() if use_rerank else NoOpReranker()
    rag_ready = True


@tool
def search_local_knowledge(user_query: str) -> str:
    """从本地文本知识库检索与用户问题相关的片段（BM25+向量混合）。回答人物、作品、经历等事实性问题前应先调用本工具。"""
    k = 4
    ensure_rag()
    docs = hybrid_retriever.retrieve(user_query, k=k)
    if not docs:
        return "未检索到相关文档片段。"

    reranked_docs = knowledge_reranker.rerank(user_query, docs, top_k=min(k, 5))
    parts = []
    for i, doc in enumerate(reranked_docs, start=1):
        parts.append(f"[片段{i}]\n{doc.page_content}")
    print("docs===========检索结果 \n", datetime.now())

    return "\n\n".join(parts)
