import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

import hashlib
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from backend.core.config import ZHIPU_API_KEY
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# from langchain_milvus import Milvus

from rank_bm25 import BM25Okapi
import numpy as np
import jieba

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("DEEPSEEK_BASE_URL")

# LLM（与 OpenAI 兼容接口，当前为 DeepSeek）
model = ChatOpenAI(
    model="deepseek-chat",
    max_tokens=1024,
    temperature=0,
    streaming=False
)

# =========================
# 1️⃣ 加载 + 切分
# =========================
def load_and_split(data_path="data"):
    loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# =========================
# 2️⃣ 用内容生成稳定 hash 作为 doc_id
# =========================
def add_doc_id(docs):
    for i, doc in enumerate(docs):
        # 用内容生成稳定 hash（企业常用）
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

        doc.metadata = doc.metadata or {}
        doc.metadata.update({
            "doc_id": content_hash,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": i
        })
    return docs

# =========================
# 2️⃣ Embedding
# =========================
def get_embeddings():
    return ZhipuAIEmbeddings(
        api_key=ZHIPU_API_KEY,
        model="embedding-3",
    )

# =========================
# 3️⃣ 向量库（Milvus）
# =========================
def get_vectorstore(docs, embedding):
    conn = {"host": "localhost", "port": "19530"}
    name = "rag_demo"

    # vectorstore = Milvus.from_documents(
    #     docs,
    #     embedding,
    #     connection_args=conn,
    #     collection_name=name,
    #     drop_old=False,  # 已有同名 collection 时不删；仅追加新数据（可能重复，需自行管理）
    # )

    ids = [doc.metadata["doc_id"] for doc in docs]
    vectorstore = Chroma.from_documents(
        docs,
        embedding,
        ids=ids,  # 👈 核心
        persist_directory="./chroma_db",
    )
    return vectorstore


# =========================
# 4️⃣ Hybrid Retriever（BM25 + 向量）
# =========================
class HybridRetriever:
    def __init__(self, vectorstore, docs):
        self.vectorstore = vectorstore
        self.docs = docs

        # 建 doc_id → doc 映射（关键）
        self.id2doc = {doc.metadata["doc_id"]: doc for doc in docs}

        # BM25
        self.texts = [doc.page_content for doc in docs]
        self.tokenized = [list(jieba.cut(text)) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query, k=4, rrf_k=60):
        """
        使用 RRF 算法进行融合
        :param query: 搜索词
        :param k: 最终返回的数量
        :param rrf_k: RRF 公式中的常数，通常取 60
        """

        # 1️⃣ BM25 检索（带分词）
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        bm25_topk_idx = np.argsort(bm25_scores)[::-1][:k]
        # ===== 2️⃣ 向量检索（MMR）=====
        vector_docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=20,
        )

        # doc_id -> RRF score
        score_dict = {}

        # BM25 rank 融合
        for rank, idx in enumerate(bm25_topk_idx):
            doc = self.docs[idx]
            doc_id = doc.metadata["doc_id"]

            rrf_score = 1.0 / (rrf_k + rank)
            score_dict[doc_id] = score_dict.get(doc_id, 0) + rrf_score


      
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata["doc_id"]

            rrf_score = 1.0 / (rrf_k + rank)
            score_dict[doc_id] = score_dict.get(doc_id, 0) + rrf_score


        # ===== 3️⃣ 排序 =====
        sorted_ids = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

        # ===== 4️⃣ 还原 doc =====
        final_docs = [self.id2doc[doc_id] for doc_id, _ in sorted_ids]

        return final_docs[:k]


# =========================
# 5 CrossEncoder 精排（可用 USE_RERANKER=0 关闭以加快启动、省显存）
# =========================
class NoOpReranker:
    def rerank(self, query, docs, top_k=3):
        return docs[:top_k] if docs else []


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return []

        # 构造 (query, doc) 对
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        # 打分排序
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

def make_search_tool(retriever: HybridRetriever, reranker: Reranker, k: int = 4):
    """把混合检索封装成 LangChain tool，供 create_agent 调用。"""

    @tool
    def search_local_knowledge(user_query: str) -> str:
        """从本地文本知识库检索与用户问题相关的片段（BM25+向量混合）。回答人物、作品、经历等事实性问题前应先调用本工具。"""
        docs = retriever.retrieve(user_query, k=k)
        print("docs===========检索结果 \n",docs)
        if not docs:
            return "未检索到相关文档片段。"
        
        reranked_docs = reranker.rerank(user_query, docs, top_k=min(k, 5))
        parts = []
        for i, doc in enumerate(reranked_docs, start=1):
            parts.append(f"[片段{i}]\n{doc.page_content}")
        return "\n\n".join(parts)

    return search_local_knowledge


RAG_AGENT_SYSTEM = """你是问答助手。用户问的问题若可能来自本地资料，请先调用工具 search_local_knowledge 获取片段，再只根据工具返回的内容作答；若工具无相关内容，请如实说明。用中文、简洁回答。"""


def build_rag_agent(retriever: HybridRetriever, reranker: Reranker, k: int = 4):
    search_tool = make_search_tool(retriever, reranker, k=k)
    return create_agent(
        model,
        tools=[search_tool],
        system_prompt=RAG_AGENT_SYSTEM,
    )


def run_rag_agent(agent, user_query: str) -> str:
    """执行 Agent 图，返回最后一条 AI 文本。"""
    result = agent.invoke({"messages": [HumanMessage(content=user_query)]})
    messages = result.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    return getattr(last, "content", None) or str(last)


# =========================
# 5️⃣ 仅检索（调试检索质量，不走 Agent）
# =========================
def query_rag(query, retriever, reranker, retrieve_k: int = 20, rerank_top_k: int = 5):
    docs = retriever.retrieve(query, k=retrieve_k)
    reranked_docs = reranker.rerank(query, docs, top_k=rerank_top_k)

    print("\n🔍 检索结果：\n")
    for i, doc in enumerate(reranked_docs):
        print(f"--- 文档 {i+1} ---")
        print(doc.page_content[:200])

    return reranked_docs


# =========================
# 6️⃣ 主函数（直接运行）
# =========================
if __name__ == "__main__":
    print("🚀 开始加载数据...")

    docs = load_and_split("data")
    docs = add_doc_id(docs)
    print(f"✅ 文档切分完成，共 {len(docs)} chunks")

    embedding = get_embeddings()

    print("🚀 构建向量库...")
    vectorstore = get_vectorstore(docs, embedding)

    retriever = HybridRetriever(vectorstore, docs)
    use_rerank = os.getenv("USE_RERANKER", "1").lower() not in ("0", "false", "no")
    print("use_rerank===========是否使用精排 \n",use_rerank)
    reranker = Reranker() if use_rerank else NoOpReranker()
    if not use_rerank:
        print("   已关闭 CrossEncoder 精排（USE_RERANKER=0）\n")
    use_agent = os.getenv("USE_AGENT", "1").lower() not in ("0", "false", "no")
    agent = build_rag_agent(retriever, reranker, k=4) if use_agent else None

    print("✅ 初始化完成，可以开始查询")
    if use_agent:
        print("   模式：Agent + 检索工具（设置 USE_AGENT=0 可仅看检索片段）\n")
    else:
        print("   模式：仅混合检索，不调用 LLM\n")

    while True:
        query = input("请输入问题（q退出）：")
        if query.lower() == "q":
            break

        # query_rag(query, retriever, reranker)



        if os.getenv("RAG_DEBUG_RETRIEVAL", "").lower() in ("1", "true", "yes"):
            query_rag(query, retriever, reranker)

        if use_agent and agent is not None:
            answer = run_rag_agent(agent, query)
            print("\n🤖 回答：\n")
            print(answer)
            print()
        else:
            query_rag(query, retriever, reranker)