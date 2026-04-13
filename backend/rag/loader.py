from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import hashlib

from rag.embedding import get_embeddings

# 切分策略：True 用语义切分（多次 embed_documents，慢、有 API 费用，但边界更贴话题）；
# False 用固定窗口，与旧行为一致。需要语义化切分时改为 True。
USE_SEMANTIC_CHUNKING = False

# 语义切分时用于拆句的正则：英文在标点后要有空白；中文标点后常无空白，用「标点 + 下一字非空白」切分。
SENTENCE_SPLIT_REGEX = r"(?:(?<=[.?!])\s+|(?<=[。！？])(?=\S))"


# =========================
# 1️⃣ 加载 + 切分
# =========================
def load_and_split(data_path="data", embeddings=None):
    print("load_and_split===========加载 + 切分 \n",data_path)
    loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

    if USE_SEMANTIC_CHUNKING:
        # 与 ensure_rag 共用同一 Embeddings 实例，避免切分与建库各 new 一份客户端
        emb = embeddings if embeddings is not None else get_embeddings()
        # percentile：相邻句组向量距离超过全体距离的该分位点则断块；块偏细可调高（如 98），偏粗调低（如 90）
        splitter = SemanticChunker(
            emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            sentence_split_regex=SENTENCE_SPLIT_REGEX,
        )
    else:
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
# 获取文档
# =========================
def get_docs(data_path, embeddings=None):
    print("get_docs===========获取文档 \n",data_path)
    # embeddings 可选：由调用方（如 ensure_rag）传入时，语义切分与向量库共用同一嵌入客户端
    docs = load_and_split(data_path, embeddings=embeddings)
    docs = add_doc_id(docs)
    return docs
