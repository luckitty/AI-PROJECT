from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import hashlib

from rag.embedding import get_embeddings
from rag.travel_loader import load_travel_cache_docs

# 切分策略：True 用语义切分（多次 embed_documents，慢、有 API 费用，但边界更贴话题）；
# False 用固定窗口，与旧行为一致。需要语义化切分时改为 True。
USE_SEMANTIC_CHUNKING = False

# 语义切分时用于拆句的正则：英文在标点后要有空白；中文标点后常无空白，用「标点 + 下一字非空白」切分。
SENTENCE_SPLIT_REGEX = r"(?:(?<=[.?!])\s+|(?<=[。！？])(?=\S))"


# =========================
# 1️⃣ 加载 + 切分
# =========================
def load_and_split_txt(data_path="data", embeddings=None):
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
        doc.metadata = doc.metadata or {}
        source_type = doc.metadata.get("source_type")
        note_id = str(doc.metadata.get("note_id", "")).strip()
        # 旅游缓存优先用 note_id 作为稳定主键，避免不同笔记正文重复导致 doc_id 冲突。
        if source_type == "travel_cache" and note_id:
            doc_id = f"travel_{note_id}"
        else:
            # 其它文档沿用内容 hash，保持旧行为。
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        doc.metadata.update({
            "doc_id": doc_id,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": i
        })
    return docs


# =========================
# 获取文档
# =========================
def get_docs(data_path, embeddings=None, source_type="txt", city_name=None):
    """
    获取文档入口：按 source_type 选择不同数据源，统一返回 Document 列表。
    source_type:
    - txt: 读取 data_path 下 txt 并按原策略切分（用于本地人物/知识库）
    - travel_cache: 读取 data_path/cache 下旅游笔记缓存，按 note_id 整篇建 Document
      可选 city_name，仅加载城市文件，减少无关向量化开销
    """
    if source_type == "travel_cache":
        # 查询链路默认不做实时 OCR 补全：只复用已有缓存，避免一次检索被图片识别拖到分钟级。
        docs = load_travel_cache_docs(
            data_path,
            city_name=city_name,
            allow_runtime_ocr=False,
        )
    else:
        # embeddings 可选：由调用方（如 ensure_rag）传入时，语义切分与向量库共用同一嵌入客户端
        docs = load_and_split_txt(data_path, embeddings=embeddings)
    docs = add_doc_id(docs)
    return docs
