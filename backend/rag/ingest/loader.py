from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib


# =========================
# 1️⃣ 加载 + 切分
# =========================
def load_and_split(data_path="data"):
    print("load_and_split===========加载 + 切分 \n",data_path)
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
# 获取文档
# =========================
def get_docs(data_path):
    print("get_docs===========获取文档 \n",data_path)
    docs = load_and_split(data_path)
    docs = add_doc_id(docs)
    return docs
