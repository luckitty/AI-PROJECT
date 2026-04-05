import os

from langchain_chroma import Chroma

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

    persist_directory = "./chroma_db"
    sqlite_path = os.path.join(persist_directory, "chroma.sqlite3")
    force_rebuild = os.getenv("RAG_FORCE_REBUILD", "").lower() in ("1", "true", "yes")

    # 已有持久化库则直接打开，避免每次启动全量重嵌入
    if not force_rebuild and os.path.isfile(sqlite_path):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )

    ids = [doc.metadata["doc_id"] for doc in docs]
    vectorstore = Chroma.from_documents(
        docs,
        embedding,
        ids=ids,  # 👈 核心
        persist_directory=persist_directory,
    )
    return vectorstore
