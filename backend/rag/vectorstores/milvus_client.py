import os
from langchain_milvus import Milvus

import backend.rag.vectorstores.milvus_langchain_compat  # noqa: F401

def _milvus_connection_args():
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    timeout = float(os.getenv("MILVUS_TIMEOUT", "60"))
    return {
        "host": host,
        "port": port,
        "timeout": timeout,
        # 避免环境变量 HTTP(S)_PROXY 让 gRPC 走代理，导致连不上本机 Milvus
        "grpc_options": {"grpc.enable_http_proxy": 0},
    }

def get_vectorstore_milvus(docs, embedding):
    conn = _milvus_connection_args()
    collection_name = os.getenv("MILVUS_COLLECTION", "rag_collection")
    skip_ingest = os.getenv("MILVUS_SKIP_INGEST", "").lower() in ("1", "true", "yes")
    drop_old = os.getenv("MILVUS_DROP_OLD", "").lower() in ("1", "true", "yes")

    if skip_ingest:
        print(f"👉 跳过写入（MILVUS_SKIP_INGEST），仅连接 collection: {collection_name}")
        return Milvus(
            embedding_function=embedding,
            connection_args=conn,
            collection_name=collection_name,
        )

    ids = [doc.metadata["doc_id"] for doc in docs]
    print(f"👉 写入 Milvus（collection={collection_name}, drop_old={drop_old}）")
    vectorstore = Milvus.from_documents(
        docs,
        embedding,
        connection_args=conn,
        collection_name=collection_name,
        drop_old=drop_old,
        ids=ids,
    )
    
    print("✅ 写入完成")
    return vectorstore
