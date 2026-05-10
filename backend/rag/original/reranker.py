from sentence_transformers import CrossEncoder


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
