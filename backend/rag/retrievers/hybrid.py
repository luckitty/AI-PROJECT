import jieba
from rank_bm25 import BM25Okapi
import numpy as np

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

        # ===== 1️⃣ BM25 检索（带分词）
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        print("bm25_scores:=========\n", bm25_scores, "\n\n")
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
