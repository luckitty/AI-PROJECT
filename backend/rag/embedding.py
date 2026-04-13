from langchain_community.embeddings import ZhipuAIEmbeddings
from backend.core.config import ZHIPU_API_KEY

# =========================
# 2️⃣ Embedding
# =========================
def get_embeddings():
    return ZhipuAIEmbeddings(
        api_key=ZHIPU_API_KEY,
        model="embedding-3",
    )
