from langchain_community.chat_message_histories import RedisChatMessageHistory
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    """
    获取 Redis 对话历史（短期记忆）
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix="chat_history:",
        ttl=60 * 60 * 24  # 1天过期（企业一般会设置）
    )