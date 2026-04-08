"""
短期记忆（Redis）

- **LangGraph Agent**：使用 ``RedisSaver`` 作为 ``checkpointer``，按 ``thread_id`` 持久化
  整图状态（含 messages、工具调用等），与服务进程无关，并支持 ``delete_thread`` 清空会话。
- **Runnable 链**（可选）：``get_redis_history`` 基于 ``RedisChatMessageHistory``，
  仅适合 ``RunnableWithMessageHistory``，与 Agent 的检查点机制不同，二者勿混用。
"""
import threading

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langgraph.checkpoint.redis import RedisSaver

from memory.redis_config import (
    AGENT_CHECKPOINT_TTL_MINUTES,
    CHAT_HISTORY_TTL_SECONDS,
    REDIS_URL,
)

# LangGraph：进程内单例，避免重复建连与重复建索引
redis_agent_checkpointer = None
# 创建一个线程锁，用来保护“初始化 Redis checkpointer”这段代码，避免并发时重复创建
redis_checkpointer_lock = threading.Lock()

# Runnable 链用的历史：与 checkpoint 不同 key 前缀，避免键冲突
CHAT_HISTORY_KEY_PREFIX = "lc_chat_msg:"


def get_short_term_checkpointer() -> RedisSaver:
    """
    返回供 ``create_agent(..., checkpointer=...)`` 使用的 RedisSaver 单例。

    须在已部署 **Redis Stack**（含 RediSearch + RedisJSON）的实例上使用，
    否则建索引会失败；见 ``redis_config`` 模块说明。
    """
    global redis_agent_checkpointer
    with redis_checkpointer_lock:
        if redis_agent_checkpointer is None:
            saver = RedisSaver(
                redis_url=REDIS_URL,
                ttl={
                    # 分钟；与 redis_config 中「约一天」对齐，过期后 Redis 自动删键
                    "default_ttl": AGENT_CHECKPOINT_TTL_MINUTES,
                },
            )
            # 在 Redis 上创建搜索索引并初始化 key registry；未调用则无法正常读写检查点
            saver.setup()
            redis_agent_checkpointer = saver
        return redis_agent_checkpointer

# 暂没用到
def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    """
    供 ``RunnableWithMessageHistory`` 使用：按 session 存取消息列表。

    与 ``get_short_term_checkpointer`` 用途不同；当前项目的 Agent 走 LangGraph 检查点，
    不经过本函数。
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix=CHAT_HISTORY_KEY_PREFIX,
        ttl=CHAT_HISTORY_TTL_SECONDS,
    )
