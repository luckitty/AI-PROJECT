"""
Redis 连接与 TTL 常量。

LangGraph 的 ``RedisSaver`` 需要带 **Redis Search + JSON** 模块的环境（例如
``redis/redis-stack-server`` 镜像），纯 ``redis:latest`` 无模块时建索引会失败。
"""

# 与 LangGraph Agent 检查点、RedisChatMessageHistory 共用  使用6380是因为6379被占用
REDIS_URL = "redis://localhost:6380/0"

# Agent 检查点 key 的默认 TTL（分钟），与短期记忆约一天对齐
AGENT_CHECKPOINT_TTL_MINUTES = 1440

# Runnable 链里 RedisChatMessageHistory 的过期时间（秒）
CHAT_HISTORY_TTL_SECONDS = 86400
