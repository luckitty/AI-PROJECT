from langchain_core.runnables.history import RunnableWithMessageHistory
from memory.redis_history import get_redis_history


def build_memory_runnable(chain):
    """
    给任意 chain / agent 加上记忆能力
    """

    return RunnableWithMessageHistory(
        chain,
        get_session_history=get_redis_history,

        input_messages_key="input",     # 你的输入字段
        history_messages_key="history", # 历史注入字段
        output_messages_key="output",   # 输出字段
    )