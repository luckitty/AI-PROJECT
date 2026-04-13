from memory.long_memory_guard import start_long_memory_save_task
from profiles.user_profile import update_user_profile_from_text

def save_memory_node(state):
    user_id = state["user_id"]
    query = state["query"]

    # 画像更新与长期记忆入库并行触发：画像更偏结构化标签，长期记忆保留原始语义文本。
    update_user_profile_from_text(user_id, query)
    start_long_memory_save_task(query, user_id)

    return {
        **state,
    }