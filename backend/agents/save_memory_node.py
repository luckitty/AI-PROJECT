from memory.long_memory_guard import start_long_memory_save_task
from profiles.user_profile import update_user_profile_from_text

def save_memory_node(state):
    user_id = state["user_id"]
    query = state["query"]

    # 画像更新失败不应影响主回复链路，避免在正文输出后再向前端抛异常提示。
    try:
        update_user_profile_from_text(user_id, query)
    except Exception as exc:
        print("save_memory_node===========update_user_profile_from_text_error \n", str(exc), "\n\n")

    # 长期记忆保存本身就是后台线程，不阻塞当前图执行。
    start_long_memory_save_task(query, user_id)

    return {
        **state,
    }