"""
聊天层长期记忆：何时检索、何时写入、如何拼接上下文。

与 ``long_memory.py`` 的向量存取分离，本模块负责规则 + LLM 判定与异步写入编排。
"""
import re
import threading

from core.llm import get_llm

from memory.long_memory import save_long_memory, search_long_memory


def normalize_yes_no(content: str) -> str:
    """
    把模型输出归一化成 YES / NO / UNKNOWN，避免格式波动导致误判。
    """
    text = (content or "").strip().upper()
    if not text:
        return "UNKNOWN"
    if text == "YES" or re.search(r"\bYES\b", text):
        return "YES"
    if text == "NO" or re.search(r"\bNO\b", text):
        return "NO"
    return "UNKNOWN"


def hit_memory_query_rule(user_input: str) -> bool:
    """
    规则层预判：明显“回忆我/偏好/历史设定”类问题直接命中。
    说明：这类请求通常对长期记忆强依赖，优先检索可降低漏检率。
    """
    text = (user_input or "").strip().lower()
    if not text:
        return False

    # 明确记忆触发词：用户在问“我之前说过什么/我的偏好是什么”等。
    trigger_patterns = [
        r"我(之?前|以前|刚才).*(说过|提过|告诉过)",
        r"记得我",
        r"你还记得",
        r"我的(偏好|习惯|口味|背景|信息|资料)",
        r"(根据|结合).*(我的|之前).*(记忆|信息|偏好|习惯)",
        r"(以后|今后).*(都|请).*(用|按)",
    ]
    return any(re.search(pattern, text) for pattern in trigger_patterns)


def skip_memory_query_rule(user_input: str) -> bool:
    """
    规则层快速跳过：明显通用问答或一次性任务，不需要长期记忆。
    """
    text = (user_input or "").strip().lower()
    if not text:
        return True
    skip_patterns = [
        r"(天气|温度|下雨|空气质量)",
        r"(股票|股价|行情)",
        r"(写|生成|解释|优化).*(代码|脚本|函数)",
        r"(翻译|总结|润色|改写)",
        r"(时间|日期|星期几)",
    ]
    return any(re.search(pattern, text) for pattern in skip_patterns)


def should_search_long_memory_by_agent(user_input: str) -> bool:
    """
    用判定 Agent 决定本轮是否需要长期记忆检索。
    """
    classify_prompt = (
        "你是长期记忆检索判定 Agent。判断当前问题是否应该搜索用户历史偏好。\n"
        "依赖用户历史偏好、过往设定、曾说过的信息 -> YES。\n"
        "通用知识、实时信息、一次性任务、代码生成 -> NO。\n"
        "不确定时输出 NO。\n"
        "只输出 YES 或 NO，不要输出其他内容。\n\n"
        f"当前问题：{user_input}"
    )
    response = get_llm().invoke(classify_prompt)
    result = getattr(response, "content", "")
    decision = normalize_yes_no(result)
    print("should_search_long_memory_by_agent===========raw \n", decision, "\n")
    return decision == "YES"


def should_search_long_memory(user_input: str) -> bool:
    """
    组合判定是否需要查长期记忆：
    1) 规则命中直接查；
    2) 明显无记忆依赖的问题直接跳过；
    3) 其余交给判定 Agent 决策。
    """
    if hit_memory_query_rule(user_input):
        return True
    if skip_memory_query_rule(user_input):
        return False
    return should_search_long_memory_by_agent(user_input)


def build_memory_context(user_input: str, user_id: str | None) -> str:
    """
    按需构建长期记忆上下文：
    - 仅在“应该检索长期记忆”时执行向量检索；
    - 无命中时返回空字符串，调用方可直接按 truthy 判断。
    """
    if not user_id or not (user_input or "").strip():
        return ""
    should_search = should_search_long_memory(user_input)
    if not should_search:
        return ""
    print("build_memory_context===========should_search \n", should_search, "\n")

    docs = search_long_memory(user_input, user_id, k=4)
    if not docs:
        return ""
    memory_texts = [doc.page_content for doc in docs if (doc.page_content or "").strip()]
    print("memory_texts=========== \n", memory_texts, "\n")
    return "\n".join(memory_texts)


def should_save_long_memory(user_text: str) -> bool:
    """
    使用 LLM 判断当前输入是否应写入长期记忆。
    目标：减少关键词规则误判。
    """
    classify_prompt = (
        "你是长期记忆分类器。判断【用户原话】是否适合存为“长期稳定偏好/个人背景/用户设定指令任务”。\n"
        "可存示例：我喜欢王安宇、我不吃香菜、以后都用中文简洁回答。\n"
        "不可存示例：我喜欢谁？、今天天气如何、这题怎么做。\n"
        "只输出 YES 或 NO，不要输出其他内容。\n\n"
        f"用户原话：{user_text}"
    )

    response = get_llm().invoke(classify_prompt)
    result = getattr(response, "content", "")
    print("should_save_long_memory===========raw \n", result, "\n\n")
    if normalize_yes_no(result) == "YES":
        return True
    return False


def start_long_memory_save_task(user_text: str, user_id: str | None) -> None:
    """
    后台异步保存长期记忆，避免阻塞主请求（尤其是 SSE 的 [DONE] 返回）。
    """
    if not user_id or not (user_text or "").strip():
        return

    def run_save() -> None:
        try:
            if should_save_long_memory(user_text):
                save_long_memory(user_text, user_id)
        except Exception as exc:
            # 后台任务失败只记录日志，不影响用户主流程响应。
            print("start_long_memory_save_task===========error \n", str(exc), "\n\n")

    save_thread = threading.Thread(target=run_save, daemon=True)
    save_thread.start()
