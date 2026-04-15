from core.llm import get_llm
import re


def should_enforce_travel_format(tool_text: str) -> bool:
    """判断是否需要启用旅游结构化输出校验。"""
    text = (tool_text or "").strip()
    return "【行程输出要求（必须遵守）】" in text


def is_travel_answer_compliant(answer_text: str) -> bool:
    """检查旅游回答是否包含关键结构字段，避免退化成普通攻略文案。"""
    text = (answer_text or "").strip()
    if not text:
        return False
    has_day = re.search(r"(第\s*1\s*天路线|Day\s*1)", text) is not None
    required_keywords = ["早餐", "午餐", "晚餐", "出片", "时间合理性说明"]
    has_keywords = all(keyword in text for keyword in required_keywords)
    return has_day and has_keywords


def has_tool_unavailable_excuse(answer_text: str) -> bool:
    """检测回答是否出现“工具不可用”类推脱措辞。"""
    text = (answer_text or "").strip()
    if not text:
        return False
    pattern = r"(工具暂时不可用|工具不可用|无法直接为您生成|无法直接生成)"
    return re.search(pattern, text) is not None


def rewrite_travel_answer(query: str, tool_text: str, draft_answer: str) -> str:
    """当旅游回答不合规时，基于工具材料进行一次强制结构化重写。"""
    rewrite_prompt = f"""
你是旅游行程编排助手。请严格按工具给出的“行程输出要求（必须遵守）”重写答案。

用户问题：
{query}

工具结果（含格式约束）：
{tool_text}

当前草稿（不合规，不能直接输出）：
{draft_answer}

重写规则：
1) 必须按“第X天路线”输出完整日程，不能写成泛泛介绍。
2) 每天必须包含：早餐、上午安排、午餐策略、下午安排、晚餐、出片点、时间合理性说明。
3) 如果用户没给天数，先给建议天数和理由，再展开每天路线。
4) 直接输出最终可执行版本，不要解释改写过程。
"""
    llm = get_llm(streaming=True)
    msg = llm.invoke(rewrite_prompt)
    return msg.content if hasattr(msg, "content") else str(msg)


def response_node(state):
    query = state["query"]
    system_prompt = state.get("system_prompt") or ""

    # .get(k, "") 在值为 None 时仍会得到 None，统一用 or "" 保证拼进 prompt 的是字符串。
    memory = state.get("memory_context") or ""
    rag = state.get("rag_context") or ""
    tool = state.get("tool_result") or ""

    prompt = f"""
        系统指令（必须优先遵守）：
        {system_prompt}

        回答规则（必须遵守）：
        1) 如果“实时信息、工具结果”里出现“【行程输出要求（必须遵守）】”，你必须严格按照该要求和结构输出，不能改写为普通概述。
        2) 当工具结果是旅游攻略材料时，优先依据工具结果组织回答，不要忽略其中的格式约束。
        3) 如果工具结果里已有内容，禁止输出“工具不可用/无法直接生成”之类措辞，直接给出可执行路线。

        请基于以下信息回答用户问题：

        用户问题：
        {query}

        用户记忆：
        {memory}

        本地知识库信息：
        {rag}

        实时信息、工具结果：
        {tool}

        请给出清晰、有用的回答：
        """

    llm = get_llm(streaming=True)
    msg = llm.invoke(prompt)
    answer = msg.content if hasattr(msg, "content") else str(msg)
    print("response_node===========answer \n", answer, "\n")
    # 旅游攻略场景做一次结构化兜底，确保输出不会退化为普通介绍文案。
    # if should_enforce_travel_format(tool) and (
    #     (not is_travel_answer_compliant(answer)) or has_tool_unavailable_excuse(answer)
    # ):
    #     answer = rewrite_travel_answer(query, tool, answer)
    return {
        **state,
        "final_answer": answer
    }