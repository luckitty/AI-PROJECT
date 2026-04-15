from core.llm import get_llm
import re
import time


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


def response_node(state):
    query = state["query"]
    system_prompt = state.get("system_prompt") or ""

    # .get(k, "") 在值为 None 时仍会得到 None，统一用 or "" 保证拼进 prompt 的是字符串。
    memory = state.get("memory_context") or ""
    rag = state.get("rag_context") or ""
    tool = state.get("tool_result") or ""
    print("response_node===========tool \n", tool, "\n")

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

    # 当前节点最终只需要完整答案，不消费 token 流，改为非流式可降低不必要开销。
    llm = get_llm(streaming=False)
    invoke_start_at = time.perf_counter()
    msg = llm.invoke(prompt)
    invoke_cost_seconds = time.perf_counter() - invoke_start_at
    answer = msg.content if hasattr(msg, "content") else str(msg)
    print(
        f"response_node===========llm_invoke耗时: {invoke_cost_seconds:.2f}s, "
        f"answer长度: {len(answer)}"
    )
    print("response_node===========answer_preview \n", answer, "\n")
  
    return {
        **state,
        "final_answer": answer
    }