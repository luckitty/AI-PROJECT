from core.llm import get_llm


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
    return {
        **state,
        "final_answer": answer
    }