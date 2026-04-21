from core.llm import get_llm
import json
import time

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer


def delta_text_from_stream_chunk(chunk) -> str:
    """
    从 LangChain 流式产出的 AIMessageChunk 取出本段可见文本（一般为 delta）。
    与 chat 接口推送字段一致，供 get_stream_writer 写入。
    """
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        texts = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text") or "")
        return "".join(texts)
    return str(c)


def response_node(state, config: RunnableConfig):
    query = state["query"]
    system_prompt = state.get("system_prompt") or ""
   
    # .get(k, "") 在值为 None 时仍会得到 None，统一用 or "" 保证拼进 prompt 的是字符串。
    memory = state.get("memory_context") or ""
    rag = state.get("rag_context") or ""
    tool = state.get("tool_result") or ""
    travel = state.get("travel_context") or ""
    # 严格模式：travel_context 必须是新版 JSON，不做任何旧格式兼容或降级。
    maybe_payload = json.loads(travel)
   
    itinerary_structured = str(maybe_payload.get("itinerary_structured") or "").strip()
    response_material = str(maybe_payload.get("response_material") or "").strip()
    writer = get_stream_writer()
    if not itinerary_structured or not response_material:
        error_text = (
            "旅游结构化数据缺失：travel_context 必须包含 itinerary_structured 与 response_material。"
        )
        writer({"content": error_text})
        return {
            **state,
            "final_answer": error_text
        }

    print("response_node===========tool \n", tool[:200], "\n")

    prompt = f"""
        系统指令（必须优先遵守）：
        {system_prompt}

        回答规则（必须遵守）：
        1) 你必须先以“结构化每日行程”为主骨架生成最终旅游攻略，保持每天的顺序和核心安排，不要打乱天数和顺序。
        2) “旅游规则与素材”中的结构要求必须遵守，不能退化成泛泛介绍。
        3) 可以对结构化行程做轻量补充说明（如节奏提醒、餐饮推荐理由），但不能改变原始行程顺序。
        4) 如果工具结果里已有内容，禁止输出“工具不可用/无法直接生成”之类措辞，直接给出可执行路线。

        请基于以下信息回答用户问题：

        用户问题：
        {query}

        用户记忆：
        {memory}

        本地知识库信息：
        {rag}

        旅游规划初稿（优先作为最终答案基础）：
        {itinerary_structured}

        旅游规则与素材（由 RAG + build_itinerary_format_instruction 汇总）：
        {response_material}

        实时信息、工具结果：
        {tool}

        请给出清晰、有用的回答：
        """

    # 流式必须用 .stream + config 传入 RunnableConfig；再用 get_stream_writer 写入 custom，
    # API 层 stream_mode="custom" 才能稳定收到小增量（messages 模式在函数节点里依赖回调，易出现整段或卡顿）。
    llm = get_llm(streaming=True)
    invoke_start_at = time.perf_counter()
    parts: list[str] = []
    for chunk in llm.stream(prompt, config=config):
        piece = delta_text_from_stream_chunk(chunk)
        if not piece:
            continue
        parts.append(piece)
        writer({"content": piece})
    answer = "".join(parts)
    invoke_cost_seconds = time.perf_counter() - invoke_start_at
    print(
        f"response_node===========llm_stream耗时: {invoke_cost_seconds:.2f}s, "
        f"answer长度: {len(answer)}"
    )
  
    return {
        **state,
        "final_answer": answer
    }