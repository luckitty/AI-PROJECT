from core.llm import get_llm
import json
import time

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from interruptController.interrupt_manager import interrupt_manager

# 流式下发分片：进一步降低聚合阈值，减少首字与段落间等待，提升“更跟手”的体感。
# 这里取 8 字 + 6ms，兼顾连贯性与发送频率，避免出现明显“攒一包再吐”的停顿。
STREAM_BATCH_CHAR_SIZE = 8
STREAM_BATCH_MAX_WAIT_SECONDS = 0.006


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
    """
    汇总上下文并流式生成最终回复。
    仅当 planner 将本回合标为旅游攻略类（need_rag）时，才解析 travel_context 并注入行程与素材；
    否则同一套 prompt 模板里走「通用规则 + 无行程字段」说明，避免重复维护两套大段模板。
    """
    query = state["query"]
    session_id = state.get("session_id") or ""
    system_prompt = state.get("system_prompt") or ""

    # .get(k, "") 在值为 None 时仍会得到 None，统一用 or "" 保证拼进 prompt 的是字符串。
    memory = state.get("memory_context") or ""
    tool = state.get("tool_result") or ""
    writer = get_stream_writer()
    # 进入最终回答节点前再检查一次中断，避免无意义调用大模型。
    if interrupt_manager.is_stopped(session_id):
        interrupted_text = "请求已中断"
        writer({"content": interrupted_text})
        return {
            **state,
            "is_interrupted": True,
            "final_answer": interrupted_text,
        }

    is_travel = bool(state.get("need_rag"))
    itinerary_structured = ""
    visible_answer_draft = ""
    response_material = ""

    # 旅游类解析 travel_context：
    # - response_material 必须有（规则与素材来源）
    # - itinerary_structured 可为空（表示不走前置草稿，直接最终流式生成）
    if is_travel:
        travel = state.get("travel_context") or ""
        try:
            maybe_payload = json.loads(travel)
        except (json.JSONDecodeError, TypeError, ValueError):
            error_text = "旅游结构化数据解析失败：travel_context 须为合法 JSON。"
            writer({"content": error_text})
            return {
                **state,
                "final_answer": error_text,
            }

        itinerary_structured = str(maybe_payload.get("itinerary_structured") or "").strip()
        visible_answer_draft = str(maybe_payload.get("visible_answer_draft") or "").strip()
        # print("response_node===========itinerary_structured \n", itinerary_structured, "\n")
        response_material = str(maybe_payload.get("response_material") or "").strip()
        # print("response_node===========response_material \n", response_material, "\n")
        if not response_material:
            error_text = (
                "旅游结构化数据缺失：travel_context 必须包含 response_material。"
            )
            writer({"content": error_text})
            return {
                **state,
                "final_answer": error_text,
            }

    # 同一套模板内用分支填「回答规则」与是否附带行程块，避免两份 prompt 分叉。
    if is_travel:
        if itinerary_structured:
            rules_block = """
            【硬约束（最高优先级）】
            1) 你必须先以“结构化每日行程”为主骨架生成最终旅游攻略，保持每天的顺序和核心安排，不要打乱天数、顺序和时间。
            2) 可以对结构化行程做轻量补充说明（如节奏提醒、餐饮推荐理由），但不能改变原始行程顺序。
            3) 如果工具结果里已有内容，禁止输出“工具不可用/无法直接生成”之类措辞，直接给出可执行路线。
            4) 当前结构化行程里的 days[].transports 已由 amap_node 基于高德路线补全；你在组织最终文案时必须优先使用这些交通信息，不要忽略。
            5) 每一段交通直接使用 transports 里的 transport 文本，无需解释。
            6) 如果提供了“旅游文案草稿”，请优先沿用其文风与表达，再按结构化行程修正细节。
            7) “旅游规则与素材”中的【结构约束】必须遵守；其中【风格建议】仅在不与硬约束冲突时采纳。
            """
            travel_sections = f"""
            旅游规划初稿（优先作为最终答案基础）：
            {itinerary_structured}

            旅游文案草稿（同一次模型调用产出，可用于文风与表达参考）：
            {visible_answer_draft or "本轮未提供"}

            旅游规则与素材（由 RAG + build_itinerary_format_instruction 汇总）：
            {response_material}
            """
        else:
            # 无结构化草稿时，直接按素材一次生成最终答案，减少前置等待。
            rules_block = """
            【硬约束（最高优先级）】
            1) 你必须严格遵守“旅游规则与素材”里的【结构约束】，直接产出可执行攻略。
            2) 禁止输出“工具不可用/无法直接生成”之类措辞，直接给出路线与建议。
            3) 输出时优先保证路线顺序、时间安排与可执行性，不要只给泛化清单。
            4) “旅游规则与素材”里的【风格建议】仅在不与硬约束冲突时采纳。
            """
            travel_sections = f"""
            旅游规划初稿：本轮未提供（请直接基于下方素材生成最终答案）

            旅游规则与素材（由 RAG + build_itinerary_format_instruction 汇总）：
            {response_material}
            """
    else:
        rules_block = """
        1) 紧扣用户当前问题作答；不要主动输出长篇旅游攻略，除非用户明确在问行程/攻略类内容。
        2) 若下方提供了本地知识库或工具结果，请优先采纳；不要编造未出现在上下文中的事实。
        3) 若上下文不足以回答，可如实说明并给出可行建议，但不要虚构数据。
        """
        travel_sections = """
        （本回合非旅游攻略类问题：无「旅游规划初稿 / 旅游规则与素材」字段，请忽略行程类约束，按上文通用规则作答。）
        """

    prompt = f"""
        系统指令（必须优先遵守）：
        {system_prompt}

        回答规则（必须遵守）：
        {rules_block}

        请基于以下信息回答用户问题：

        用户问题：
        {query}

        用户记忆：
        {memory}

        旅游缓存信息：
        {travel_sections}

        实时信息、工具结果：
        {tool}

        请给出清晰、有用的回答：
        """

    # 流式必须用 .stream + config 传入 RunnableConfig；再用 get_stream_writer 写入 custom，
    # API 层 stream_mode="custom" 才能稳定收到小增量（messages 模式在函数节点里依赖回调，易出现整段或卡顿）。
    llm = get_llm(streaming=True)
    invoke_start_at = time.perf_counter()
    parts: list[str] = []
    pending_pieces: list[str] = []
    last_flush_at = time.perf_counter()
    interrupted = False

    # 流式聚合发送：避免逐 token 推送过碎，按「字数阈值 + 最大等待时长」双条件刷新。
    # 这样每次下发字符更多、发送频率更稳定，同时不会因为过度聚合导致首字明显变慢。
    def flush_pending_stream_text(force: bool = False):
        nonlocal last_flush_at
        if not pending_pieces:
            return
        now = time.perf_counter()
        pending_text = "".join(pending_pieces)
        should_flush = force
        if not should_flush and len(pending_text) >= STREAM_BATCH_CHAR_SIZE:
            should_flush = True
        if not should_flush and (now - last_flush_at) >= STREAM_BATCH_MAX_WAIT_SECONDS:
            should_flush = True
        if not should_flush:
            return
        writer({"content": pending_text})
        pending_pieces.clear()
        last_flush_at = now

    for chunk in llm.stream(prompt, config=config):
        # 流式生成过程中按 chunk 粒度检查 stop 标记，尽快终止后续 token 生成。
        if interrupt_manager.is_stopped(session_id):
            interrupted = True
            break
        piece = delta_text_from_stream_chunk(chunk)
        if not piece:
            flush_pending_stream_text()
            continue
        parts.append(piece)
        pending_pieces.append(piece)
        flush_pending_stream_text()
    flush_pending_stream_text(force=True)
    answer = "".join(parts)
    if interrupted:
        answer = "请求已中断"
        writer({"content": answer})
    invoke_cost_seconds = time.perf_counter() - invoke_start_at
    mode_tag = "旅游" if is_travel else "非旅游"
    print(
        f"response_node===========llm_stream耗时: {invoke_cost_seconds:.2f}s, "
        f"answer长度: {len(answer)} ({mode_tag})"
    )

    return {
        **state,
        "final_answer": answer,
    }
