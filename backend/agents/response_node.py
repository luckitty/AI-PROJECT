from core.llm import get_llm
import json
import time
from textwrap import dedent

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from interruptController.interrupt_manager import interrupt_manager

# 流式下发分片：进一步降低聚合阈值，减少首字与段落间等待，提升“更跟手”的体感。
# 这里取 8 字 + 6ms，兼顾连贯性与发送频率，避免出现明显“攒一包再吐”的停顿。
STREAM_BATCH_CHAR_SIZE = 8
STREAM_BATCH_MAX_WAIT_SECONDS = 0.006

RULES_NON_TRAVEL = dedent("""
    1) 紧扣用户当前问题作答；不要主动输出长篇旅游攻略，除非用户明确在问行程/攻略类内容。
    2) 若下方提供了本地知识库或工具结果，请优先采纳；不要编造未出现在上下文中的事实。
    3) 若上下文不足以回答，可如实说明并给出可行建议，但不要虚构数据。
""").strip()

TRAVEL_SECTIONS_NON_TRIP = dedent("""
    （本回合非旅游攻略类问题：无旅游专用缓存字段，请忽略行程类约束，按上文通用规则作答。）
""").strip()


def stream_plain_text_via_custom(writer, session_id: str, text: str) -> str:
    """
    不经过 LLM，按与终稿流式相同的块大小写入 custom，便于前端同一通道展示。
    中断时与 LLM 分支一致，统一提示「请求已中断」。
    """
    if not text:
        return ""
    parts: list[str] = []
    for start in range(0, len(text), STREAM_BATCH_CHAR_SIZE):
        if interrupt_manager.is_stopped(session_id):
            writer({"content": "请求已中断"})
            return "请求已中断"
        piece = text[start : start + STREAM_BATCH_CHAR_SIZE]
        writer({"content": piece})
        parts.append(piece)
    return "".join(parts)


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
    汇总上下文并产出最终回复。
    旅游攻略回合（need_rag）：正文仅在 travel 节点的单次 LLM 中生成，本节点只负责未流式时补推全文。
    非旅游回合：沿用通用 prompt + 单次终稿大模型流式输出。
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

    # 旅游：travel_context JSON 仅含 visible_answer_draft 与 visible_guide_streamed，不再走第二遍模型。
    if bool(state.get("need_rag")):
        travel_raw = state.get("travel_context") or ""
        try:
            payload = json.loads(travel_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        guide = str(payload.get("visible_answer_draft") or "").strip()
        streamed_before = bool(payload.get("visible_guide_streamed"))
        segment_to_stream = "" if streamed_before else guide
        invoke_start_at = time.perf_counter()
        if segment_to_stream:
            streamed = stream_plain_text_via_custom(writer, session_id, segment_to_stream)
            out = streamed if streamed == "请求已中断" else guide
        else:
            out = guide
        print(
            f"response_node===========旅游攻略直出（单次模型已在 travel 完成）耗时: "
            f"{time.perf_counter() - invoke_start_at:.2f}s, answer长度: {len(out)}"
        )
        return {**state, "final_answer": out}

    rules_block = RULES_NON_TRAVEL
    travel_sections = TRAVEL_SECTIONS_NON_TRIP

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
    print(
        f"response_node===========llm_stream耗时: {invoke_cost_seconds:.2f}s, "
        f"answer长度: {len(answer)} (非旅游)"
    )

    return {
        **state,
        "final_answer": answer,
    }
