from core.llm import get_llm
import json
import time
from textwrap import dedent

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from interruptController.interrupt_manager import interrupt_manager

from graph.chat_messages import (
    dialog_messages_for_chat_model,
    format_conversation_for_prompt,
)
from tools.search_travel_tool import search_travel

# 流式下发分片：进一步降低聚合阈值，减少首字与段落间等待，提升“更跟手”的体感。
# 这里取 8 字 + 6ms，兼顾连贯性与发送频率，避免出现明显“攒一包再吐”的停顿。
STREAM_BATCH_CHAR_SIZE = 8
STREAM_BATCH_MAX_WAIT_SECONDS = 0.006

RULES_NON_TRAVEL = dedent("""
    1) 紧扣当前对话与用户最新意图作答；不要主动输出长篇旅游攻略，除非用户明确在问行程/攻略类内容。
    2) 你必须结合「多轮对话消息」理解指代（上面、之前、刚才等），不要说看不到上文。
    3) 若系统消息里提供了本地知识库或工具结果，请优先采纳；不要编造未出现在上下文中的事实。
    4) 若上下文不足以回答，可如实说明并给出可行建议，但不要虚构数据。
""").strip()

# planner 判定 need_history=false 时使用：未注入多轮消息，避免模型假装读过上文。
RULES_NON_TRAVEL_NO_HISTORY = dedent("""
    1) 紧扣用户当前问题作答；不要主动输出长篇旅游攻略，除非用户明确在问行程/攻略类内容。
    2) 本回合系统未向模型提供更早的多轮对话：请勿臆测用户在上文说过什么；若问题明显依赖前文而信息不足，如实说明并请用户补充要点。
    3) 若系统消息里提供了本地知识库或工具结果，请优先采纳；不要编造未出现在上下文中的事实。
    4) 若上下文不足以回答，可如实说明并给出可行建议，但不要虚构数据。
""").strip()


def stream_plain_text_via_custom(writer, session_id: str, text: str) -> str:
    """不经过 LLM，按与终稿流式相同的块大小写入 custom；中断时返回「请求已中断」。"""
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
    """从流式 chunk 取出可见文本 delta，供 get_stream_writer 写入。"""
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
    汇总上下文并产出最终回复：旅游回合走 search_travel；否则 SystemMessage + 多轮消息流式 LLM。
    need_history=false 时不注入多轮与滚动摘要。
    """
    query = state["query"]
    session_id = state.get("session_id") or ""
    system_prompt = state.get("system_prompt") or ""
    # planner 的 need_history：false 时终稿不带多轮与滚动摘要，仅当前句 + 记忆/工具等。
    want_history = bool(state.get("need_history", True))

    # 攻略分支：按需把窗口内多轮对话格式化成一块正文（与 planner/tool 同源逻辑）。
    conversation_history = ""
    if want_history:
        conversation_history = format_conversation_for_prompt(
            state.get("messages"),
            state.get("conversation_summary"),
            max_chars=4800,
            max_messages=14,
        ).strip()

    # .get(k, "") 在值为 None 时仍会得到 None，统一用 or "" 保证拼进 prompt 的是字符串。
    memory = state.get("memory_context") or ""
    tool = state.get("tool_result") or ""
    writer = get_stream_writer()
    # 进入最终回答节点前再检查协作 stop（断连 /api/chat/stop），避免无意义调用大模型。
    if interrupt_manager.is_stopped(session_id):
        interrupted_text = "请求已中断"
        writer({"content": interrupted_text})
        return {
            **state,
            "human_halt": True,
            "messages": [AIMessage(content=interrupted_text)],
        }

    # 旅游攻略统一入口：search_travel 内部等价 stream_travel_guide_llm + 素材组装；rag_context 可为空（非六城或未召回）。
    if bool(state.get("travel_itinerary_in_response")) or bool(state.get("need_rag")):
        stream_writer = writer if bool(state.get("stream_sink_active")) else None
        invoke_start_at = time.perf_counter()
        travel_raw = search_travel(
            query,
            state.get("rag_context") or [],
            stream_writer,
            conversation_history=(
                conversation_history if (want_history and conversation_history) else None
            ),
        )
        try:
            payload = json.loads(travel_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        guide = str(payload.get("visible_answer_draft") or "").strip()
        streamed_before = bool(payload.get("visible_guide_streamed"))
        segment_to_stream = "" if streamed_before else guide
        if segment_to_stream:
            streamed = stream_plain_text_via_custom(writer, session_id, segment_to_stream)
            coerced_stop = streamed == "请求已中断"
            out = streamed if coerced_stop else guide
        else:
            coerced_stop = False
            out = guide
        print(
            "response_node===========旅游攻略（search_travel 单次模型）总耗时: "
            f"{time.perf_counter() - invoke_start_at:.2f}s, answer长度: {len(out)}"
        )
        patch: dict = {"messages": [AIMessage(content=out)]}
        if coerced_stop:
            patch["human_halt"] = True
        return {**state, **patch}

    rules_block = RULES_NON_TRAVEL if want_history else RULES_NON_TRAVEL_NO_HISTORY
    if want_history:
        summary_slot = (state.get("conversation_summary") or "").strip()
        summary_block = (
            f"更早轮次摘要（窗口外已压缩）：\n{summary_slot}\n"
            if summary_slot
            else "更早轮次摘要（窗口外已压缩）：\n（无）\n"
        )
    else:
        summary_block = "更早轮次摘要与多轮对话：本回合未注入（need_history=false）。\n"

    system_block = f"""{system_prompt}

回答规则（必须遵守）：
{rules_block}

{summary_block}
用户记忆：
{memory}

旅游缓存信息：
（本回合非旅游攻略类问题：无旅游专用缓存字段，请忽略行程类约束，按上文通用规则作答。）

实时信息、工具结果：
{tool}

请基于系统消息与随后的用户消息给出清晰、有用的回答。"""

    # 一条 SystemMessage 承载系统人设与本回合工具/记忆；其后为滑动窗口内的多轮 user/assistant（由 want_history 决定）。
    # 终稿对话轮数：need_history 为 false 时只传当前用户句，避免 token 与注意力干扰。
    if want_history:
        history_lc = dialog_messages_for_chat_model(
            state.get("messages"), max_messages=18
        )
    else:
        history_lc = [HumanMessage(content=query)]
    lc_messages = [SystemMessage(content=system_block)] + history_lc

    # 流式：Chat 模型接收 OpenAI 风格 messages；get_stream_writer 写入 custom 供 SSE。
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

    for chunk in llm.stream(lc_messages, config=config):
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

    out_state = {**state, "messages": [AIMessage(content=answer)]}
    if interrupted:
        out_state["human_halt"] = True
    return out_state
