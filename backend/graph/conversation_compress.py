"""
多轮对话滚动压缩：checkpoint 内 messages 过长时，将最早一批轮次摘要进 conversation_summary，
并用 RemoveMessage 从状态中删除对应消息，降低 Redis 检查点体积与各节点 prompt 冗余。

触发条件与 KEEP_RAW_MESSAGE_COUNT、MIN_OVERFLOW_MESSAGES 对齐，避免偶发多一条就调模型。
"""
import logging
import time

from langchain_core.messages import RemoveMessage
from langgraph.constants import TAG_NOSTREAM

from core.llm import get_llm
from graph.chat_messages import format_conversation_history_block
from interruptController.interrupt_manager import interrupt_manager

logger = logging.getLogger(__name__)

# checkpoint 中保留的最近消息条数（user/assistant 各算一条），与 response 窗口同量级。
KEEP_RAW_MESSAGE_COUNT = 14
# 相对 KEEP_RAW 至少溢出这么多条才触发折叠，避免「多一条」就调用摘要模型。
MIN_OVERFLOW_MESSAGES = 6
# 单次送入摘要模型的待折叠正文上限（字符），防止首轮超长会话撑爆路由上下文。
FOLD_BLOCK_MAX_CHARS = 12000


def merge_conversation_summary(previous_summary: str, fold_dialog_block: str) -> str:
    """
    将已有摘要与一段「待折叠」对话正文合并为新的连贯摘要。
    输出纯文本，保留关键意图、实体与约束；尽量不重复啰嗦。
    """
    prev = (previous_summary or "").strip()
    fold = (fold_dialog_block or "").strip()
    if not fold:
        return prev
    prev_section = f"【已有摘要】\n{prev}\n\n" if prev else ""
    prompt = f"""你是对话压缩助手。{prev_section}【待并入的多轮对话片段】
{fold}

任务：合并为一段中文摘要，供后续模型理解会话背景。
要求：
1) 保留：用户目标、关键实体（地名/景点/时间/人数等约束）、已确认的事实、未完成的请求。
2) 淡化或省略：寒暄、重复措辞、与后续决策无关的细碎闲聊。
3) 不要编造对话里不存在的内容；不要用 Markdown 标题；不要输出「摘要：」之类前缀。
4) 总长度控制在约 1200 字以内。"""

    llm = get_llm(streaming=False, max_tokens=900, temperature=0)
    msg = llm.invoke(prompt, config={"tags": [TAG_NOSTREAM]})
    text = (msg.content if hasattr(msg, "content") else str(msg)) or ""
    return (text or "").strip()


def conversation_compress_node(state: dict) -> dict:
    """
    在 planner 之前运行：若 messages 显著超过保留窗口，则折叠最早溢出部分进摘要并删除对应消息。
    若协作 stop 已触发或无需折叠，返回空字典不写状态。
    """
    session_id = state.get("session_id") or ""
    if interrupt_manager.is_stopped(session_id):
        return {}

    messages = list(state.get("messages") or [])
    if len(messages) <= KEEP_RAW_MESSAGE_COUNT:
        return {}

    overflow = len(messages) - KEEP_RAW_MESSAGE_COUNT
    if overflow < MIN_OVERFLOW_MESSAGES:
        return {}

    to_fold = messages[:overflow]
    fold_block = format_conversation_history_block(
        to_fold,
        max_chars=FOLD_BLOCK_MAX_CHARS,
        max_messages=len(to_fold),
    ).strip()
    if not fold_block:
        return {}

    prev_summary = (state.get("conversation_summary") or "").strip()
    started = time.perf_counter()
    try:
        new_summary = merge_conversation_summary(prev_summary, fold_block)
    except Exception:
        logger.exception(
            "conversation_compress_node 摘要模型调用失败，本轮跳过折叠 session_id=%s",
            session_id,
        )
        return {}

    if not new_summary:
        return {}

    removals: list[RemoveMessage] = []
    for m in to_fold:
        mid = getattr(m, "id", None)
        if mid:
            removals.append(RemoveMessage(id=mid))

    if not removals:
        # 无稳定 id 时无法安全裁剪 checkpoint，仅写入摘要容易造成与原文重复，故放弃更新。
        logger.warning(
            "conversation_compress_node 待折叠消息缺少 id，跳过 RemoveMessage session_id=%s",
            session_id,
        )
        return {}

    cost = time.perf_counter() - started
    print(
        f"pipeline_timing===========conversation_compress 折叠 {len(to_fold)} 条消息 "
        f"耗时: {cost:.2f}s，摘要长度: {len(new_summary)}"
    )

    return {
        "conversation_summary": new_summary,
        "messages": removals,
    }
