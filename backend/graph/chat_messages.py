"""
ChatGPT 式短期记忆：checkpoint 里按序保存 user/assistant；读入模型时带滑动窗口内的多轮。

注意：窗口过大时 planner / 工具 / 终稿都会重复吃大量 token，首包与总耗时会明显变慢；
因此按调用场景传入 max_messages、max_chars，而不是一律用超大窗口。
"""
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# format_conversation_history_block 未显式传 max_messages 时的默认条数上限（偏保守，避免字符串版历史过长）。
DEFAULT_MAX_MESSAGES_FOR_TEXT_BLOCK = 22

# 终稿 Chat 接口默认最多带最近若干条（user/ai 各算一条），控制上下文体积。
DEFAULT_MAX_MESSAGES_FOR_CHAT_MODEL = 18

# 格式化成一大段「用户：/助手：」正文时，默认尾部字数上限。
CONVERSATION_BLOCK_MAX_CHARS = 12000


def message_text_content(msg: BaseMessage) -> str:
    """从单条消息取出可见字符串正文（兼容 content 为块列表的多模态形态）。"""
    c = getattr(msg, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        texts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text") or "")
        return "".join(texts).strip()
    return str(c).strip()


def trim_dialog_messages(
    messages: Sequence[BaseMessage] | None,
    max_messages: int = DEFAULT_MAX_MESSAGES_FOR_TEXT_BLOCK,
) -> list[BaseMessage]:
    """
    取尾部滑动窗口：最多 max_messages 条；截断后若以助手开头则去掉，
    保证传给模型的序列尽量以用户消息开头。
    """
    if not messages:
        return []
    msgs = list(messages)
    cap = max(2, int(max_messages))
    if len(msgs) > cap:
        msgs = msgs[-cap:]
    while msgs and isinstance(msgs[0], AIMessage):
        msgs.pop(0)
    return msgs


def dialog_messages_for_chat_model(
    messages: Sequence[BaseMessage] | None,
    max_messages: int = DEFAULT_MAX_MESSAGES_FOR_CHAT_MODEL,
) -> list[BaseMessage]:
    """
    截断窗口后生成可直接传入 Chat 接口的多轮 HumanMessage/AIMessage（正文为纯字符串）。
    与常见 OpenAI 风格 messages 数组一致：system 由调用方单独前置。
    """
    out: list[BaseMessage] = []
    for m in trim_dialog_messages(messages, max_messages=max_messages):
        text = message_text_content(m)
        if isinstance(m, HumanMessage):
            out.append(HumanMessage(content=text))
        elif isinstance(m, AIMessage):
            out.append(AIMessage(content=text))
    return out


def format_conversation_for_prompt(
    messages: Sequence[BaseMessage] | None,
    conversation_summary: str | None,
    *,
    max_chars: int,
    max_messages: int,
) -> str:
    """
    供 planner / tool / 攻略等「整段字符串 prompt」使用：可选滚动摘要 + 滑动窗口内原文。
    摘要对应已从 checkpoint 裁剪掉的更早轮次，避免与窗口内最近几句重复堆叠。
    """
    chunks: list[str] = []
    summary_text = (conversation_summary or "").strip()
    if summary_text:
        chunks.append(f"【更早对话摘要】\n{summary_text}")
    body = format_conversation_history_block(
        messages,
        max_chars=max_chars,
        max_messages=max_messages,
    ).strip()
    if body:
        chunks.append(body)
    return "\n\n".join(chunks)


def format_conversation_history_block(
    messages: Sequence[BaseMessage] | None,
    max_chars: int = CONVERSATION_BLOCK_MAX_CHARS,
    max_messages: int = DEFAULT_MAX_MESSAGES_FOR_TEXT_BLOCK,
) -> str:
    """
    将窗口内多轮对话格式化为「用户：…\\n助手：…」文本块，供单条字符串 prompt 的路由/工具/攻略节点使用。
    先按 max_messages 截断，再按 max_chars 从尾部保留正文，更早轮次省略。
    """
    lines: list[str] = []
    for m in trim_dialog_messages(messages, max_messages=max_messages):
        t = message_text_content(m)
        if not t:
            continue
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{t}")
        elif isinstance(m, AIMessage):
            lines.append(f"助手：{t}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "…（更早对话已省略）…\n" + text[-max_chars:]
    return text


def last_assistant_text(messages: Sequence[BaseMessage] | None) -> str:
    """整条会话里最近一条助手正文（通常即本轮刚生成的回复，供 HTTP 封装取 reply）。"""
    if not messages:
        return ""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return message_text_content(messages[i])
    return ""
