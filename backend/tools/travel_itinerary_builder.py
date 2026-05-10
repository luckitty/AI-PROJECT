import time
from typing import Any, Callable, List

from core.llm import get_llm

# LangGraph custom 通道写入器：由 response 节点经 search_travel 传入，把攻略正文增量推到 SSE。
StreamWriter = Callable[[dict[str, Any]], None]


def build_itinerary_format_instruction() -> str:
    """
    旅游攻略唯一指令文案：角色、结构约束、输出形态均在同一字符串内，供单次 LLM 调用整段作为系统侧规则。
    """
    return (
        "你现在是“旅游路线编排助手”，请严格执行以下规则，不要输出与规则无关的泛化攻略。\n\n"
        "【输出形态】\n"
        "- 直接输出一篇给用户看的旅游攻略正文（自然语言，可按天分段）。\n"
        "- 禁止输出 JSON；不要用 markdown 代码块包裹全文。\n\n"
        "【行程输出要求（必须遵守）】\n"
        "根据城市规模、景点密度生成游玩天数 \n"
        "推荐 1–2 个适合住宿的区域或酒店类型，可给出酒店名称与价格区间（基于常识或素材，勿编造精确报价）。\n\n"
        "【篇幅（必须遵守）】\n"
        "- 全文精炼：少寒暄少重复，每个景点一两句话写玩法+建议时长即可；不要堆砌长段科普。\n\n"
        "【全局要求】\n"
        "- 整个路线中餐厅不重复，景点不重复，每日景点安排尽量顺路。\n"
        "- 行程最后用两三句话总结，并推荐几个未提到的餐厅即可（不必展开）。\n"
        "- 语言稍微幽默风趣，不要过于正式；可适当使用表情或图标提升可读性。\n\n"
        "【每日结构（每一天必须遵守）】\n"
        "游玩安排：推荐 2–3 个景点并写明建议游玩时长。景点安排要合理，不要东一个西一个\n"
        "美食推荐：一定要推荐景点附近的美食，距离景点一定不能远。菜系不能重复（例如已推荐烤鸭则另一顿避免重复同一菜系）。\n"
    )


def delta_text_from_llm_chunk(chunk) -> str:
    """
    从 LLM 流式 chunk 取出本段文本，与 response_node 行为一致，供 travel 节点推流。
    """
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        texts: List[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text") or "")
        return "".join(texts)
    return str(c)


def stream_travel_guide_llm(
    query: str,
    body_material: str,
    stream_writer: StreamWriter | None,
    conversation_history: str | None = None,
) -> str:
    """
    旅游攻略唯一一次大模型调用：指令仅用 build_itinerary_format_instruction，
    附多轮对话节选 + 当前问句 + 检索素材；有 stream_writer 时流式写出 token。
    """
    instruction = build_itinerary_format_instruction()
    ch = (conversation_history or "").strip()
    history_block = (
        f"\n【本轮为止的多轮对话（理解偏好与指代；不要说看不到上文）】\n{ch}\n"
        if ch
        else ""
    )
    prompt = f"""{instruction}
{history_block}
用户当前问题：
{query}

检索素材（优先依据下列事实编排；不够再合理补充，勿编造素材中不存在的关键事实）：
{body_material}
"""
    # max_tokens 压低可明显缩短流式结束时间；与上文「精炼」约束一致，避免模型写到上限。
    llm = get_llm(streaming=bool(stream_writer), temperature=0.45, max_tokens=1100)
    parts: List[str] = []
    # 攻略正文是唯一长输出阶段；记录耗时与素材体量便于对照「首字晚」是否卡在模型。
    material_chars = len(body_material or "")
    travel_llm_started_at = time.perf_counter()
    if stream_writer:
        for chunk in llm.stream(prompt):
            piece = delta_text_from_llm_chunk(chunk)
            if piece:
                stream_writer({"content": piece})
            parts.append(piece)
    else:
        raw = llm.invoke(prompt)
        parts.append(str(getattr(raw, "content", "") or ""))
    travel_llm_seconds = time.perf_counter() - travel_llm_started_at
    answer_text = "".join(parts).strip()
    print(
        "pipeline_timing===========travel 攻略正文LLM "
        f"流式={bool(stream_writer)} 素材字符数={material_chars} "
        f"invoke/stream耗时: {travel_llm_seconds:.2f}s 输出字符数: {len(answer_text)}"
    )
    return answer_text


def build_llm_itinerary_bundle(
    query: str,
    body_material: str,
    stream_writer: StreamWriter | None = None,
    conversation_history: str | None = None,
) -> dict:
    """
    组装单次攻略生成：返回 visible_answer，供 search_travel 序列化后由 response 解析。
    """
    visible_answer = stream_travel_guide_llm(
        query,
        body_material,
        stream_writer,
        conversation_history=conversation_history,
    )
    return {"visible_answer": visible_answer}
