import time
from typing import Any, Callable, List

from core.llm import get_llm
from skills.skill_loader import buildCombinedTravelSkillInstruction

# LangGraph custom 通道写入器：由 response 节点经 search_travel 传入，把攻略正文增量推到 SSE。
StreamWriter = Callable[[dict[str, Any]], None]


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
    instruction = buildCombinedTravelSkillInstruction()
    print(
        "travel_skills===========攻略 LLM prompt 已注入双 skill，"
        f"query={query!r} 指令字符数={len(instruction)}"
    )
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
    # 双 skill 合成后正文略长，略提高上限；仍与 skill 内「精炼」约束一致。
    llm = get_llm(streaming=bool(stream_writer), temperature=0.45, max_tokens=1400)
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
