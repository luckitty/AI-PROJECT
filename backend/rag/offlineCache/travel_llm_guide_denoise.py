"""
小红书攻略「正文级」去噪：用大模型合并 desc 与 OCR，去广告/废话/重复，只保留可执行攻略叙述。

设计要点：
- 仅适合离线预处理（逐条调 API），默认关闭；结果写入磁盘缓存，按 desc+ocr 内容签名复用。
- 与 travel_text_denoise 的规则去噪互补：入参应先经 denoise_xhs_note_text 再送入本模块。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import DEEPSEEK_API_KEY
from core.llm import get_llm

# 单次拼接过长时截断，避免超出模型上下文或费用失控；截断后仍尽量保留 desc 与 OCR 各一段。
LLM_GUIDE_INPUT_MAX_CHARS = 28000

# 去噪输出偏长攻略时放宽上限；与 DeepSeek 等非流式一次返回兼容。
LLM_GUIDE_MAX_OUTPUT_TOKENS = 8192

SYSTEM_PROMPT = """你是小红书旅游笔记的文本编辑。用户会提供【标题】【正文 desc】【配图 OCR】。
OCR 文字常与正文大量重复，且可能夹杂图片里的水印、菜单碎片。

你的任务：整理成一段「纯攻略正文」：
1) 删除硬广软广、引流话术、求关注三连、店铺推广链接式语句；
2) 删除与行程无关的闲聊、饭圈式语气堆砌、无信息量的 emoji 与重复标点；
3) 正文 desc 与 OCR 中语义重复的内容只保留一份，按阅读顺序自然衔接；
4) 保留景点/路线顺序、交通方式、耗时、费用或预算线索、预约与闭馆提示、实用 tips；
5) 不编造未出现在素材中的事实；不输出标题行；不要开场白或结尾说明。

只输出整理后的正文纯文本，不要使用 markdown 代码围栏。"""


def guide_body_source_signature(desc: str, ocr_text: str) -> str:
    """用去噪后的 desc 与 OCR 生成稳定签名，素材未变则复用缓存。"""
    raw = f"{desc}\n\x1e\n{ocr_text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_llm_guide_cache(cache_path: Path) -> dict:
    """读取大模型攻略正文缓存。"""
    if not cache_path.is_file():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_llm_guide_cache(cache_path: Path, cache_data: dict) -> None:
    """持久化攻略正文缓存。"""
    try:
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(cache_data, file, ensure_ascii=False)
    except OSError:
        pass


def trim_desc_ocr_for_prompt(desc: str, ocr_text: str) -> tuple[str, str]:
    """
    控制拼接长度：优先整段保留较短的一侧，对较长侧截断并标注。
    返回 (desc 片段, ocr 片段)。
    """
    desc = (desc or "").strip()
    ocr_text = (ocr_text or "").strip()
    total = len(desc) + len(ocr_text)
    if total <= LLM_GUIDE_INPUT_MAX_CHARS:
        return desc, ocr_text
    half = LLM_GUIDE_INPUT_MAX_CHARS // 2
    if len(desc) <= half:
        room = LLM_GUIDE_INPUT_MAX_CHARS - len(desc)
        ocr_use = ocr_text[: max(0, room - 20)] + "\n（以下 OCR 已截断）"
        return desc, ocr_use.strip()
    if len(ocr_text) <= half:
        room = LLM_GUIDE_INPUT_MAX_CHARS - len(ocr_text)
        desc_use = desc[: max(0, room - 20)] + "\n（以下正文已截断）"
        return desc_use.strip(), ocr_text
    return (
        desc[: half - 10] + "\n（正文已截断）",
        ocr_text[: half - 10] + "\n（OCR 已截断）",
    )


def unwrap_markdown_fence(text: str) -> str:
    """去掉模型偶发包裹的 ``` 围栏，避免污染正文。"""
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def invoke_llm_guide_body(title: str, desc: str, ocr_text: str) -> str:
    """
    调用大模型生成合并去噪后的攻略正文；失败时抛出异常由上层决定是否降级。
    """
    llm = get_llm(
        temperature=0,
        max_tokens=LLM_GUIDE_MAX_OUTPUT_TOKENS,
        streaming=False,
    )
    desc_use, ocr_use = trim_desc_ocr_for_prompt(desc, ocr_text)
    human_parts = []
    if title:
        human_parts.append(f"【标题】\n{title}")
    human_parts.append(f"【正文 desc】\n{desc_use or '（无）'}")
    human_parts.append(f"【配图 OCR 文字】\n{ocr_use or '（无）'}")
    human_content = "\n\n".join(human_parts)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = getattr(response, "content", None) or ""
    if isinstance(content, list):
        # 少数多模态返回为块列表
        pieces = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                pieces.append(str(block["text"]))
            else:
                pieces.append(str(block))
        content = "".join(pieces)
    return unwrap_markdown_fence(str(content).strip())


def resolve_guide_body_for_note(
    note_id: str,
    title: str,
    desc: str,
    ocr_text: str,
    cache: dict,
    invoke_llm_on_miss: bool,
    force_refresh: bool = False,
) -> tuple[str, bool]:
    """
    解析单条笔记的「大模型攻略正文」：优先命中磁盘缓存；仅在 invoke_llm_on_miss 且未命中时调用 API。

    返回 (guide_body, cache_updated)。无素材、无 Key、调用失败时 guide_body 为空字符串，
    由调用方继续用 desc+ocr；cache_updated 为 True 时表示已写入内存中的 cache 字典，需落盘。
    """
    nid = str(note_id or "").strip()
    if not nid:
        return "", False
    desc = (desc or "").strip()
    ocr_text = (ocr_text or "").strip()
    if not desc and not ocr_text:
        return "", False

    signature = guide_body_source_signature(desc, ocr_text)
    cached = cache.get(nid)
    if (
        not force_refresh
        and isinstance(cached, dict)
        and cached.get("signature") == signature
    ):
        return str(cached.get("guide_body") or "").strip(), False

    if not invoke_llm_on_miss:
        return "", False

    if not DEEPSEEK_API_KEY:
        print("travel_llm_guide_denoise===========未配置 DEEPSEEK_API_KEY，跳过大模型攻略去噪")
        return "", False

    try:
        body = invoke_llm_guide_body(title, desc, ocr_text)
    except Exception as ex:
        print(f"travel_llm_guide_denoise===========LLM 去噪失败 note_id={nid!r}: {ex!r}")
        return "", False

    if not body:
        print(f"travel_llm_guide_denoise===========LLM 返回空 note_id={nid!r}")
        return "", False

    cache[nid] = {"signature": signature, "guide_body": body}
    return body, True
