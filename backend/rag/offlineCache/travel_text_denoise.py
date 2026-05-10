"""
小红书笔记正文 / OCR 常见噪声清洗：话题标签、URL、@、多余空白等。
用于建库 page_content、metadata 与攻略素材拼装，不截断长度。

更深层的广告剔除、emoji 与 desc/OCR 语义去重见 rag.offlineCache.travel_llm_guide_denoise（离线大模型，带磁盘缓存）。
"""
import re


# 小红书「#关键词[话题]#」整块
_TOPIC_HASH_PATTERN = re.compile(r"#[^#\n]{0,80}?\[话题\]#", re.DOTALL)
# 剩余 #标签#（中英数字下划线与间隔号，避免误伤普通 # 号）
_PLAIN_HASH_TAG_PATTERN = re.compile(r"#[\u4e00-\u9fa5a-zA-Z0-9_·\-]{1,40}#")
_URL_PATTERN = re.compile(r"https?://[^\s\u4e00-\u9fff]+")
_AT_PATTERN = re.compile(r"@[\w\u4e00-\u9fff]{1,30}")
# 连续空白（不含换行）压成单空格
_MULTI_SPACE_PATTERN = re.compile(r"[ \t\xa0]{2,}")
# 过多空行收束为双换行，便于阅读又不过长
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{4,}")


def denoise_xhs_note_text(text: str) -> str:
    """
    去掉小红书常见话术噪声，保留完整正文长度（不截断）。
    顺序：URL → @ → [话题]标签 → 普通 #标签# → 空白规整。
    """
    if not text or not str(text).strip():
        return ""
    s = str(text).strip()
    s = _URL_PATTERN.sub("", s)
    s = _AT_PATTERN.sub("", s)
    s = _TOPIC_HASH_PATTERN.sub(" ", s)
    s = _PLAIN_HASH_TAG_PATTERN.sub(" ", s)
    s = _MULTI_SPACE_PATTERN.sub(" ", s)
    s = _MULTI_NEWLINE_PATTERN.sub("\n\n\n", s)
    # 行首尾空格 + 行内连续空格（删除 URL/标签后常留下双空格）
    lines = []
    for ln in s.splitlines():
        lines.append(_MULTI_SPACE_PATTERN.sub(" ", ln.strip()))
    s = "\n".join(lines)
    return s.strip()
