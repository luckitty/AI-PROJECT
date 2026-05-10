"""
路线类问题快速分流：跳过 Tool Selector 的 LLM，直接构造 amap_route 参数，降低端到端延迟。
城市不写死枚举：city / destination_city 留空，由 amap_route 内 resolve_route_endpoints 做全国 POI 同城配对。
"""
import re


def try_route_fast_decision(query: str):
    """
    若能判定为「A 到 B 怎么走」类问题，返回 {tool, args}；否则返回 None 交由 LLM 选工具。
    """
    q = (query or "").strip()
    if not q or len(q) > 160:
        return None
    # 明显不是路径规划的意图，避免抢答天气、检索等。
    block_words = ("天气", "气温", "下雨", "降雨", "台风", "新闻", "股票", "搜一下", "百度", "谷歌")
    if any(w in q for w in block_words):
        return None
    route_kw = (
        "怎么走",
        "怎么去",
        "路线",
        "导航",
        "多远",
        "多久到",
        "坐地铁",
        "公交",
        "地铁",
        "打车",
    )
    if not any(k in q for k in route_kw):
        return None
    if "到" not in q:
        return None

    origin: str | None = None
    dest: str | None = None
    m = re.search(
        r"从\s*([^从]{1,34}?)\s*到\s*([^到]{1,44}?)(?:怎么走|怎么去|路线|导航|地铁|公交|[，。！？\?\!]|$)",
        q,
    )
    if m:
        origin = m.group(1).strip()
        dest = m.group(2).strip()
    if not origin or not dest:
        m2 = re.search(
            r"([^从]{1,34}?)到([^怎么]{1,44}?)(?:怎么走|怎么去|路线|导航|[，。！？\?\!]|$)",
            q,
        )
        if m2:
            origin = m2.group(1).strip()
            dest = m2.group(2).strip()

    if origin:
        origin = re.sub(r"^(查+|请问|问一下|想知道|帮我|我要)", "", origin).strip()
    if dest:
        dest = re.split(r"[，。！？怎么]", dest)[0].strip()

    if not origin or not dest or len(origin) < 2 or len(dest) < 2:
        return None
    if len(origin) > 40 or len(dest) > 48:
        return None

    return {
        "tool": "amap_route",
        "args": {
            "origin_place": origin,
            "destination_place": dest,
            "city": "",
            "destination_city": "",
        },
    }
