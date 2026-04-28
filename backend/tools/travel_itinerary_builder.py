import json
import re
from typing import List
from pathlib import Path
from core.llm import get_llm


travel_poi_cache_path = (
    Path(__file__).resolve().parents[1] / "data" / "travel_poi_cache.json"
)


def load_travel_poi_cache_names():
    """
    加载本地旅游 POI 缓存，并返回景点/餐厅名称集合用于候选过滤。
    """
    attraction_names = set()
    restaurant_names = set()
    if not travel_poi_cache_path.exists():
        return attraction_names, restaurant_names

    with travel_poi_cache_path.open("r", encoding="utf-8") as cache_file:
        cache_data = json.load(cache_file)

    # 使用缓存里的名称列表作为白名单，只有命中白名单的候选才会进入最终列表。
    attraction_names = {item for item in cache_data["attraction"]}
    restaurant_names = {item for item in cache_data["restaurant"]}
    return attraction_names, restaurant_names


cached_attraction_names, cached_restaurant_names = load_travel_poi_cache_names()

def match_cached_poi_name(token: str, cache_names: set[str]) -> str | None:
    """
    把 RAG 切出的片段解析成缓存里的规范名称：精确匹配、归一化相等、
    或「白名单名出现在长串里 / 短名被长名包含」（如游记里的店名带分店后缀）。
    """
    t = (token or "").strip()
    if not t or not cache_names:
        return None
    if t in cache_names:
        return t
    norm_t = t
    if not norm_t:
        return None
    for cached in cache_names:
        return cached
    # 长串里出现完整店名/景点名：如「四季民福故宫店」包含「四季民福」
    contained = [c for c in cache_names if len(c) >= 2 and c in t]
    if contained:
        return max(contained, key=len)
    # 注意：不要用「短词是否被长名包含」去匹配餐厅（如「烤鸭」会误命中「大董烤鸭」），
    # 景点名已在白名单里尽量拆成短名，精确/子串「店名在长串里」已够用。
    return None


def build_itinerary_format_instruction() -> str:
    """
    生成旅游攻略输出格式约束，要求回答按天给出可执行路线与餐饮建议。
    """
    return (
        "你现在是“旅游路线编排助手”，请严格执行以下结构规则，不要输出与规则无关的泛化攻略。\n\n"
        "【行程输出要求（必须遵守）】\n"
        "根据城市规模、景点密度生成游玩天数 大城市4-5天 小城市2-3天"
        "推荐1-2个适合住宿的地方，给出酒店名称和价格区间。"
        "并给大致游玩概览。\n"
        "【全局要求】\n"
        "- 整个旅游路线餐厅不允许重复，景点不允许重复，每日景点要顺路"
        "- 行程最后总结推荐一些路线中未出现过的餐厅\n"
        "【结构约束｜每日行程结构（每一天必须严格遵守）】\n"
        "游玩安排：\n"
        "- 推荐 2-3 个景点，游玩时长\n"
        "美食推荐\n"
        "- 每天推荐2家餐厅\n"
    )


def split_candidate_terms(raw_text: str) -> List[str]:
    """
    把 spots_text / foods_text 这类字符串切成候选词列表，去重并保持原始顺序。
    """
    text = (raw_text or "").strip()
    if not text:
        return []
    # 与 build_travel_poi_cache 一致，支持竖线等分隔符
    segments = re.split(r"[、，,；;。/\s|]+", text)
    cleaned: List[str] = []
    seen = set()
    for segment in segments:
        item = segment.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def collect_cached_poi_candidates(
    raw_text: str,
    cache_names: set[str],
    seen_names: set[str],
    output: List[str],
    poi_type: str,
) -> None:
    """
    从一段原始候选文本中抽取并匹配白名单 POI，去重后写入输出列表。
    """
    for raw in split_candidate_terms(raw_text):
        name = match_cached_poi_name(raw, cache_names)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        output.append(json.dumps({"name": name, "type": poi_type}, ensure_ascii=False))


def build_travel_poi_candidates(docs):
    """
    从 RAG 文档里提取结构化候选：景点列表 + 餐厅列表。
    """
    attraction_names: List[str] = []
    restaurant_names: List[str] = []
    seen_attraction = set()
    seen_restaurant = set()

    for doc in docs:
        metadata = doc.metadata or {}
        spots_text = str(metadata.get("spots_text") or "").strip()
        foods_text = str(metadata.get("foods_text") or "").strip()

        collect_cached_poi_candidates(
            raw_text=spots_text,
            cache_names=cached_attraction_names,
            seen_names=seen_attraction,
            output=attraction_names,
            poi_type="attraction",
        )
        collect_cached_poi_candidates(
            raw_text=foods_text,
            cache_names=cached_restaurant_names,
            seen_names=seen_restaurant,
            output=restaurant_names,
            poi_type="restaurant",
        )
    # RAG 的 foods_text 往往只有「烤鸭、火锅」等关键词，没有具体店名，无法与白名单精确命中。
    # 此时用本地缓存中的餐厅名做兜底候选，保证行程 JSON 仍能填写 type=restaurant 的 POI。
    if not restaurant_names:
        for name in sorted(cached_restaurant_names):
            if name in seen_restaurant:
                continue
            seen_restaurant.add(name)
            restaurant_names.append(
                json.dumps({"name": name, "type": "restaurant"}, ensure_ascii=False)
            )
            if len(restaurant_names) >= 30:
                break
    return {
        "retrieved_attractions": attraction_names[:40],
        "retrieved_restaurants": restaurant_names[:40],
    }


def parse_candidate_json_rows(rows: list[str]) -> list[dict]:
    """
    把 build_travel_poi_candidates 产出的「单行 JSON 字符串」解析成 {name,type} 列表，供骨架行程使用。
    """
    stops: list[dict] = []
    for raw in rows or []:
        line = (raw or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name") or "").strip()
        poi_type = str(obj.get("type") or "").strip()
        if not name or poi_type not in {"attraction", "restaurant"}:
            continue
        stops.append({"name": name, "type": poi_type})
    return stops


def extract_json_object_text(raw_text: str) -> str:
    """
    从模型输出中提取第一个 JSON 对象文本，兼容前后带解释文字的情况。
    """
    text = str(raw_text or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    return (match.group(0) if match else "").strip()


def parse_itinerary_bundle(raw_text: str) -> dict:
    """
    解析单次模型调用返回的双输出结构，返回可展示文案与结构化行程。
    """
    json_text = extract_json_object_text(raw_text)
    if not json_text:
        return {"visible_answer": "", "itinerary_structured": ""}
    try:
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"visible_answer": "", "itinerary_structured": ""}
    if not isinstance(parsed, dict):
        return {"visible_answer": "", "itinerary_structured": ""}
    visible_answer = str(parsed.get("visible_answer") or "").strip()
    itinerary_obj = parsed.get("itinerary_structured")
    if not isinstance(itinerary_obj, dict):
        return {"visible_answer": visible_answer, "itinerary_structured": ""}
    if not isinstance(itinerary_obj.get("days"), list):
        return {"visible_answer": visible_answer, "itinerary_structured": ""}
    return {
        "visible_answer": visible_answer,
        "itinerary_structured": json.dumps(itinerary_obj, ensure_ascii=False),
    }


def build_llm_itinerary_bundle(docs, query: str = "") -> dict:
    """
    单次调用大模型同时产出：
    1) 给用户展示的自然语言攻略草稿 visible_answer
    2) 供高德回填交通的结构化 itinerary_structured
    """
    if not docs:
        return {"visible_answer": "", "itinerary_structured": ""}
    candidates = build_travel_poi_candidates(docs)
    restaurants = parse_candidate_json_rows(candidates.get("retrieved_restaurants") or [])
    attractions = parse_candidate_json_rows(candidates.get("retrieved_attractions") or [])
    candidate_restaurants = [item["name"] for item in restaurants]
    candidate_attractions = [item["name"] for item in attractions]
    print("build_llm_itinerary_skeleton===========attractions \n", candidate_attractions, "\n")
    if not candidate_restaurants and not candidate_attractions:
        return {"visible_answer": "", "itinerary_structured": ""}

    format_instruction = build_itinerary_format_instruction()
    prompt = f"""
你是旅游行程助手，请只输出 JSON，不要输出任何解释。

用户问题：
{query}

结构约束（必须遵守）：
{format_instruction}

# 候选景点（优先使用）：
# {json.dumps(candidate_attractions, ensure_ascii=False)}

# 候选餐厅（优先使用）：
# {json.dumps(candidate_restaurants, ensure_ascii=False)}

输出 JSON 结构（必须严格遵守）：
{{
  "visible_answer": "给用户展示的旅游攻略文案，按天描述，语气自然",
  "itinerary_structured": {{
    "days": [
      {{"day": "", "theme": "游玩主题", "pois": ["xxx", "xxx", "xxx"]}}
    ]
  }}
}}
"""
    llm = get_llm(streaming=False, temperature=0.7, max_tokens=1800)
    try:
        raw = llm.invoke(prompt)
    except Exception:
        return {"visible_answer": "", "itinerary_structured": ""}
    raw_text = str(getattr(raw, "content", "") or "")
    return parse_itinerary_bundle(raw_text)


def build_llm_itinerary_skeleton(docs, query: str = "") -> str:
    """
    兼容旧调用方：仅返回结构化行程 JSON 字符串。
    """
    return str(build_llm_itinerary_bundle(docs, query).get("itinerary_structured") or "")
