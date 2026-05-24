"""
从旅游攻略正文抽取多日途径点，经 maps_tools 内高德 MCP 工具补全 POI 后生成专属地图唤端链接。
"""

import json
import re

from agents.rule.planner_rules import hit_allowed_travel_city, TRAVEL_RAG_CITY_ALLOWLIST
from core.llm import get_llm
from mcp_servers.maps_tools import (
    amap_mcp_personal_map,
    amap_mcp_poi_detail,
    amap_mcp_poi_search,
)


# 攻略回复末尾追问文案（与 response_node 一致，供单测或复用）。
# 「可以」为 chat-reply 链接，前端渲染成可点击按钮并自动代发用户回复。
AMAP_PERSONAL_MAP_OFFER_TEXT = (
    "\n\n---\n\n需要我把这份攻略导入高德地图，为你生成专属地图吗？"
    "[可以](chat-reply:可以)"
)

# 途径点 kind：spot 景点、food 餐饮、hotel 住宿（与 LLM 抽取及 POI 检索约定一致）。
POINT_KIND_SPOT = "spot"
POINT_KIND_FOOD = "food"
POINT_KIND_HOTEL = "hotel"
POINT_KIND_ALIASES = {
    "spot": POINT_KIND_SPOT,
    "景点": POINT_KIND_SPOT,
    "scenic": POINT_KIND_SPOT,
    "food": POINT_KIND_FOOD,
    "餐饮": POINT_KIND_FOOD,
    "美食": POINT_KIND_FOOD,
    "restaurant": POINT_KIND_FOOD,
    "hotel": POINT_KIND_HOTEL,
    "住宿": POINT_KIND_HOTEL,
}
# 高德 POI type / typecode 判断用关键词与前缀（typecode 前两位见高德分类表）。
FOOD_TYPE_KEYWORDS = ("餐饮", "美食", "餐厅", "饭店", "小吃", "咖啡", "茶饮", "酒楼", "食堂")
HOTEL_TYPE_KEYWORDS = ("住宿", "酒店", "宾馆", "民宿", "旅馆", "客栈")
SPOT_TYPE_KEYWORDS = ("风景名胜", "景点", "博物馆", "公园", "广场", "古迹", "寺庙", "景区", "文化", "展览", "塔", "街")


def parse_json_from_mcp_text(raw: str) -> dict | list | None:
    """解析 MCP 工具返回的正文 JSON；失败或明显为错误文案时返回 None。"""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("高德 MCP") or text.startswith("【MCP"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_point_kind(raw_kind: str) -> str:
    """把 LLM 或别名归一化为 spot / food / hotel。"""
    key = str(raw_kind or "").strip().lower()
    if not key:
        return POINT_KIND_SPOT
    return POINT_KIND_ALIASES.get(key, POINT_KIND_SPOT)


def poi_type_text(poi: dict) -> str:
    """取 POI 的类型描述字段，供 kind 匹配。"""
    return str(poi.get("type") or "")


def poi_typecode(poi: dict) -> str:
    """取 POI 六位 typecode。"""
    return str(poi.get("typecode") or "")


def poi_matches_food(poi: dict) -> bool:
    """判断 POI 是否属于餐饮类。"""
    typecode = poi_typecode(poi)
    if typecode.startswith("05"):
        return True
    text = poi_type_text(poi)
    return any(keyword in text for keyword in FOOD_TYPE_KEYWORDS)


def poi_matches_hotel(poi: dict) -> bool:
    """判断 POI 是否属于住宿类。"""
    typecode = poi_typecode(poi)
    if typecode.startswith("10"):
        return True
    text = poi_type_text(poi)
    return any(keyword in text for keyword in HOTEL_TYPE_KEYWORDS)


def poi_matches_spot(poi: dict) -> bool:
    """判断 POI 是否像景点/游玩类（非餐饮、非住宿）。"""
    if poi_matches_food(poi) or poi_matches_hotel(poi):
        return False
    typecode = poi_typecode(poi)
    # 11 风景名胜、14 科教文化等常见游玩 POI 大类
    if typecode.startswith(("11", "14", "08", "07")):
        return True
    text = poi_type_text(poi)
    return any(keyword in text for keyword in SPOT_TYPE_KEYWORDS) or bool(text)


def poi_matches_kind(poi: dict, kind: str) -> bool:
    """按途径点 kind 校验 POI 类型是否匹配。"""
    if kind == POINT_KIND_FOOD:
        return poi_matches_food(poi)
    if kind == POINT_KIND_HOTEL:
        return poi_matches_hotel(poi)
    return poi_matches_spot(poi)


def poi_name_score(poi: dict, keyword: str) -> int:
    """名称与关键词越接近得分越高，用于多条候选里择优。"""
    poi_name = str(poi.get("name") or "").strip()
    key = str(keyword or "").strip()
    if not poi_name or not key:
        return 0
    if poi_name == key:
        return 8
    if key in poi_name or poi_name in key:
        return 5
    # 去掉括号分店信息后比较
    base_key = re.sub(r"[（(].*[）)]", "", key).strip()
    base_name = re.sub(r"[（(].*[）)]", "", poi_name).strip()
    if base_key and base_name and (base_key in base_name or base_name in base_key):
        return 4
    return 0


def pick_poi_from_search_payload(payload: dict, keyword: str, kind: str) -> dict | None:
    """
    从 maps_text_search 返回体中按 kind 与名称相似度挑选最佳 POI，避免餐饮误匹配成景点。
    """
    pois = payload.get("pois")
    if not isinstance(pois, list) or not pois:
        return None
    best: dict | None = None
    best_score = -1
    for item in pois[:8]:
        if not isinstance(item, dict):
            continue
        score = poi_name_score(item, keyword)
        if poi_matches_kind(item, kind):
            score += 12
        elif kind == POINT_KIND_SPOT and not poi_matches_food(item) and not poi_matches_hotel(item):
            # 景点类：非餐饮住宿也可接受，但低于明确匹配
            score += 4
        if score > best_score:
            best_score = score
            best = item
    if best is not None:
        return best
    # 餐饮/住宿若无类型匹配候选，不盲取第一条，避免误标为景点
    if kind in (POINT_KIND_FOOD, POINT_KIND_HOTEL):
        return None
    top = pois[0]
    return top if isinstance(top, dict) else None


def search_keywords_for_kind(name: str, kind: str) -> list[str]:
    """
    按 kind 生成检索关键词序列：餐饮优先带「餐厅」后缀，提高命中餐饮 POI 的概率。
    """
    key = str(name or "").strip()
    if not key:
        return []
    if kind == POINT_KIND_FOOD:
        variants = [key]
        if not any(word in key for word in FOOD_TYPE_KEYWORDS):
            variants.append(f"{key} 餐厅")
        return variants
    if kind == POINT_KIND_HOTEL:
        variants = [key]
        if "酒店" not in key and "宾馆" not in key and "民宿" not in key:
            variants.append(f"{key} 酒店")
        return variants
    return [key]


def infer_city_name(guide_text: str, query: str) -> str:
    """
    从用户句或攻略正文中推断主城市名，供 POI 地理编码限定同城。
    优先六城白名单子串命中，否则返回空串由高德全国检索。
    """
    merged = f"{query or ''}\n{guide_text or ''}"
    for city in TRAVEL_RAG_CITY_ALLOWLIST:
        if city in merged:
            return city
    if hit_allowed_travel_city(query or ""):
        for city in TRAVEL_RAG_CITY_ALLOWLIST:
            if city in (query or ""):
                return city
    return ""


def resolve_poi_via_mcp(keyword: str, city_name: str, citylimit: bool, kind: str = POINT_KIND_SPOT) -> dict | None:
    """
    通过 amap_mcp_poi_search + amap_mcp_poi_detail 解析途径点（id、名称、经纬度、类型）。
    kind 用于挑选与类型匹配的 POI，避免餐饮被解析成景点。
    """
    name = str(keyword or "").strip()
    if not name:
        return None
    city = str(city_name or "").strip()
    point_kind = normalize_point_kind(kind)
    keyword_variants = search_keywords_for_kind(name, point_kind)
    best_hit: dict | None = None
    best_score = -1
    for variant in keyword_variants:
        search_args: dict = {"keywords": variant, "citylimit": citylimit}
        if city:
            search_args["city"] = city
        search_raw = amap_mcp_poi_search.invoke(search_args)
        search_data = parse_json_from_mcp_text(search_raw)
        if not isinstance(search_data, dict):
            continue
        poi = pick_poi_from_search_payload(search_data, name, point_kind)
        if not poi:
            continue
        score = poi_name_score(poi, name)
        if poi_matches_kind(poi, point_kind):
            score += 12
        if score > best_score:
            best_score = score
            best_hit = poi
    if not best_hit:
        return None
    poi_id = str(best_hit.get("id") or "").strip()
    if not poi_id:
        return None
    detail_raw = amap_mcp_poi_detail.invoke({"poi_id": poi_id})
    detail = parse_json_from_mcp_text(detail_raw)
    if not isinstance(detail, dict):
        return None
    location = str(detail.get("location") or "").strip()
    if not location or "," not in location:
        return None
    # 详情与搜索结果合并 type，供后续校验；展示名仍用高德官方名
    merged_type = str(detail.get("type") or best_hit.get("type") or "")
    merged_typecode = str(detail.get("typecode") or best_hit.get("typecode") or "")
    return {
        "name": str(detail.get("name") or best_hit.get("name") or name),
        "location": location,
        "poi_id": str(detail.get("id") or poi_id),
        "type": merged_type,
        "typecode": merged_typecode,
        "kind": point_kind,
    }


def resolve_poi_for_personal_map(name: str, city_name: str, kind: str = POINT_KIND_SPOT) -> dict | None:
    """
    为专属地图解析途径点：先同城 MCP 关键词搜，未命中再放宽 citylimit；按 kind 区分景点与餐饮检索。
    """
    city = str(city_name or "").strip()
    point_kind = normalize_point_kind(kind)
    if city:
        hit = resolve_poi_via_mcp(name, city, citylimit=True, kind=point_kind)
        if hit:
            return hit
    return resolve_poi_via_mcp(name, city, citylimit=False, kind=point_kind)


def parse_day_points(points_raw: list) -> list[dict]:
    """
    解析单日途径点列表，保留 kind（spot/food/hotel）；兼容纯字符串（默认 spot）。
    """
    points: list[dict] = []
    seen: set[str] = set()
    for item in points_raw:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            kind = normalize_point_kind(str(item.get("kind") or ""))
        else:
            name = str(item or "").strip()
            kind = POINT_KIND_SPOT
        if not name:
            continue
        dedupe_key = f"{kind}::{name}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        points.append({"name": name, "kind": kind})
    return points


def parse_location_to_lon_lat(location: str) -> tuple[float, float] | None:
    """把「经度,纬度」字符串解析为浮点数对。"""
    raw = str(location or "").strip()
    if "," not in raw:
        return None
    left, right = raw.split(",", 1)
    try:
        lon = float(left.strip())
        lat = float(right.strip())
    except ValueError:
        return None
    return lon, lat


def extract_itinerary_json(guide_text: str, query: str) -> dict | None:
    """
    用 LLM 从攻略正文中抽取专属地图所需结构：org_name 与按日分组的 POI 名称列表。
    返回 dict 或解析失败时 None。
    """
    guide = (guide_text or "").strip()
    if len(guide) < 30:
        return None
    city_hint = infer_city_name(guide, query)
    city_line = f"主城市（POI 搜索时优先使用）：{city_hint}" if city_hint else "主城市：请从正文推断"
    prompt = f"""你是行程结构化助手。根据下方旅游攻略，抽取生成高德专属地图所需的 JSON。

要求：
1) 只输出一个 JSON 对象，不要 Markdown，不要解释。
2) org_name：行程总名称，简短，如「杭州三日游」。
3) days：数组，每项含 title（如「第1天·西湖」）与 points（对象数组，按当日游玩顺序，每天 2～8 个为宜）。
4) points 每项必须含 name（地图可检索的地名）与 kind：
   - kind=spot：▎游玩安排 中的景点、景区、博物馆、步行街等
   - kind=food：▎美食推荐 中的餐厅、小吃店（写店名，勿写「XX周边」「某商圈」等模糊描述）
   - kind=hotel：正文明确提到的酒店/民宿名称（无则省略）
5) 景点与餐饮分块识别：游玩安排 → spot，美食推荐 → food；按时间顺序交错排列（如上午景点、中午 food、下午 spot）。
6) 忽略交通方式、门票价格、注意事项等非地点信息；合并重复地名。
7) 若攻略未按天拆分，按逻辑拆成 1～3 天即可。

{city_line}

用户问题：
{query or "（无）"}

攻略正文：
{guide[:6000]}

输出格式示例：
{{"org_name":"杭州三日游","days":[{{"title":"第1天","points":[{{"name":"西湖风景名胜区","kind":"spot"}},{{"name":"楼外楼","kind":"food"}},{{"name":"雷峰塔","kind":"spot"}}]}}]}}"""
    llm = get_llm(streaming=False, temperature=0, max_tokens=800)
    msg = llm.invoke(prompt)
    text = (msg.content if hasattr(msg, "content") else str(msg)) or ""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data


def build_line_list(structure: dict, city_name: str) -> list[dict]:
    """
    将 LLM 抽取的结构转为高德 lineList：每个途径点经 MCP 搜 POI + 查详情得到 poiId 与坐标。
    """
    org_days = structure.get("days") or []
    if not isinstance(org_days, list):
        return []
    line_list: list[dict] = []
    for day in org_days:
        if not isinstance(day, dict):
            continue
        title = str(day.get("title") or "行程").strip() or "行程"
        points_raw = day.get("points") or []
        if not isinstance(points_raw, list):
            continue
        day_points = parse_day_points(points_raw)
        if not day_points:
            continue
        point_info_list: list[dict] = []
        for point in day_points:
            name = str(point.get("name") or "").strip()
            kind = normalize_point_kind(str(point.get("kind") or ""))
            if not name:
                continue
            hit = resolve_poi_for_personal_map(name, city_name, kind=kind)
            if not hit:
                continue
            poi_id = str(hit.get("poi_id") or "").strip()
            if not poi_id:
                continue
            pair = parse_location_to_lon_lat(str(hit.get("location") or ""))
            if not pair:
                continue
            lon, lat = pair
            display_name = str(hit.get("name") or name)
            point_info_list.append(
                {
                    "name": display_name,
                    "lon": lon,
                    "lat": lat,
                    "poiId": poi_id,
                }
            )
        if point_info_list:
            line_list.append({"title": title, "pointInfoList": point_info_list})
    return line_list


def invoke_personal_map_from_guide(guide_text: str, query: str) -> str:
    """
    从攻略正文生成高德专属地图链接；POI 与地图生成均经 maps_tools 的 MCP 封装。
    """
    structure = extract_itinerary_json(guide_text, query)
    if not structure:
        return "未能从上一份攻略中识别出可导入的行程地点，请说明城市与每日景点后重试。"
    org_name = str(structure.get("org_name") or "").strip()
    if not org_name:
        org_name = "我的旅游行程"
    city_name = infer_city_name(guide_text, query)
    line_list = build_line_list(structure, city_name)
    if not line_list:
        return "未能解析出带坐标的途径点，请确认攻略中含具体景点或地名后再试。"
    line_list_json = json.dumps(line_list, ensure_ascii=False)
    return amap_mcp_personal_map.invoke(
        {"org_name": org_name, "line_list_json": line_list_json}
    )
