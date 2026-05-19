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
AMAP_PERSONAL_MAP_OFFER_TEXT = (
    "\n\n---\n\n需要我把这份攻略导入高德地图，为你生成专属地图吗？"
    "回复「可以」或「不用」即可。"
)


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


def first_poi_from_search_payload(payload: dict) -> dict | None:
    """从 maps_text_search 返回体中取第一条 POI。"""
    pois = payload.get("pois")
    if not isinstance(pois, list) or not pois:
        return None
    top = pois[0]
    return top if isinstance(top, dict) else None


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


def resolve_poi_via_mcp(keyword: str, city_name: str, citylimit: bool) -> dict | None:
    """
    通过 amap_mcp_poi_search + amap_mcp_poi_detail 解析途径点（id、名称、经纬度）。
    """
    name = str(keyword or "").strip()
    if not name:
        return None
    city = str(city_name or "").strip()
    search_args: dict = {"keywords": name, "citylimit": citylimit}
    if city:
        search_args["city"] = city
    search_raw = amap_mcp_poi_search.invoke(search_args)
    search_data = parse_json_from_mcp_text(search_raw)
    if not isinstance(search_data, dict):
        return None
    poi = first_poi_from_search_payload(search_data)
    if not poi:
        return None
    poi_id = str(poi.get("id") or "").strip()
    if not poi_id:
        return None
    detail_raw = amap_mcp_poi_detail.invoke({"poi_id": poi_id})
    detail = parse_json_from_mcp_text(detail_raw)
    if not isinstance(detail, dict):
        return None
    location = str(detail.get("location") or "").strip()
    if not location or "," not in location:
        return None
    return {
        "name": str(detail.get("name") or poi.get("name") or name),
        "location": location,
        "poi_id": str(detail.get("id") or poi_id),
    }


def resolve_poi_for_personal_map(name: str, city_name: str) -> dict | None:
    """
    为专属地图解析途径点：先同城 MCP 关键词搜，未命中再放宽 citylimit。
    """
    city = str(city_name or "").strip()
    if city:
        hit = resolve_poi_via_mcp(name, city, citylimit=True)
        if hit:
            return hit
    return resolve_poi_via_mcp(name, city, citylimit=False)


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
3) days：数组，每项含 title（如「第1天·西湖」）与 points（字符串数组，仅景点/餐饮/住宿等具体地名，按游玩顺序，每天 2～6 个为宜）。
4) 忽略交通方式、门票价格、注意事项等非地点信息；合并重复地名。
5) 若攻略未按天拆分，按逻辑拆成 1～3 天即可。

{city_line}

用户问题：
{query or "（无）"}

攻略正文：
{guide[:6000]}

输出格式示例：
{{"org_name":"杭州三日游","days":[{{"title":"第1天","points":["西湖","雷峰塔"]}}]}}"""
    llm = get_llm(streaming=False, temperature=0, max_tokens=500)
    msg = llm.invoke(prompt)
    text = (msg.content if hasattr(msg, "content") else str(msg)) or ""
    text = text.strip()
    print("travel_personal_map_builder===========text \n", text, "\n")
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
        names: list[str] = []
        seen: set[str] = set()
        for p in points_raw:
            name = str(p or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        if not names:
            continue
        point_info_list: list[dict] = []
        for name in names:
            hit = resolve_poi_for_personal_map(name, city_name)
            if not hit:
                print(
                    "travel_personal_map_builder===========poi_miss_mcp",
                    name,
                    "city",
                    city_name,
                )
                continue
            poi_id = str(hit.get("poi_id") or "").strip()
            if not poi_id:
                continue
            pair = parse_location_to_lon_lat(str(hit.get("location") or ""))
            if not pair:
                continue
            lon, lat = pair
            point_info_list.append(
                {
                    "name": str(hit.get("name") or name),
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
    print(
        "travel_personal_map_builder===========org_name",
        org_name,
        "days",
        len(line_list),
        "city",
        city_name,
    )
    return amap_mcp_personal_map.invoke(
        {"org_name": org_name, "line_list_json": line_list_json}
    )
