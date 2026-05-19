"""
将高德 MCP 常用能力封装为 LangChain StructuredTool，供 amap_mcp_registry / Agent 选用。
"""

import json

from langchain.tools import tool

from mcp_servers.client import call_amap_maps_tool


@tool
def amap_mcp_weather(city: str) -> str:
    """
    通过高德 MCP 查询城市天气预报（含多日预报），比仅实况天气更全。
    用户问某城市未来几天天气、气温趋势时调用；参数 city 为城市名或 adcode。
    """
    result = call_amap_maps_tool("maps_weather", {"city": city})
    return result


@tool
def amap_mcp_geo(address: str, city: str = "") -> str:
    """
    通过高德 MCP 把地址或 POI 名称解析为经纬度坐标。
    用户要给某地定位、查坐标、或后续算路/周边搜需要先 geocode 时调用。
    """
    address = str(address or "").strip()
    if not address:
        return "请提供待解析的地址或地名。"
    args: dict[str, str] = {"address": address}
    city = str(city or "").strip()
    if city:
        args["city"] = city
    return call_amap_maps_tool("maps_geo", args)


@tool
def amap_mcp_poi_search(keywords: str, city: str = "", citylimit: bool = False) -> str:
    """
    通过高德 MCP 按关键字搜索 POI（景点、餐厅、酒店等）。
    用户要找某地附近的某类地点、或按关键词搜店名时调用。
    """
    keywords = str(keywords or "").strip()
    if not keywords:
        return "请提供搜索关键词。"
    args: dict[str, str | bool] = {"keywords": keywords, "citylimit": citylimit}
    city = str(city or "").strip()
    if city:
        args["city"] = city
    return call_amap_maps_tool("maps_text_search", args)


@tool
def amap_mcp_poi_detail(poi_id: str) -> str:
    """
    通过高德 MCP 查询 POI 详情（含经纬度 location），供专属地图等需 id+坐标 的场景使用。
    poi_id 来自 amap_mcp_poi_search 返回结果中的 id 字段。
    """
    poi_id = str(poi_id or "").strip()
    if not poi_id:
        return "请提供 POI id（poi_id）。"
    return call_amap_maps_tool("maps_search_detail", {"id": poi_id})


@tool
def amap_mcp_around_search(keywords: str, location: str, radius: str = "3000") -> str:
    """
    通过高德 MCP 做周边 POI 搜索；location 为「经度,纬度」，radius 为半径（米）。
    用户已知道中心点坐标、要在附近找店/景点时调用。
    """
    keywords = str(keywords or "").strip()
    location = str(location or "").strip()
    if not keywords or not location:
        return "周边搜需要 keywords 与 location（经度,纬度）。"
    return call_amap_maps_tool(
        "maps_around_search",
        {"keywords": keywords, "location": location, "radius": str(radius or "3000")},
    )


@tool
def amap_mcp_transit(
    origin: str,
    destination: str,
    city: str,
    cityd: str,
) -> str:
    """
    通过高德 MCP 规划公交/地铁/火车等综合公共交通通勤方案。
    用户问两地之间怎么坐公交、地铁、换乘、跨城火车时调用；须已知起终点经纬度（经度,纬度）。
    跨城场景 city（起点城市）与 cityd（终点城市）必填且须准确；同城时两者填同一城市名。
    若用户只提供地名尚未 geocode，应先调用 amap_mcp_geo 解析坐标后再调用本工具。
    返回含 origin、destination、distance、transits 等方案详情。
    """
    origin = str(origin or "").strip()
    destination = str(destination or "").strip()
    city = str(city or "").strip()
    cityd = str(cityd or "").strip()
    if not origin or not destination:
        return "公交路径规划需要 origin 与 destination（经度,纬度）。"
    if not city or not cityd:
        return "公交路径规划需要 city（起点城市）与 cityd（终点城市）。"
    return call_amap_maps_tool(
        "maps_direction_transit_integrated",
        {
            "origin": origin,
            "destination": destination,
            "city": city,
            "cityd": cityd,
        },
    )


@tool
def amap_mcp_personal_map(org_name: str, line_list_json: str) -> str:
    """
    通过高德 MCP 将行程规划导入高德地图，生成专属地图唤端链接。
    用户要把多日行程、途径点位生成可打开的高德地图链接时调用。
    org_name 为行程/地图名称；line_list_json 为 JSON 数组，每项含 title（当日描述）与
    pointInfoList（途径点数组，每点含 name、lon、lat、poiId；poiId 须为高德 POI 关键词搜返回的 id，不可为空）。
    返回结果为高德地图 URI 链接，应原样返回给用户，勿改写或省略。
    """
    org_name = str(org_name or "").strip()
    if not org_name:
        return "请提供行程名称 org_name。"
    raw = (line_list_json or "").strip()
    if not raw:
        return "请提供 line_list_json（每日行程与途径点 JSON 数组）。"
    try:
        line_list = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"line_list_json 不是合法 JSON: {exc}"
    if not isinstance(line_list, list):
        return "line_list_json 解析后必须是 JSON 数组。"
    # 高德专属地图接口要求每个途径点 poiId 非空，调用前校验避免 MCP 侧报错。
    for day_index, day in enumerate(line_list):
        if not isinstance(day, dict):
            return f"line_list[{day_index}] 必须是对象。"
        points = day.get("pointInfoList") or []
        if not isinstance(points, list):
            return f"line_list[{day_index}].pointInfoList 必须是数组。"
        for point_index, point in enumerate(points):
            if not isinstance(point, dict):
                return f"途径点 line_list[{day_index}].pointInfoList[{point_index}] 必须是对象。"
            poi_id = str(point.get("poiId") or "").strip()
            if not poi_id:
                pname = point.get("name") or "未知地点"
                return f"途径点「{pname}」缺少有效 poiId，请用 POI 关键词搜索获取 id 后再导入。"
    return call_amap_maps_tool(
        "maps_schema_personal_map",
        {"orgName": org_name, "lineList": line_list},
    )


@tool
def amap_mcp_invoke(tool_name: str, arguments_json: str) -> str:
    """
    调用高德 MCP 任意工具（高级）。tool_name 为 MCP 工具名（如 maps_regeocode）；
    arguments_json 为 JSON 对象字符串，键名须与该工具参数一致。
    仅在其它专用工具无法覆盖、且已知工具名与参数时使用。
    """
    tool_name = str(tool_name or "").strip()
    print("amap_mcp_invoke===========tool_name \n", tool_name, "\n")
    if not tool_name:
        return "请提供 MCP 工具名 tool_name。"
    raw = (arguments_json or "").strip() or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"arguments_json 不是合法 JSON: {exc}"
    if not isinstance(args, dict):
        return "arguments_json 解析后必须是 JSON 对象。"
    return call_amap_maps_tool(tool_name, args)


# 供 amap_mcp_registry 与 assistant 批量注册
AMAP_MCP_LANGCHAIN_TOOLS = [
    amap_mcp_weather,
    amap_mcp_geo,
    amap_mcp_poi_search,
    amap_mcp_poi_detail,
    amap_mcp_around_search,
    amap_mcp_transit,
    amap_mcp_personal_map,
    amap_mcp_invoke,
]
