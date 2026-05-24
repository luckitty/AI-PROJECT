"""
将高德 MCP 常用能力封装为 LangChain StructuredTool，供 ToolRegistry / Agent 选用。
"""

import json

from langchain.tools import tool

from tool_registry.mcp.client import call_amap_maps_tool

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


# 供 ToolRegistry 批量注册
AMAP_MCP_LANGCHAIN_TOOLS = [
    amap_mcp_weather,
    amap_mcp_geo,
    amap_mcp_poi_search,
    amap_mcp_around_search,
    amap_mcp_invoke,
]
