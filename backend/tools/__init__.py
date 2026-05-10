"""
工具模块 - 集中管理所有工具
"""
from tools.weather_tool import get_weather
from tools.search_web_tool import web_search
from tools.amap_route_tool import amap_route

# 工具集合
ALL_TOOLS = [get_weather, web_search, amap_route]
