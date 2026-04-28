"""
工具模块 - 集中管理所有工具
"""
from tools.weather_tool import get_weather
from tools.search_web_tool import web_search

# 工具集合
ALL_TOOLS = [get_weather, web_search]
