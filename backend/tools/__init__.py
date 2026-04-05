"""
工具模块 - 集中管理所有工具
"""
from tools.weather_tool import get_weather
from tools.stock_tool import get_stock_price
from tools.search_txt_tool import search_local_knowledge

# 工具集合
ALL_TOOLS = [get_weather, get_stock_price, search_local_knowledge]
