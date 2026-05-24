"""
工具模块 - 集中管理所有工具
"""
from tools.search_web_tool import web_search

# 通用联网工具；天气/算路/POI 见 mcp_servers.maps_tools
ALL_TOOLS = [web_search]
