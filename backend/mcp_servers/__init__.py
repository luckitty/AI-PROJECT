"""
高德 amap-maps-streamableHTTP MCP 适配。

示例::

    from mcp_servers import call_amap_maps_tool
    text = call_amap_maps_tool("maps_weather", {"city": "上海"})
"""

from mcp_servers.client import (
    call_amap_maps_tool,
    call_amap_maps_tool_async,
    list_amap_maps_tools,
    text_from_call_tool_result,
)
from mcp_servers.config import AMAP_MCP_ENDPOINT, amap_mcp_url
from mcp_servers.maps_tools import AMAP_MCP_LANGCHAIN_TOOLS

__all__ = [
    "AMAP_MCP_ENDPOINT",
    "AMAP_MCP_LANGCHAIN_TOOLS",
    "amap_mcp_url",
    "call_amap_maps_tool",
    "call_amap_maps_tool_async",
    "list_amap_maps_tools",
    "text_from_call_tool_result",
]
