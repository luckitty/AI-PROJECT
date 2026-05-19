"""
高德 maps MCP（streamable HTTP）端点配置。

与 Cursor ``mcp.json`` 里 ``amap-maps-streamableHTTP`` 的 URL 形式一致：
``https://mcp.amap.com/mcp?key=...``
"""

from core.config import AMAP_KEY

# MCP 服务根路径（key 由运行时从配置拼接）
AMAP_MCP_ENDPOINT = "https://mcp.amap.com/mcp"


def amap_mcp_url() -> str:
    """
    返回可连接的 MCP URL；未配置 ``AMAP_KEY`` 时返回空字符串，由调用方决定是否报错。
    """
    key = (AMAP_KEY or "").strip()
    if not key:
        return ""
    return f"{AMAP_MCP_ENDPOINT}?key={key}"
