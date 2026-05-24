"""
高德 ``amap-maps-streamableHTTP`` MCP 客户端（Streamable HTTP 传输）。

通过官方 MCP 端点调用 maps_geo、maps_text_search、maps_weather 等工具，
与 Cursor 里配置的 ``amap-maps-streamableHTTP`` 为同一套服务。
"""

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import httpx
import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import (
    MCP_DEFAULT_SSE_READ_TIMEOUT,
    MCP_DEFAULT_TIMEOUT,
)

from mcp_servers.config import amap_mcp_url

# 高德 MCP 不走系统 HTTP 代理（trust_env=False），避免本机代理导致 mcp.amap.com 连不上。
AMAP_MCP_HTTP_TRUST_ENV = False


def text_from_call_tool_result(result: mcp_types.CallToolResult) -> str:
    """把 MCP tools/call 返回的 content 块拼成一段可读文本。"""
    lines: list[str] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            lines.append(block.text)
        else:
            lines.append(str(block))
    if result.isError:
        prefix = "【MCP 工具错误】"
        body = "\n".join(lines).strip() or "无详细说明"
        return f"{prefix}{body}"
    return "\n".join(lines).strip()


def ensure_mcp_url(mcp_url: str | None) -> str:
    """校验 MCP URL 已配置，避免静默连到空地址。"""
    url = (mcp_url or amap_mcp_url() or "").strip()
    if not url:
        raise RuntimeError("未配置 AMAP_KEY，无法连接高德 MCP（amap-maps-streamableHTTP）。")
    return url


def build_amap_mcp_http_client() -> httpx.AsyncClient:
    """
    构造高德 MCP 专用 httpx 客户端。

    关闭 trust_env，避免读取 HTTP_PROXY 等环境变量后 TLS 握手超时（LangSmith 常见 TaskGroup 根因）。
    """
    return httpx.AsyncClient(
        follow_redirects=True,
        trust_env=AMAP_MCP_HTTP_TRUST_ENV,
        timeout=httpx.Timeout(MCP_DEFAULT_TIMEOUT, read=MCP_DEFAULT_SSE_READ_TIMEOUT),
    )


def format_mcp_exc(exc: BaseException) -> str:
    """展开 TaskGroup / ExceptionGroup 子异常，便于日志与 LangSmith 排查。"""
    message = f"{type(exc).__name__}: {exc}"
    sub_list = getattr(exc, "exceptions", None)
    if sub_list:
        details = []
        for index, sub in enumerate(sub_list, 1):
            details.append(f"[{index}] {type(sub).__name__}: {sub}")
        message = f"{message} ({'; '.join(details)})"
    return message


def run_sync_async(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
    """
    在同步上下文执行 async 协程。

    FastAPI / LangGraph 流式请求线程里往往已有 asyncio loop，不能再嵌套 anyio.run；
    此时放到独立线程里跑 anyio.run，避免 RuntimeError。
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    async def runner() -> Any:
        return await coro_factory()

    if not in_loop:
        return anyio.run(runner)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(anyio.run, runner).result()


async def with_amap_maps_session(
    callback: Callable[[ClientSession], Awaitable[Any]],
    mcp_url: str | None = None,
) -> Any:
    """
    建立一次 streamable HTTP 会话，执行 callback(session) 后关闭。

    callback 内可连续 list_tools / call_tool，共享同一连接。
    """
    url = ensure_mcp_url(mcp_url)
    async with build_amap_mcp_http_client() as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await callback(session)


async def list_amap_maps_tools_async(mcp_url: str | None = None) -> list[mcp_types.Tool]:
    """列出高德 MCP 服务暴露的全部工具定义。"""

    async def runner(session: ClientSession) -> list[mcp_types.Tool]:
        listed = await session.list_tools()
        return list(listed.tools)

    return await with_amap_maps_session(runner, mcp_url=mcp_url)


def list_amap_maps_tools(mcp_url: str | None = None) -> list[mcp_types.Tool]:
    """同步：列出 MCP 工具清单。"""

    async def runner() -> list[mcp_types.Tool]:
        return await list_amap_maps_tools_async(mcp_url=mcp_url)

    return run_sync_async(runner)


async def call_amap_maps_tool_async(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    mcp_url: str | None = None,
) -> str:
    """
    异步调用指定 MCP 工具（如 maps_weather、maps_geo），返回文本结果。

    arguments 为 JSON 可序列化的参数字典，键名须与 MCP 工具 schema 一致。
    """
    name = str(tool_name or "").strip()

    print("amap_maps_mcp===========tool_name \n", name, "\n")
    payload = arguments if isinstance(arguments, dict) else {}

    async def runner(session: ClientSession) -> str:
        result = await session.call_tool(name, payload)
        return text_from_call_tool_result(result)

    try:
        return await with_amap_maps_session(runner, mcp_url=mcp_url)
    except Exception as exc:
        return f"高德 MCP 调用失败: {format_mcp_exc(exc)}"


def call_amap_maps_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    mcp_url: str | None = None,
) -> str:
    """
    同步封装：在普通 def / ToolRegistry 中调用 MCP。

    若当前已在 running event loop 中，请改用 ``call_amap_maps_tool_async``。
    """

    async def runner() -> str:
        return await call_amap_maps_tool_async(tool_name, arguments, mcp_url=mcp_url)

    return run_sync_async(runner)
