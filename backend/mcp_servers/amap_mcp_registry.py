"""
高德 MCP 工具注册与 LLM 选路执行器。

供 LangGraph amap_mcp 节点调用；与 tools/tool_registry 对称，MCP 域逻辑集中在本模块。
"""

import json

from core.llm import get_llm
from mcp_servers.maps_tools import AMAP_MCP_LANGCHAIN_TOOLS

# 高德 MCP 工具路由说明，供 LLM 选工具时使用。
AMAP_MCP_SELECTOR_ROUTING = """
            路由原则（仅在高德 MCP 工具中选一）：
            - amap_mcp_weather：查询城市天气预报（含实况、多日预报、未来几天气温趋势）；city 为城市名或 adcode。
            - amap_mcp_geo：把地址/地名解析为经纬度；address 必填，可选 city 限定城市。
            - amap_mcp_poi_search：按关键词+城市搜 POI（景点、餐厅、酒店等）；keywords 必填，可选 city、citylimit。
            - amap_mcp_poi_detail：按 POI id 查详情（含经纬度）；poi_id 必填，id 通常来自 poi_search。
            - amap_mcp_around_search：已知中心点经纬度做周边搜；keywords、location（经度,纬度）必填，可选 radius。
            - amap_mcp_transit：公交/地铁/火车等综合公共交通路径规划；origin、destination（经度,纬度）、city、cityd 必填；跨城须分别填起终点城市。用户只给地名时须先 amap_mcp_geo 再本工具。
            - amap_mcp_personal_map：将行程导入高德生成专属地图唤端链接；org_name、line_list_json 必填；链接结果原样返回勿总结。
            - amap_mcp_invoke：其它高德 MCP 能力（步行路径 maps_direction_walking、逆地理、距离等）且参数明确时用；tool_name、arguments_json 必填。
            通勤「A 到 B 怎么走」：通常先 geo 起终点，再 transit；需要步行方案时可 invoke maps_direction_walking。
            args 内键名必须与工具参数名完全一致：
            - amap_mcp_weather → city
            - amap_mcp_geo → address；可选 city
            - amap_mcp_poi_search → keywords；可选 city、citylimit
            - amap_mcp_poi_detail → poi_id
            - amap_mcp_around_search → keywords, location；可选 radius
            - amap_mcp_transit → origin, destination, city, cityd
            - amap_mcp_personal_map → org_name, line_list_json
            - amap_mcp_invoke → tool_name, arguments_json
"""


class AmapMcpRegistry:
    """高德 MCP 工具注册与执行器。"""

    def __init__(self):
        # 按固定字符串名注册 MCP 工具，便于 prompt 与代码对齐。
        self.tools = {mcp_tool.name: mcp_tool for mcp_tool in AMAP_MCP_LANGCHAIN_TOOLS}

    def get_tool_descriptions(self) -> str:
        """给 LLM 看的 MCP 工具说明。"""
        desc = []
        for tool_name, tool in self.tools.items():
            desc.append(f"- {tool_name}: {tool.description}")
        return "\n".join(desc)

    def select_tool(self, query: str, conversation_history: str | None = None) -> dict | None:
        """
        用 LLM 选择高德 MCP 工具；conversation_history 为多轮对话节选，便于抽取地名与指代。
        """
        tool_descriptions = self.get_tool_descriptions()
        ch = (conversation_history or "").strip()
        history_block = ""
        if ch:
            history_block = (
                "\n\n【本轮为止的多轮对话节选（请从中归纳城市、地址、关键词）】\n"
                f"{ch}\n"
            )
        prompt = f"""
            你是一个高德 MCP 工具选择器。请根据用户问题选择最合适的高德 MCP 工具并输出严格 JSON。

            {AMAP_MCP_SELECTOR_ROUTING}

            可用工具（tool 字段必须与以下名称完全一致）：
            {tool_descriptions}

            请返回严格 JSON（不要输出任何额外文字），格式如下：
            {{
            "tool": "工具名",
            "args": {{
                "参数名": "值"
            }}
            }}

            当前用户问题：
            {query}
            {history_block}
            """

        llm = get_llm(streaming=False, temperature=0, max_tokens=180)
        msg = llm.invoke(prompt)
        text = msg.content if hasattr(msg, "content") else str(msg)
        print("amap_mcp_registry===========text \n", text, "\n")
        try:
            return json.loads((text or "").strip())
        except Exception:
            return None

    def run(self, query: str, conversation_history: str | None = None) -> str:
        """对外执行入口：选工具并 invoke。"""
        decision = self.select_tool(query, conversation_history=conversation_history)
        if not isinstance(decision, dict):
            return "未选择合适的高德 MCP 工具"

        tool_name = decision.get("tool")
        args = decision.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}

        tool = self.tools.get(tool_name)
        print("amap_mcp_registry===========tool \n", tool, "\n")
        if not tool:
            return f"高德 MCP 工具不存在: {tool_name}"

        try:
            return tool.invoke(args)
        except TypeError as exc:
            return f"高德 MCP 工具执行失败(参数错误): {str(exc)}"
        except Exception as exc:
            return f"高德 MCP 工具执行失败: {str(exc)}"


# 单例
amap_mcp_executor = AmapMcpRegistry()
