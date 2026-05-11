import json
from core.llm import get_llm
from tools.weather_tool import get_weather
from tools.search_web_tool import web_search
from tools.amap_route_tool import amap_route
from tools.route_fast_path import try_route_fast_decision

# 全局唯一工具清单：ToolRegistry 与 assistant 共用，避免两处定义漂移。
REGISTERED_TOOLS = [get_weather, web_search, amap_route]

# 仅用于 select_tool：集中写清互斥与优先级，比分散在各 @tool 长描述里更易维护；create_agent 仍读各工具的简短 description。
TOOL_SELECTOR_ROUTING = """
            路由原则（专事专用，逐条比对用户意图后再选 tool）：
            - get_weather：用户明确要**某城市**天气、气温、实况或短时预报。
            - amap_route：用户问**两地之间怎么走、怎么去**；一次返回**步行、打车提示、公交/地铁**摘要；须抽取起点地名、终点地名、城市（同城时 destination_city 与 city 填同一城市名）。若用户用「上面」「这段」指代，应从多轮对话节选里归纳真实起终点地名再调用本工具。
            - web_search：需要**互联网上的较新信息或可在线核对的客观事实**（新闻、政策法规、产品版本、时效数据、训练知识可能过时或需多源佐证等）。不要把天气、演示股价、本地旅游缓存、本地库专有检索交给本工具。
            args 内键名必须与工具参数名完全一致：
            - get_weather → city
            - web_search → query（自然语言检索式，不要塞工具名或 JSON）
            - amap_route → origin_place, destination_place, city, destination_city
"""


class ToolRegistry:

    def __init__(self):
        # 注册所有工具
        # @tool 装饰后已是 StructuredTool 实例，不可再调用 ()
        # 按固定字符串名注册，便于 prompt 与代码对齐。
        self.tools = {
            "get_weather": get_weather,
            "web_search": web_search,
            "amap_route": amap_route,
        }

    def get_tool_descriptions(self):
        """
        给LLM看的工具说明
        """
        desc = []
        for tool_name, tool in self.tools.items():
            # 简化实现：只提供工具名与描述，减少额外逻辑。
            desc.append(f"- {tool_name}: {tool.description}")
        return "\n".join(desc)

    def select_tool(self, query: str, conversation_history: str | None = None):
        """
        用 LLM 选择工具；conversation_history 为多轮对话节选，便于从全文抽取地名与指代。
        """
        tool_descriptions = self.get_tool_descriptions()
        ch = (conversation_history or "").strip()
        history_block = ""
        if ch:
            history_block = (
                "\n\n【本轮为止的多轮对话节选（通勤类请从中归纳 origin/destination）】\n"
                f"{ch}\n"
            )
        prompt = f"""
            你是一个工具选择器（Tool Selector）。请根据用户问题选择最合适的工具并输出严格 JSON。如果没有合适的工具，就不要选择工具。

            {TOOL_SELECTOR_ROUTING}

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

        # JSON 生成要稳定；关闭 streaming，降低随机性
        # max_tokens 收紧：仅产出一段 JSON，加快工具选择回合。
        llm = get_llm(streaming=False, temperature=0, max_tokens=180)
        msg = llm.invoke(prompt)
        text = msg.content if hasattr(msg, "content") else str(msg)
        print("select_tool===========select_tool \n", text, "\n")
        try:
            # 最简单实现：要求模型直接返回 JSON 字符串。
            return json.loads((text or "").strip())
        except Exception:
            return None

    def run(self, query: str, conversation_history: str | None = None):
        """
        对外执行入口；conversation_history 为多轮节选，供选路与参数对齐上文。
        """
        decision = try_route_fast_decision(query)
        if decision is None:
            decision = self.select_tool(query, conversation_history=conversation_history)
        if not isinstance(decision, dict):
            return "未选择合适工具"

        tool_name = decision.get("tool")
        args = decision.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}

        tool = self.tools.get(tool_name)
        print("tool===========tool \n", tool, "\n")

        if not tool:
            return f"工具不存在: {tool_name}"

        try:
            # StructuredTool 推荐用 invoke(dict) 传参；run(**kwargs) 会触发签名不匹配。
            # 这里统一用 invoke，避免出现“unexpected keyword argument”参数错误。
            return tool.invoke(args)
        except TypeError as e:
            # 参数不匹配（缺参/类型不对）通常由模型生成错误触发
            return f"工具执行失败(参数错误): {str(e)}"
        except Exception as e:
            return f"工具执行失败: {str(e)}"


# 单例🔥
tool_executor = ToolRegistry()