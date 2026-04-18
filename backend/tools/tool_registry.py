import json
from core.llm import get_llm
from tools.weather_tool import get_weather
from tools.stock_tool import get_stock_price
from tools.search_web_tool import web_search
from tools.search_travel_tool import search_travel
# 全局唯一工具清单：ToolRegistry 与 assistant 共用，避免两处定义漂移。
REGISTERED_TOOLS = [get_weather, get_stock_price, web_search, search_travel]

# 仅用于 select_tool：集中写清互斥与优先级，比分散在各 @tool 长描述里更易维护；create_agent 仍读各工具的简短 description。
TOOL_SELECTOR_ROUTING = """
            路由原则（专事专用，逐条比对用户意图后再选 tool）：
            - get_weather：用户明确要**某城市**天气、气温、实况或短时预报，用户输入多个城市就需要多次查询。
            - get_stock_price：用户明确要**演示股票**（苹果 / 腾讯 / 特斯拉）的价格或走势类问题（本工具为演示数据）。
            - web_search：需要**互联网上的较新信息或可在线核对的客观事实**（新闻、政策法规、产品版本、时效数据、训练知识可能过时或需多源佐证等）。不要把天气、演示股价、本地旅游缓存、本地库专有检索交给本工具。
            - search_travel：用户问**旅游攻略、行程路线、目的地景点与美食玩法**等，且应优先从**本地已缓存的笔记（data/cache）**中检索时；会返回笔记正文(desc)与配图 OCR。不要用于查天气、股价、全网新闻。
            args 内键名必须与工具参数名完全一致：
            - get_weather → city
            - get_stock_price → stock_name、stock_time（缺省可填「今天」等合理值）
            - web_search → query（自然语言检索式，不要塞工具名或 JSON）
            - search_travel → query（自然语言检索式，不要塞工具名或 JSON）
"""


class ToolRegistry:

    def __init__(self):
        # 注册所有工具
        # @tool 装饰后已是 StructuredTool 实例，不可再调用 ()
        # 按固定字符串名注册，便于 prompt 与代码对齐。
        self.tools = {
            "get_weather": get_weather,
            "get_stock_price": get_stock_price,
            "web_search": web_search,
            "search_travel": search_travel,
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

    def select_tool(self, query: str):
        """
        用LLM选择工具🔥
        """
        tool_descriptions = self.get_tool_descriptions()
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

            用户问题：
            {query}
            """

        # JSON 生成要稳定；关闭 streaming，降低随机性
        llm = get_llm(streaming=False, temperature=0)
        msg = llm.invoke(prompt)
        text = msg.content if hasattr(msg, "content") else str(msg)
        try:
            # 最简单实现：要求模型直接返回 JSON 字符串。
            return json.loads((text or "").strip())
        except Exception:
            return None

    def run(self, query: str):
        """
        对外执行入口
        """
        decision = self.select_tool(query)
        if not isinstance(decision, dict):
            return "未选择合适工具"

        tool_name = decision.get("tool")
        args = decision.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}

        tool = self.tools.get(tool_name)

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