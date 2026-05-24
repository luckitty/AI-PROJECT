import json
from core.llm import get_llm
from tools.search_web_tool import web_search

# 全局唯一工具清单：ToolRegistry 与 assistant 共用；天气/算路/POI 已统一走高德 MCP。
REGISTERED_TOOLS = [web_search]

# 仅用于 select_tool：联网检索等非地图类外部信息。
TOOL_SELECTOR_ROUTING = """
            路由原则：
            - web_search：需要**互联网上的较新信息或可在线核对的客观事实**（新闻、政策法规、产品版本、时效数据、训练知识可能过时或需多源佐证等）。不要把天气、本地旅游缓存、本地库专有检索、地图 POI/算路交给本工具。
            args 内键名必须与工具参数名完全一致：
            - web_search → query（自然语言检索式，不要塞工具名或 JSON）
"""


class ToolRegistry:

    def __init__(self):
        # @tool 装饰后已是 StructuredTool 实例，不可再调用 ()
        self.tools = {
            "web_search": web_search,
        }

    def get_tool_descriptions(self):
        """
        给LLM看的工具说明
        """
        desc = []
        for tool_name, tool in self.tools.items():
            desc.append(f"- {tool_name}: {tool.description}")
        return "\n".join(desc)

    def select_tool(self, query: str, conversation_history: str | None = None):
        """
        用 LLM 选择工具；conversation_history 为多轮对话节选，便于从全文抽取检索词与指代。
        """
        tool_descriptions = self.get_tool_descriptions()
        ch = (conversation_history or "").strip()
        print("tool_registry===========ch", ch, "\n")
        history_block = ""
        if ch:
            history_block = (
                "\n\n【本轮为止的多轮对话节选】\n"
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

        llm = get_llm(streaming=False, temperature=0, max_tokens=180)
        msg = llm.invoke(prompt)
        text = msg.content if hasattr(msg, "content") else str(msg)
        print("select_tool===========select_tool \n", text, "\n")
        try:
            return json.loads((text or "").strip())
        except Exception:
            return None

    def run(self, query: str, conversation_history: str | None = None):
        """
        对外执行入口；conversation_history 为多轮节选，供选工具与参数对齐上文。
        """
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
            return tool.invoke(args)
        except TypeError as e:
            return f"工具执行失败(参数错误): {str(e)}"
        except Exception as e:
            return f"工具执行失败: {str(e)}"


# 单例🔥
tool_executor = ToolRegistry()
