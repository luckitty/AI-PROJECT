"""
Agent 模块 - 智能 Agent 实现
"""
import threading

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
# 使用相对导入
from tools.weather_tool import get_weather
from tools.stock_tool import get_stock_price
from tools.search_txt_tool import search_local_knowledge

from core.llm import get_llm

ALL_TOOLS = [get_weather, get_stock_price, search_local_knowledge]

default_agent_singleton = None
agent_singleton_lock = threading.Lock()

# 系统提示词
DEFAULT_SYSTEM_PROMPT = """你是一个乐于助人的AI助手，具有丰富的知识和耐心的态度。
你可以使用以下工具来帮助用户：
1. 查询天气
2. 查询本地知识库（search_local_knowledge）
3. 查询股票价格

请以专业但友好的方式回答用户的问题。
如果问题涉及代码，请提供清晰、格式良好的示例。
"""


def create_assistant(
    tools: list = None,
    system_prompt: str = None,
) -> ChatOpenAI:
    """
    创建 AI 助手 Agent

    Args:
        model: 模型名称
        tools: 工具列表
        system_prompt: 系统提示词
        temperature: 温度参数

    Returns:
        Agent 实例
    """
    global default_agent_singleton

    if tools is None and system_prompt is None and default_agent_singleton is not None:
        return default_agent_singleton

    if tools is None:
        tools = ALL_TOOLS

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    llm = get_llm()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    if tools is ALL_TOOLS and system_prompt is DEFAULT_SYSTEM_PROMPT:
        with agent_singleton_lock:
            if default_agent_singleton is None:
                default_agent_singleton = agent
        return default_agent_singleton

    return agent
