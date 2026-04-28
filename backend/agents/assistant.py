"""
Agent 模块 - 智能 Agent 实现
"""
import threading
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from core.llm import get_llm
from memory.short_memory import get_short_term_checkpointer
from tools.tool_registry import REGISTERED_TOOLS

# 与 ToolRegistry 复用同一份工具清单，防止新增工具后两处不一致。
ALL_TOOLS = REGISTERED_TOOLS


default_agent_singleton = None
agent_singleton_lock = threading.Lock()

# 系统提示词
DEFAULT_SYSTEM_PROMPT = """你是一个乐于助人的 AI 助手，回答专业、友好。

你附带若干工具，但**是否调用、何时调用、调用哪一个，完全由你根据用户意图自行判断**：
- 闲聊、常识、编程思路、解题说明等，你能可靠回答的，**直接回答，不必调用工具**。
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
    # 短期记忆：LangGraph RedisSaver，按 thread_id（与前端 session_id 一致）持久化对话状态
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=get_short_term_checkpointer(),
    )

    if tools is ALL_TOOLS and system_prompt is DEFAULT_SYSTEM_PROMPT:
        with agent_singleton_lock:
            if default_agent_singleton is None:
                default_agent_singleton = agent
        return default_agent_singleton

    return agent


def clear_agent_session(thread_id: str) -> None:
    """按 thread_id 删除 Redis 中该会话的 LangGraph 检查点（与前端 session_id 一致）。"""
    if not thread_id:
        return
    get_short_term_checkpointer().delete_thread(thread_id)
