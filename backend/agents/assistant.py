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
- 用户**明确要某地实时天气**时，使用天气工具；不要对无关问题调用天气工具。
- 用户**明确要查询股票价格**时，使用股票工具。
- 仅当问题**明显依赖本地知识库中的专有资料**（例如知识库里的人物传记、内部文档类事实）时，再使用 search_local_knowledge；通用百科类问题无需检索知识库。
- 当问题需要**互联网上的较新信息或可在线核对的事实**（新闻、政策、产品更新、时效数据等），且不应由天气/股价/本地库工具覆盖时，再使用 web_search；返回摘要可能不完整或过时，需自行甄别后回答。

**使用本地知识库检索（search_local_knowledge）时务必遵守：**
- 检索片段是**候选材料**，可能不完整或与用户问题不完全相关；先判断片段是否**真正回答**了当前问题。
- **只根据片段中明确出现的内容**陈述事实；片段里没有的信息不要编造，并说明「知识库中未找到」或「片段未覆盖该点」。
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
