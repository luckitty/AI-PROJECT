"""
Agent 模块 - 智能 Agent 实现
"""
import threading

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
# 使用相对导入
from tools.weather_tool import get_weather
from tools.stock_tool import get_stock_price
from tools.search_txt_tool import search_local_knowledge

from core.llm import get_llm

ALL_TOOLS = [get_weather, get_stock_price, search_local_knowledge]

# 与 LangGraph Agent 共用：按 thread_id（对应前端的 session_id）隔离多轮对话状态
agent_checkpointer = InMemorySaver()

default_agent_singleton = None
agent_singleton_lock = threading.Lock()

# 系统提示词
DEFAULT_SYSTEM_PROMPT = """你是一个乐于助人的 AI 助手，回答专业、友好。

你附带若干工具，但**是否调用、何时调用、调用哪一个，完全由你根据用户意图自行判断**：
- 闲聊、常识、编程思路、解题说明等，你能可靠回答的，**直接回答，不必调用工具**。
- 用户**明确要某地实时天气**时，使用天气工具；不要对无关问题调用天气工具。
- 用户**明确要查询股票价格**时，使用股票工具。
- 仅当问题**明显依赖本地知识库中的专有资料**（例如知识库里的人物传记、内部文档类事实）时，再使用 search_local_knowledge；通用百科类问题无需检索知识库。

**使用本地知识库检索（search_local_knowledge）时务必遵守：**
- 检索片段是**候选材料**，可能不完整或与用户问题不完全相关；先判断片段是否**真正回答**了当前问题。
- **只根据片段中明确出现的内容**陈述事实；片段里没有的信息不要编造，并说明「知识库中未找到」或「片段未覆盖该点」。
- 若片段与问题明显不符或相互矛盾，应如实说明检索结果不理想，不要强行把无关内容当成答案。
- 需要时简要说明依据来自哪条片段（例如「根据片段1…」），便于用户核对。

不要为「显得专业」而滥用工具；能直接答就直答，需要外部或库内事实再调用。
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
        system_prompt=system_prompt,
        checkpointer=agent_checkpointer,
    )

    if tools is ALL_TOOLS and system_prompt is DEFAULT_SYSTEM_PROMPT:
        with agent_singleton_lock:
            if default_agent_singleton is None:
                default_agent_singleton = agent
        return default_agent_singleton

    return agent


def clear_agent_session(thread_id: str) -> None:
    """按 thread_id 清除内存中的 Agent 检查点（与前端 session_id 一致）。"""
    if not thread_id:
        return
    agent_checkpointer.storage.pop(thread_id, None)
    write_keys = [k for k in agent_checkpointer.writes if k and k[0] == thread_id]
    for k in write_keys:
        del agent_checkpointer.writes[k]
    blob_keys = [k for k in agent_checkpointer.blobs if k and k[0] == thread_id]
    for k in blob_keys:
        del agent_checkpointer.blobs[k]
