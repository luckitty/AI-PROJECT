"""
对话链模块 - 支持历史记忆的对话
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from core.llm import get_llm
from datetime import datetime

# 使用相对导入

# ============ 内存历史存储 ============

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """内存中的聊天历史"""
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self):
        self.messages = []


# 会话存储
_session_store: dict[str, InMemoryHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取或创建会话历史"""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryHistory()
    return _session_store[session_id]


def clear_session_history(session_id: str = None):
    """清空会话历史"""
    global _session_store
    if session_id:
        if session_id in _session_store:
            _session_store[session_id].clear()
    else:
        _session_store.clear()


# ============ 对话链 ============

def create_chat_chain(
    system_prompt: str = "你是一个智能AI助手，请用中文回答问题。",
):
    """
    创建带历史记忆的对话链

    Args:
        system_prompt: 系统提示词

    Returns:
        RunnableWithMessageHistory 实例
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    print("create_chat_chain===========创建对话链 \n", datetime.now())
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history


# 默认对话链
default_chat_chain = create_chat_chain()
