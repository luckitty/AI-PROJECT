"""
LLM 模块 - 统一管理 LLM 实例
"""
from langchain_openai import ChatOpenAI
from core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME


print("MODEL_NAME=========",MODEL_NAME)

def get_llm(
    model: str = MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    streaming: bool = False
) -> ChatOpenAI:
    """
    获取 LLM 实例

    Args:
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
        streaming: 是否启用流式输出

    Returns:
        ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=model,
        openai_api_key= DEEPSEEK_API_KEY,
        openai_api_base= DEEPSEEK_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming
    )


# 默认 LLM 实例
default_llm = get_llm()
