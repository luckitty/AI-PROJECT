from core.llm import get_llm
from langchain_core.output_parsers import JsonOutputParser
from .schema import UserProfile

PROMPT = """
你是一个用户画像分析助手。

请从用户输入中提取：
- interests
- personality
- lifestyle
- consumption_level
- recent_intents

要求：
1. 不要编造
2. 没有就返回空
3. 只输出一个 JSON 对象，键为 interests、personality、lifestyle、consumption_level、recent_intents，不要其它说明文字
"""


def extract_profile(user_input: str) -> UserProfile:
    json_parser = JsonOutputParser(pydantic_object=UserProfile)
    chain = get_llm() | json_parser
    # system / human 分开；JSON 模式保证 content 为合法 JSON 字符串，用 json.loads 即可。
    res = chain.invoke(
        [
            ("system", PROMPT),
            ("human", user_input),
        ]
    )
    return UserProfile.model_validate(res)
