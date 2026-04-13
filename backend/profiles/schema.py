from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserProfile(BaseModel):
    """
    与 extractor 提示词字段一致；extra=ignore 避免模型多输出无关键时校验失败。
    """

    model_config = ConfigDict(extra="ignore")

    interests: List[str] = Field(default_factory=list)
    personality: List[str] = Field(default_factory=list)
    lifestyle: List[str] = Field(default_factory=list)
    consumption_level: Optional[str] = None
    recent_intents: List[str] = Field(default_factory=list)

    @field_validator(
        "interests",
        "personality",
        "lifestyle",
        "recent_intents",
        mode="before",
    )
    @classmethod
    def coerce_str_or_list_to_str_list(cls, value: Any) -> Any:
        # LLM 有时给单个字符串，统一成去空白后的字符串列表，便于下游合并逻辑一致。
        if value is None:
            return []
        if isinstance(value, str):
            token = value.strip()
            return [token] if token else []
        return value