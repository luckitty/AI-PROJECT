from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "北京" if user_id == "1" else "上海"
