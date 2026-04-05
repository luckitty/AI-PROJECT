"""
核心配置模块
"""
import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 模型配置
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# LangSmith 配置（可选）
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")

# 高德api
AMAP_KEY = os.getenv("AMAP_KEY")

# 智谱api
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 设置环境变量
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
os.environ["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
