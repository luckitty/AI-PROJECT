"""
travel-guide-agent 后端服务（旅游攻略对话与工具编排）

FastAPI 主入口
"""
import logging
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须在 import LangChain / LangGraph 之前执行：加载 .env 并写入 LangSmith 相关 os.environ
import core.config  # noqa: E402

# 进程级日志：uvicorn 启动时也会打访问日志，此处统一根 logger 格式，便于容器/终端排查异常堆栈
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 使用相对导入
from api.chat import router as chat_router

# 创建 FastAPI 应用
app = FastAPI(
    title="travel-guide-agent API",
    description="旅游攻略助手 API：对话、工具调用与会话历史；编排走 LangGraph，缺失信息由模型补充。",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "travel-guide-agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
