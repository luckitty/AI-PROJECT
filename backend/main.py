"""
DeepSeek Chat - AI 助手服务

FastAPI 主入口
"""
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 使用相对导入
from api.chat import router as chat_router

# 创建 FastAPI 应用
app = FastAPI(
    title="DeepSeek Chat API",
    description="你是一个AI助手服务，支持工具调用和对话历史，如果在工具中找不到相关信息，则调用大模型数据回答问题",
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
        "message": "DeepSeek Chat API",
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
