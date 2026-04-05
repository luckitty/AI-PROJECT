"""
聊天 API 路由
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from typing import Optional

# 使用相对导入
from agents.assistant import create_assistant
from chains.chat_chain import create_chat_chain, clear_session_history

from datetime import datetime

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    model: str = "deepseek-chat"
    session_id: Optional[str] = None
    use_tools: bool = Field(True, description="是否使用 Agent+工具（含知识库工具）")


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str
    session_id: Optional[str] = None


class HistoryRequest(BaseModel):
    """历史请求"""
    session_id: str


# ============ API 端点 ============

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送聊天消息"""
    try:
        if request.use_tools:
            print("chat===========使用 Agent+工具（含知识库工具） \n", datetime.now())
            # 使用 Agent（支持工具调用）
            agent = create_assistant()
            response = agent.invoke({
                "messages": [{"role": "user", "content": request.message}]
            })
            reply = response["messages"][-1].content
            print("chat===========返回结果 \n", datetime.now())
        else:
            # 使用普通对话链（支持历史记忆）
            chain = create_chat_chain()
            config = {"configurable": {"session_id": request.session_id or "default"}}
            reply = chain.invoke({"question": request.message}, config)

        return ChatResponse(reply=reply, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天"""
    async def generate():
        try:
            agent = create_assistant()
            print("流式聊天=========== \n", datetime.now())

            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": request.message}]},
                stream_mode="messages",
            ):
                if hasattr(chunk[0], 'content') and chunk[0].content:
                    yield f"data: {chunk[0].content}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """清空会话历史"""
    clear_session_history(session_id)
    return {"message": "History cleared"}


@router.get("/models")
async def get_models():
    """获取可用模型列表"""
    return {
        "models": [
            {"id": "deepseek-chat", "name": "DeepSeek Chat"},
            {"id": "deepseek-coder", "name": "DeepSeek Coder"}
        ]
    }