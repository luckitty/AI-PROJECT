"""
聊天 API 路由
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# 使用相对导入
from agents.assistant import create_assistant, clear_agent_session
from chains.chat_chain import clear_session_history

import json

from datetime import datetime


def stream_chunk_text(chunk) -> str:
    """
    LangGraph 在 stream_mode="messages" 时，每次 yield 一般是 (消息块, 元数据) 二元组。
    消息块的 .content 可能是 str，也可能是多模态用的 list（如 type=text 的字典列表）。
    这里统一转成一段可展示的纯文本；没有内容则返回空串（本拍不向客户端推 SSE）。
    """
    if chunk is None:
        return ""
    msg = chunk[0] if isinstance(chunk, tuple) and len(chunk) > 0 else chunk
    # 获取msg的content 如果没有则返回None
    raw = getattr(msg, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                print("block=========\n",block,"\n\n")
                if block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif "text" in block:
                    parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(raw)


router = APIRouter(prefix="/api/chat", tags=["chat"])


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    """聊天请求（由 Agent 内的模型自行决定是否、何时调用工具）"""
    message: str
    model: str = "deepseek-chat"
    session_id: Optional[str] = None


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
        print("chat===========Agent（由模型决定是否调用工具） \n", datetime.now(),"\n\n")
        session_id = (request.session_id or "").strip() or "default"
        agent = create_assistant()
        response = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config={"configurable": {"thread_id": session_id}},
        )
        reply = response["messages"][-1].content
        print("chat===========返回结果 \n", datetime.now(),"\n\n")

        return ChatResponse(reply=reply, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式接口：浏览器用 fetch + ReadableStream 一段段读。
    每行格式：data: <JSON 或纯文本>\\n\\n；结束发 data: [DONE]。
    正文用 JSON {\"content\": \"...\"} 包一层，避免模型输出里自带换行时弄断 SSE 行。
    """

    async def generate():
        try:
            session_id = (request.session_id or "").strip() or "default"
            agent = create_assistant()

            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": request.message}]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode="messages",
            ):
                token = stream_chunk_text(chunk)
                if token:
                    # 一行一个 JSON 对象，json.dumps把对象解析成字符串后前端按行解析后取 content，再交给 onChunk
                    yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 告诉 Nginx 等代理不要缓冲 SSE，否则前端要很久才收到第一包
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """清空会话历史（对话链内存 + Agent 检查点）"""
    clear_session_history(session_id)
    clear_agent_session(session_id)
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