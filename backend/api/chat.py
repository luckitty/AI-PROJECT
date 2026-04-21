"""
聊天 API 路由
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional
import json
# 使用相对导入
from agents.assistant import (
    DEFAULT_SYSTEM_PROMPT,
    create_assistant,
    clear_agent_session,
)
from chains.chat_chain import clear_session_history

from memory.long_memory_guard import (
    start_long_memory_save_task,
)

from graph.orchestrator import AgentOrchestrator

# 图编排：planner → memory | rag | tool → response（与 create_assistant 二选一或分流使用）
agent_graph = AgentOrchestrator()

def sse_payload_from_custom_stream_chunk(chunk: Any) -> str:
    """
    解析 LangGraph ``stream_mode="custom"`` 的单次产出。
    response 节点通过 ``get_stream_writer`` 写入 ``{"content": "增量"}``，此处取出正文增量给 SSE。
    """
    if chunk is None:
        return ""
    if isinstance(chunk, dict):
        return str(chunk.get("content") or chunk.get("message") or "")
    return ""


router = APIRouter(prefix="/api/chat", tags=["chat"])


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    """聊天请求（由 Agent 内的模型自行决定是否、何时调用工具）"""
    message: str
    model: str = "deepseek-chat"
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class HistoryRequest(BaseModel):
    """历史请求"""
    session_id: str


# ============ API 端点 ============

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送聊天消息"""
    try:
        session_id = request.session_id
        user_id = request.user_id
        # 使用 create_assistant 创建助手，调用助手的方法
        # agent = create_assistant()
        # response = agent.invoke(
        #     {"messages": [{"role": "user", "content": request.message}]},
        #     config={"configurable": {"thread_id": session_id}},
        # )
        # reply = response["messages"][-1].content
       
        # 使用 LangGraph 节点编排（planner / memory / rag / tool / response）
        # 图编排路径也显式注入 create_assistant 的默认系统提示词。
        reply = agent_graph.run(
            request.message,
            user_id,
            session_id,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        # print("response===========图编排响应 \n", reply, "\n\n")
        # 记忆写入放到后台，优先保证主请求低延迟返回。
        # start_long_memory_save_task(request.message, user_id)

        return ChatResponse(reply=reply, session_id=session_id, user_id=user_id)

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
            session_id = request.session_id
            user_id = request.user_id
            # 使用 create_assistant 创建助手，调用助手的方法
            # agent = create_assistant()
            # for chunk in agent.stream(
            #     {"messages": [{"role": "user", "content": request.message}]},
            #     config={"configurable": {"thread_id": session_id}},
            #     stream_mode="messages",
            # ):

            # 流式接口改为走 LangGraph 编排，和非流式 run 保持同一条执行链路。
            for chunk in agent_graph.stream(
                request.message,
                user_id ,
                session_id ,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            ):
                token = sse_payload_from_custom_stream_chunk(chunk)
                if token:
                    # 一行一个 JSON 对象，json.dumps把对象解析成字符串后前端按行解析后取 content，再交给 onChunk
                    yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
            # 先发 [DONE]，再后台写长期记忆，避免 SSE 因写入耗时而一直 pending。
            yield "data: [DONE]\n\n"
            # start_long_memory_save_task(request.message, user_id)

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