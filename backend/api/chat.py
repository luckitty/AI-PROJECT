"""
聊天 API 路由
"""
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional, Literal
import json
import asyncio

# 本模块异常写入服务端日志（含堆栈），与返回给前端的简短文案分离
logger = logging.getLogger(__name__)
from interruptController.interrupt_manager import interrupt_manager
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

from graph.chat_messages import last_assistant_text
from graph.orchestrator import AgentOrchestrator, MAX_HITL_RESUME_STEPS

# 图编排：planner → memory | rag | tool → response（与 create_assistant 二选一或分流使用）
agent_graph = AgentOrchestrator()


def interrupts_to_serializable(raw: Any) -> list:
    """
    将 LangGraph 的 ``Interrupt`` 序列（元组或列表）转成可 JSON 化的结构，供 REST / SSE 下发。
    """
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list = []
    for item in items:
        vid = getattr(item, "id", None)
        val = getattr(item, "value", item)
        out.append({"id": vid, "value": val})
    return out


def sse_payload_from_custom_stream_chunk(chunk: Any) -> str:
    """
    解析 LangGraph ``stream_mode="custom"`` 的单次产出。
    response / travel 节点通过 ``get_stream_writer`` 写入 ``{"content": "增量"}``，此处取出正文增量给 SSE。
    兼容部分版本以 ``("custom", payload)`` 元组形式吐出的情况。
    """
    if chunk is None:
        return ""
    if isinstance(chunk, tuple) and len(chunk) >= 2:
        payload = chunk[-1]
        if isinstance(payload, dict):
            return str(payload.get("content") or payload.get("message") or "")
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
    # 非空时表示恢复同 thread 上挂起的人机断点，对应 ``Command(resume=resume)``。
    resume: Optional[Any] = None
    # True：每个 interrupt() 挂起都交给客户端 resume；False（默认）：服务端自动 resume=True 跑完全程。
    human_breakpoints: bool = False


class ChatResponse(BaseModel):
    """聊天响应：人机断点模式下 status=interrupted，客户端再带同一 session_id 与 resume 继续。"""
    reply: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    status: Literal["completed", "interrupted"] = "completed"
    interrupts: Optional[list] = None


def graph_result_to_chat_response(result: dict, session_id: Optional[str], user_id: Optional[str]):
    """
    将 ``invoke`` 的完整返回拆成 ChatResponse：存在 ``__interrupt__`` 时表示停在人机断点。
    """
    intr = result.get("__interrupt__")
    if intr:
        return ChatResponse(
            reply=None,
            session_id=session_id,
            user_id=user_id,
            status="interrupted",
            interrupts=interrupts_to_serializable(intr),
        )
    return ChatResponse(
        reply=last_assistant_text(result.get("messages")) or "",
        session_id=session_id,
        user_id=user_id,
        status="completed",
        interrupts=None,
    )


class HistoryRequest(BaseModel):
    """历史请求"""
    session_id: str


def bind_user_active_session(user_id: Optional[str], session_id: Optional[str]):
    """
    绑定用户当前活跃会话；若检测到同一用户已有旧会话，先标记旧会话中断。
    """
    if user_id and session_id:
        interrupt_manager.register_user_session(user_id, session_id)


# ============ API 端点 ============

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送聊天消息"""
    try:
        session_id = request.session_id
        user_id = request.user_id
        if request.resume is not None and not session_id:
            raise HTTPException(
                status_code=400,
                detail="resume 恢复人机断点时必须提供 session_id（与 LangGraph thread_id 一致）",
            )
        if request.resume is None:
            # 同一用户新请求到来时，优先中断旧会话，避免刷新后旧请求继续占用资源。
            bind_user_active_session(user_id, session_id)
            # 新请求开始前清理 stop 标记，避免上一轮协作式中断影响本轮请求。
            if session_id:
                interrupt_manager.reset(session_id)
        # 使用 LangGraph 节点编排；遇 ``interrupt()`` 时返回 interrupted 与 payloads。
        result = agent_graph.run(
            request.message,
            user_id,
            session_id or "",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            resume=request.resume,
            human_breakpoints=request.human_breakpoints,
        )
        # print("response===========图编排响应 \n", reply, "\n\n")
        # 记忆写入放到后台，优先保证主请求低延迟返回。
        # start_long_memory_save_task(request.message, user_id)

        return graph_result_to_chat_response(result, session_id, user_id)

    except HTTPException:
        # 业务层 4xx 等不要记成服务端未处理异常，也不要改成 500
        raise
    except Exception as e:
        # 非流式接口：记录完整异常链，便于对照 session / Milvus / 模型调用排障
        logger.exception("chat 非流式请求失败 session_id=%s user_id=%s", session_id, user_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """
    SSE 流式接口：浏览器用 fetch + ReadableStream 一段段读。
    每行格式：data: <JSON 或纯文本>\\n\\n；结束发 data: [DONE]。
    正文用 JSON {\"content\": \"...\"} 包一层，避免模型输出里自带换行时弄断 SSE 行。
    """

    async def generate():
        stream_iterator = None
        session_id = request.session_id
        # 若在进入 try 后立刻异常，except 里仍能安全打日志
        user_id = None
        try:
            user_id = request.user_id
            if request.resume is not None and not session_id:
                yield f"data: {json.dumps({'error': 'resume 必须提供 session_id'}, ensure_ascii=False)}\n\n"
                return
            if request.resume is None:
                # 同一用户新请求到来时，优先中断旧会话，避免刷新后旧请求继续输出。
                bind_user_active_session(user_id, session_id)
                # 新流式请求开始前清理 stop 标记，确保只响应本轮 stop 操作。
                if session_id:
                    interrupt_manager.reset(session_id)
            # 连接一旦建立立刻推一包 typing，前端可马上占位助手气泡（体感接近 ChatGPT 首包）。
            yield f"data: {json.dumps({'typing': True}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
            # 使用 create_assistant 创建助手，调用助手的方法
            # agent = create_assistant()
            # for chunk in agent.stream(
            #     {"messages": [{"role": "user", "content": request.message}]},
            #     config={"configurable": {"thread_id": session_id}},
            #     stream_mode="messages",
            # ):

            # 流式：默认不在 SSE 里暴露 interrupt，而是在服务端连续 resume，直到产出 custom 正文。
            pending_resume = request.resume
            human_bp = request.human_breakpoints
            auto_resume_steps = 0
            while True:
                stream_iterator = agent_graph.stream(
                    request.message,
                    user_id,
                    session_id or "",
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    resume=pending_resume,
                )
                pending_resume = None
                stop_for_client_interrupt = False
                need_auto_resume = False
                # 人机断点自动 resume 时切勿在收到 __interrupt__ 后立刻 break 出 for：本轮 graph.stream 生成器
                # 尚未跑到 StopIteration，失引用时 CPython 会 close 该生成器 → GeneratorExit → pregel 里
                # ``except BaseException: run_manager.on_chain_error``，LangSmith 整条链标红。应 continue
                # 把当前迭代器自然耗尽再开下一轮 ``stream(Command(resume=True))``。勿显式 .close()，同理。
                try:
                    for chunk in stream_iterator:
                        # 客户端一旦中断（前端 AbortController.abort），尽快停止继续消耗模型流。
                        if await http_request.is_disconnected():
                            if session_id:
                                interrupt_manager.stop(session_id)
                            break
                        mode = None
                        payload = chunk
                        if isinstance(chunk, tuple) and len(chunk) == 2:
                            mode, payload = chunk
                        if (
                            mode == "updates"
                            and isinstance(payload, dict)
                            and "__interrupt__" in payload
                        ):
                            # 例如已超过自动 resume 次数并已下发 error 后，仅吞掉后续 updates 直到迭代结束。
                            if stop_for_client_interrupt:
                                continue
                            if human_bp:
                                intr_list = interrupts_to_serializable(
                                    payload["__interrupt__"]
                                )
                                yield f"data: {json.dumps({'type': 'interrupt', 'interrupts': intr_list}, ensure_ascii=False)}\n\n"
                                stop_for_client_interrupt = True
                                break
                            auto_resume_steps += 1
                            if auto_resume_steps > MAX_HITL_RESUME_STEPS:
                                yield f"data: {json.dumps({'error': '人机断点自动恢复次数超过上限'}, ensure_ascii=False)}\n\n"
                                stop_for_client_interrupt = True
                                # 仍尽量把本轮迭代耗完，减轻未耗尽生成器被 GC close 时的 GeneratorExit 噪音。
                                continue
                            need_auto_resume = True
                            # 勿 break：须迭代到本轮 stream 自然结束，见上方注释。
                            continue
                        if mode == "custom" or mode is None:
                            token = sse_payload_from_custom_stream_chunk(
                                payload if mode == "custom" else chunk
                            )
                            if token:
                                yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0)
                        if await http_request.is_disconnected():
                            if session_id:
                                interrupt_manager.stop(session_id)
                            break
                finally:
                    stream_iterator = None

                if stop_for_client_interrupt:
                    break
                if need_auto_resume:
                    pending_resume = True
                    continue
                break
            # 先发 [DONE]，再后台写长期记忆，避免 SSE 因写入耗时而一直 pending。
            if not await http_request.is_disconnected():
                yield "data: [DONE]\n\n"
            # start_long_memory_save_task(request.message, user_id)

        except asyncio.CancelledError:
            # 浏览器断连时，ASGI 服务器可能直接取消协程；按正常中断处理即可。
            # 这里也补 stop 标记，避免协程被取消后图内长调用继续跑完。
            if session_id:
                interrupt_manager.stop(session_id)
            return
        except Exception as e:
            # SSE 仍只下发简短错误串，完整堆栈只进服务端日志
            logger.exception(
                "chat 流式请求失败 session_id=%s user_id=%s",
                session_id,
                user_id,
            )
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # 不显式 close graph.stream 迭代器，避免 GeneratorExit 污染 LangSmith（见内层 while 注释）。
            stream_iterator = None

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


@router.get("/stop")
async def stop(session_id: str):
    """中断会话"""
    interrupt_manager.stop(session_id)
    return {"message": "stopped"}

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