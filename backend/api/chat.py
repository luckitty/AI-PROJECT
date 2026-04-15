"""
聊天 API 路由
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional
from langchain_core.messages import AIMessage
import json
import re
# 使用相对导入
from agents.assistant import (
    DEFAULT_SYSTEM_PROMPT,
    create_assistant,
    clear_agent_session,
)
from crawler.runner import run_travel_route_agent
from chains.chat_chain import clear_session_history

from memory.long_memory_guard import (
    start_long_memory_save_task,
)

from graph.orchestrator import AgentOrchestrator

# 图编排：planner → memory | rag | tool → response（与 create_assistant 二选一或分流使用）
agent_graph = AgentOrchestrator()

# 不向 SSE 推送这些节点的 LLM 输出；仅屏蔽名单，避免误伤 response（见 extract_langgraph_node_name）。
STREAM_HIDDEN_GRAPH_NODES = frozenset(
    {"planner", "memory", "rag", "tool", "save_memory"}
)


def extract_langgraph_node_name(metadata: dict) -> Optional[str]:
    """
    从 LangGraph 流式 metadata 解析当前节点短名。
    新版里 langgraph_node 可能是 list（如 [\"response\"]）或带路径的字符串，不能再用 ``!= \"response\"`` 字符串比较。
    """
    if not metadata:
        return None
    raw: Any = metadata.get("langgraph_node")
    if raw is None:
        raw = metadata.get("langgraph_node_path")
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        raw = raw[-1]
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    if not name:
        return None
    if "/" in name:
        name = name.split("/")[-1]
    return name


def looks_like_planner_route_json(text: str) -> bool:
    """metadata 异常时仍可能漏出 planner 的路由 JSON，按内容再挡一层。"""
    t = (text or "").strip()
    if not t.startswith("{"):
        return False
    try:
        data = json.loads(t)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    return "need_rag" in data and "need_tool" in data and "need_memory" in data


# 把 AIMessage.content 转成可展示的纯文本（str / 多模态 list 均支持）。
def message_content_to_text(content) -> str:
    """把 AIMessage.content 转成可展示的纯文本（str / 多模态 list 均支持）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
         # 提取所有 type 为 text 的块
        texts = [block.get("text", "") for block in content 
                 if isinstance(block, dict) and block.get("type") == "text"]
        return "".join(texts)
    return str(content)

# 流式输出
def stream_chunk_text(chunk) -> str:
    """
    解析 LangGraph ``stream_mode="messages"`` 的单次产出，得到应推给前端的文本。

    - 入参多为 ``(message, metadata)``，先取 ``message``。
    - 只输出 **AIMessage**（含流式用的 AIMessageChunk）：与 ``invoke`` 取最后一条助手回复语义一致。
    - 若 AIMessage 仅有 ``tool_calls``、无可见正文，视为「路由到工具」的中间态，不推送。
    - ``messages`` 模式会透出图中每一次 LLM 调用；非流式 ``invoke`` 也会在 ``on_llm_end`` 打出整段
      正文，因此 planner 的路由 JSON 会混进 SSE。metadata 里 ``langgraph_node`` 标明来源节点，
      只把 ``response`` 节点的文本推给前端。
    """
    if chunk is None:
        return ""

    msg = chunk
    metadata = None
    # LangGraph 对 messages 模式约定为 (BaseMessage, metadata_dict)。
    if isinstance(chunk, tuple) and len(chunk) >= 2 and isinstance(chunk[1], dict):
        msg = chunk[0]
        metadata = chunk[1]

    if metadata is not None:
        node_name = extract_langgraph_node_name(metadata)
        if node_name in STREAM_HIDDEN_GRAPH_NODES:
            return ""

    if not isinstance(msg, AIMessage):
        return ""

    text = message_content_to_text(getattr(msg, "content", None))
    if looks_like_planner_route_json(text):
        return ""
    if getattr(msg, "tool_calls", None) and not text.strip():
        return ""
    return text


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
    route_plan: Optional[dict] = None


class HistoryRequest(BaseModel):
    """历史请求"""
    session_id: str


class TravelPlanRequest(BaseModel):
    """旅游路线请求：仅传小红书搜索用的关键词，由用户自由描述。"""
    keyword: str


def parse_destination_and_days(message: str) -> tuple[str, int]:
    """从输入文本中提取目的地和天数，形如“杭州三日游攻略”"""
    normalized_message = (message or "").strip()
    if not normalized_message:
        raise ValueError("请输入旅游请求，例如：杭州三日游攻略")

    # 目的地优先提取“xx三日游”前面的中文地名。
    destination_match = re.search(r"([\u4e00-\u9fa5]{2,12})(?:\s*)([一二三四五六七八九十\d]+)日游", normalized_message)
    if destination_match:
        destination = destination_match.group(1)
        day_token = destination_match.group(2)
    else:
        raise ValueError("未识别到目的地和天数，请使用“杭州三日游攻略”格式输入")

    day_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if day_token.isdigit():
        days = int(day_token)
    else:
        if day_token not in day_map:
            raise ValueError("未识别到有效天数，请使用“X日游”格式")
        days = day_map[day_token]
    days = max(1, min(days, 10))
    return destination, days


def derive_route_meta_from_keyword(keyword: str) -> tuple[str, int]:
    """从用户关键词中解析路线展示用的目的地与天数；无「X日游」时取首段中文地名并默认 3 天。"""
    text = (keyword or "").strip()
    if not text:
        return "旅行", 3
    try:
        return parse_destination_and_days(text)
    except ValueError:
        match = re.search(r"([\u4e00-\u9fa5]{2,12})", text)
        destination = match.group(1) if match else "旅行"
        return destination, 3


def format_route_plan_as_markdown(route_plan: dict) -> str:
    """把结构化路线结果转成前端可展示 Markdown。"""
    destination = route_plan.get("destination", "")
    days = route_plan.get("days", 0)
    top_spots = route_plan.get("top_spots", [])
    lines = [f"# {destination}{days}日游路线建议", ""]

    if top_spots:
        lines.append("## 高频景点参考")
        lines.append("、".join(top_spots[:10]))
        lines.append("")

    lines.append("## 每日行程")
    for day_item in route_plan.get("route", []):
        day_number = day_item.get("day", "")
        spots = day_item.get("spots", [])
        notes = day_item.get("notes", "")
        lines.append(f"### Day {day_number}")
        lines.append(f"- 推荐路线：{' -> '.join(spots) if spots else '待补充'}")
        if notes:
            lines.append(f"- 备注：{notes}")
        lines.append("")
    return "\n".join(lines).strip()


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
        print("response===========图编排响应 \n", reply, "\n\n")
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
                token = stream_chunk_text(chunk)
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


@router.post("/travel-plan", response_model=ChatResponse)
def travel_plan(request: TravelPlanRequest):
    """旅游路线生成：仅根据 keyword 抓取笔记并生成路线（keyword 即小红书搜索词）。"""
    try:
        keyword = (request.keyword or "").strip()
        if not keyword:
            raise HTTPException(status_code=400, detail="keyword 不能为空")
        destination, days = derive_route_meta_from_keyword(keyword)
        route_plan = run_travel_route_agent(
            keyword=keyword,
            destination=destination,
            days=days,
            headless=True,
            # 多滚几屏以便凑够约 50 条笔记卡片。
            max_scroll_rounds=14,
            max_note_count=50,
            # 强制实时抓取，避免命中旧缓存导致详情字段为空。
            use_cache=False,
            force_refresh=True,
            # 页面抓取优先复用本机已登录 Chrome，降低空结果概率。
            use_cdp=True,
            cdp_url="http://127.0.0.1:9222",
            # 明确使用页面抓取详情，不走 HTTP 接口降级链路。
            use_http_api=False,
        )
        print("travel_plan===========route_plan \n", route_plan, "\n")
        markdown_reply = format_route_plan_as_markdown(route_plan)
        return ChatResponse(reply=markdown_reply, session_id=None, user_id=None, route_plan=route_plan)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"旅游路线生成失败: {error}")


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