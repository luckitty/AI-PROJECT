import re
import time

from langgraph.constants import TAG_NOSTREAM

from core.llm import get_llm


def parse_router_plan(text: str) -> dict:
    """
    从模型输出中解析三个布尔开关，不依赖 json.loads。
    兼容 JSON 形态（如 "need_rag": false）以及 need_rag=false 等变体。
    """
    raw = text or ""
    plan = {"need_rag": False, "need_tool": False, "need_memory": False}
    for key in plan:
        match = re.search(rf'["\']?{key}["\']?\s*[:=]\s*(true|false)', raw, re.I)
        if match:
            plan[key] = match.group(1).lower() == "true"
    return plan


def planner_node(state):
    query = state["query"]
    # 旅游意图做显式规则短路：保证「旅游攻略」稳定进入 rag_travel -> search_travel 的固定链路。

    prompt = f"""你是任务路由器。仅根据用户当前这句话判断是否需要以下能力：
        1) need_rag: 明确是旅游攻略问题，返回 true
        2) need_tool: 需要外部工具/API/实时数据/执行动作；包括联网搜索(web_search)；
        3) need_memory: 需要读取或写入用户长期记忆

        判定规则：
        - 如果需要则返回 true，如果不需要则返回 false，不确定时优先返回 false
        - 仅输出一行 JSON 形态即可，不要解释，不要 Markdown 代码块
        - 字段必须完整且只包含这三个键

        用户输入：{query}
        输出：
        {{"need_rag": false, "need_tool": false, "need_memory": false}}"""

    # get_llm 是工厂函数；规划需要稳定可解析输出，关闭流式便于一次性取 content。
    # TAG_NOSTREAM：告诉 LangGraph 的 messages 流不要收录本节点 LLM 产出，避免路由原文进 SSE。
    # 路由只需一行 JSON，压低 max_tokens 可略减 planner 往返耗时。
    llm = get_llm(streaming=False, max_tokens=120, temperature=0)
    planner_invoke_started_at = time.perf_counter()
    msg = llm.invoke(prompt, config={"tags": [TAG_NOSTREAM]})
    planner_invoke_seconds = time.perf_counter() - planner_invoke_started_at
    print(
        "pipeline_timing===========planner_node LLM路由 invoke "
        f"耗时: {planner_invoke_seconds:.2f}s"
    )
    text = msg.content if hasattr(msg, "content") else str(msg)
    print("planner_node===========text \n", text, "\n")
    plan = parse_router_plan((text or "").strip())
    print("planner_node===========plan \n", plan, "\n")

    return {
        **state,
        "need_rag": bool(plan.get("need_rag", False)),
        "need_tool": bool(plan.get("need_tool", False)) ,
        "need_memory": bool(plan.get("need_memory", False)) ,
    }