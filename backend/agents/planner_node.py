import json
from langgraph.constants import TAG_NOSTREAM

from core.llm import get_llm


def planner_node(state):
    query = state["query"]

    prompt = f"""你是任务路由器。仅根据用户当前这句话判断是否需要以下能力：
        1) need_rag: 需要从知识库/文档检索事实（本地知识库只涉及到王安宇、樊振东、李荣浩，其他的不要检索）
        2) need_tool: 需要外部工具/API/实时数据/执行动作，搜索信息时需要调用web_search工具
        3) need_memory: 需要读取或写入用户长期记忆

        判定规则：
        - 如果需要则返回 true，如果不需要则返回 false，不确定时优先返回 false
        - 仅输出 JSON，不要解释，不要 Markdown
        - 字段必须完整且只包含这三个键

        用户输入：{query}
        输出：
        {{"need_rag": false, "need_tool": false, "need_memory": false}}"""

    # get_llm 是工厂函数；规划需要稳定 JSON，关闭流式便于一次性取 content。
    # TAG_NOSTREAM：告诉 LangGraph 的 messages 流不要收录本节点 LLM 产出，避免路由 JSON 进 SSE。
    llm = get_llm(streaming=False)
    msg = llm.invoke(prompt, config={"tags": [TAG_NOSTREAM]})
    text = msg.content if hasattr(msg, "content") else str(msg)
    try:
        plan = json.loads((text or "").strip())
        print("planner_node===========plan \n", plan, "\n")
    except (json.JSONDecodeError, TypeError, ValueError):
        plan = {}

    return {
        **state,
        "need_rag": bool(plan.get("need_rag", False)),
        "need_tool": bool(plan.get("need_tool", False)),
        "need_memory": bool(plan.get("need_memory", False)),
    }