import re
import time

from langgraph.constants import TAG_NOSTREAM

from agents.rule.planner_rules import (
    TRAVEL_RAG_CITY_ALLOWLIST,
    hit_allowed_travel_city,
    infer_need_history_by_rules,
    infer_router_plan_by_rules,
)
from core.llm import get_llm
from graph.chat_messages import format_conversation_for_prompt


def parse_router_plan(text: str) -> dict:
    """
    从模型输出中解析路由布尔开关，不依赖 json.loads。
    兼容 JSON 形态（如 "need_rag": false）以及 need_rag=false 等变体。
    """
    raw = text or ""
    plan = {
        "need_rag": False,
        "need_tool": False,
        "need_amap_mcp": False,
        "need_memory": False,
        "need_travel_itinerary": False,
    }
    for key in plan:
        match = re.search(rf'["\']?{key}["\']?\s*[:=]\s*(true|false)', raw, re.I)
        if match:
            plan[key] = match.group(1).lower() == "true"
    # need_history 单独解析：模型若漏输出该键，默认 true，行为与升级前一致。
    hm = re.search(r'["\']?need_history["\']?\s*[:=]\s*(true|false)', raw, re.I)
    if hm:
        plan["need_history"] = hm.group(1).lower() == "true"
    else:
        plan["need_history"] = True
    return plan


def format_router_history_section(state: dict) -> str:
    """滑动窗口内多轮对话块；无内容返回空串（路由只看意图时可仅用当前 query）。"""
    conversation_block = format_conversation_for_prompt(
        state.get("messages"),
        state.get("conversation_summary"),
        max_chars=3600,
        max_messages=12,
    ).strip()
    if not conversation_block:
        return ""
    return (
        "\n        【本轮为止的多轮对话（更长已在尾部保留，更早可能已省略）】\n"
        f"        {conversation_block}\n"
    )


def build_router_prompt(history_section: str, query: str) -> str:
    """组装路由 LLM 单段提示：场景判定、字段说明与输出格式。"""
    cities = "、".join(TRAVEL_RAG_CITY_ALLOWLIST)
    return f"""你是任务路由器。结合下方多轮对话理解指代与意图，为当前用户句输出六个布尔开关。

        {history_section}
        【全局约束】
        - need_tool 与 need_amap_mcp 互斥，同一轮至多一个为 true。
        - 除 need_history 外，各字段不确定时倾向 false；need_history 不确定时倾向 true。
        - 服务端可能用规则覆盖 need_history，你仍按语义照常输出。

        【按场景判定】（命中即采用对应组合，其余未列字段均为 false）
        A. 天气（实况/多日预报/穿衣建议等）→ need_amap_mcp=true
        B. 两地通勤算路（句中可看成两个具体地名之间的走/去/打车/多久；含指代上文某段路）→ need_amap_mcp=true
        C. 高德 MCP（POI 搜、周边搜、地理编码、公交路径规划、专属地图）→ need_amap_mcp=true
        D. 六城旅游攻略（目的地为 {cities}，且问攻略/行程/怎么玩/景点美食推荐）→ need_rag=true，need_travel_itinerary=true，need_memory=true（结合用户偏好）
        E. 非六城旅游攻略（如新疆、成都、深圳等，或泛问旅游但未落在上述六城）→ need_rag=false，need_travel_itinerary=true，need_memory=true
        F. 延续上文改行程结构（对话里已有行程，用户问几天怎么排、景点顺序、加减景点、时间够不够）→ need_travel_itinerary=true，need_memory=true
        G. 联网检索等非地图类外部信息 → need_tool=true
        H. 读取或写入用户长期记忆 → need_memory=true
        I. 与旅游/工具无关的闲聊、编程、解题等 → 各路由字段均为 false

        【字段说明】（场景未覆盖时再参考）
        - need_rag：仅 D 类为 true；其它地区攻略、天气、通勤、POI 均为 false。
        - need_travel_itinerary：D/E/F 为 true；纯天气、纯通勤、纯 POI 必须为 false。
        - need_tool：仅 G 为 true；含 web_search 联网检索。
        - need_amap_mcp：A/B/C 为 true；含天气、通勤算路、POI/地理、公交路径规划、专属地图。
        - need_memory：H 类为 true；D/E/F 类也必须为 true（攻略与行程调整需结合用户长期记忆）。
        - need_history：终稿是否需要多轮对话。true=依赖上文（指代、改行程、续问）；false=当前一句可独立作答。

        【易混边界】（仅保留规则层未覆盖的歧义 case）
        - 上文已有行程，问「上面的路线怎么走/这段坐地铁怎么去」→ B（need_amap_mcp=true），不是 F。
        - 「杭州有什么好吃的餐厅」→ C（POI 搜），不是 E（整篇攻略）。
        - 「北京未来几天天气」→ A，不是 D；「上海今天天气」→ A，不是 E。

        【输出】仅一行 JSON，不要解释，不要 Markdown 代码块；必须且只含六个键。
        当前用户句（对话最后一轮「用户：」与之冲突时以对话为准）：{query}
        {{"need_rag": false, "need_travel_itinerary": false, "need_tool": false, "need_amap_mcp": false, "need_memory": false, "need_history": true}}"""


def state_patch_from_router_plan(plan: dict, query: str, state: dict) -> dict:
    """
    将模型路由结果与六城白名单合并为写入 state 的开关（含 travel_itinerary_in_response）。
    need_history：规则高置信时覆盖 LLM，减少误判与多余 token。
    """
    hit_city = hit_allowed_travel_city(query)
    # 模型应对非六城输出 need_rag=false；此处再与 allowlist 与门，避免误判仍进 Milvus。
    need_rag = bool(plan.get("need_rag", False)) and hit_city
    want_itinerary = bool(plan.get("need_travel_itinerary", False))
    need_history_llm = bool(plan.get("need_history", True))
    ruled = infer_need_history_by_rules(state, query)
    need_history = need_history_llm if ruled is None else ruled
    if ruled is not None:
        print(
            "planner_node===========need_history rule_based 覆盖 LLM: ",
            ruled,
            "（模型原为 ",
            need_history_llm,
            "）\n",
        )
    need_tool = bool(plan.get("need_tool", False))
    need_amap_mcp = bool(plan.get("need_amap_mcp", False))
    # 终稿走攻略/改行程体时统一拉长期记忆，避免路由模型漏标 need_memory。
    need_memory = bool(plan.get("need_memory", False))
    if want_itinerary:
        need_memory = True
    # need_tool 与 need_amap_mcp 互斥：模型若同时输出 true，优先保留 MCP 分支。
    if need_tool and need_amap_mcp:
        need_tool = False

    patch = {
        "need_rag": need_rag,
        "need_tool": need_tool,
        "need_amap_mcp": need_amap_mcp,
        "need_memory": need_memory,
        "need_history": need_history,
        "travel_itinerary_in_response": want_itinerary and not need_rag,
        # 单回合开关：未命中时显式 false，避免 checkpoint 残留上一轮 true。
        "confirm_amap_personal_map": bool(plan.get("confirm_amap_personal_map", False)),
        "decline_amap_personal_map": bool(plan.get("decline_amap_personal_map", False)),
    }
    return patch


def planner_node(state):
    query = state["query"]

    # 规则路由：明显寒暄、纯通勤算路、六城明确攻略向时直接定 plan，跳过后续路由 LLM。
    rule_plan = infer_router_plan_by_rules(state, query)
    if rule_plan is not None:
        print(
            "planner_node===========route rule_based 跳过 LLM\n",
            rule_plan,
            "\n",
        )
        return {**state, **state_patch_from_router_plan(rule_plan, query, state)}

    history_section = format_router_history_section(state)
    prompt = build_router_prompt(history_section, query)

    # get_llm 为工厂；路由要稳定可解析，关闭流式一次取 content。
    # TAG_NOSTREAM：LangGraph messages 流不收录本节点产出，避免路由原文进 SSE。
    # 五个布尔路由字段；略放宽上限避免模型输出被截断。
    llm = get_llm(streaming=False, max_tokens=200, temperature=0)
    planner_invoke_started_at = time.perf_counter()
    msg = llm.invoke(prompt, config={"tags": [TAG_NOSTREAM]})
    planner_invoke_seconds = time.perf_counter() - planner_invoke_started_at
    print(
        "pipeline_timing===========planner_node LLM路由 invoke "
        f"耗时: {planner_invoke_seconds:.2f}s"
        "\n"
    )
    text = msg.content if hasattr(msg, "content") else str(msg)
    plan = parse_router_plan((text or "").strip())
    print("planner_node===========plan \n", plan, "\n")

    return {**state, **state_patch_from_router_plan(plan, query, state)}
