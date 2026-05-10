import re
import time

from langgraph.constants import TAG_NOSTREAM

from core.llm import get_llm
from graph.chat_messages import format_conversation_for_prompt

# 仅当用户句子里包含下列城市之一时，才走本地旅游缓存 RAG；其余攻略类问题在 response 用攻略专用指令生成。
TRAVEL_RAG_CITY_ALLOWLIST = ("北京", "广州", "杭州", "上海", "西安", "长沙")


def hit_allowed_travel_city(query: str) -> bool:
    """判断用户输入是否命中允许走旅游 RAG 的六个城市（子串匹配）。"""
    q = query or ""
    return any(city in q for city in TRAVEL_RAG_CITY_ALLOWLIST)


# 当前句出现这些子串时，高概率依赖多轮上文（指代、改口径、续问）；规则命中则强制 need_history=true。
NEED_HISTORY_HINT_SUBSTRINGS = (
    "上面",
    "上文",
    "前文",
    "这段",
    "刚才",
    "前面说的",
    "你刚才",
    "改成",
    "改为",
    "换成",
    "删掉",
    "去掉",
    "再加上",
    "接着说",
    "还有吗",
    "同上",
    "按照你的",
    "按你的",
    "依你的",
    "那份攻略",
    "这个行程",
)

# 整句匹配的短反馈：多轮场景下通常承接上一轮，终稿需要上文（避免子串误伤「好不好」等）。
SHORT_ACK_EXACT = frozenset(
    {
        "嗯",
        "好",
        "好的",
        "可以",
        "行",
        "谢谢",
        "多谢",
        "没问题",
        "ok",
        "OK",
    }
)


def infer_need_history_by_rules(state: dict, query: str) -> bool | None:
    """
    规则推断终稿是否要带多轮历史。
    返回 None 表示不覆盖，沿用路由模型输出的 need_history；
    返回 True/False 表示高置信，直接写入 state。
    """
    q = (query or "").strip()
    msgs = state.get("messages") or []
    summary_nonempty = bool((state.get("conversation_summary") or "").strip())

    # 仅一轮用户消息且无滚动摘要：没有更早对话可延续，终稿不必挂完整历史。
    if len(msgs) <= 1 and not summary_nonempty:
        return False

    if any(h in q for h in NEED_HISTORY_HINT_SUBSTRINGS):
        return True

    # 明显的续接型短反馈：多轮里通常承接上一轮，终稿需要上文。
    if len(msgs) > 1 and q in SHORT_ACK_EXACT:
        return True

    # 「调整顺序」「改第一天」类改写行程结构，未列入关键词时也偏向要历史。
    if re.search(
        r"(改|换|调|调整).{0,6}(顺序|行程|安排|计划)|第[一二三四五六七八九十\d]+天",
        q,
    ):
        return True

    return None


def parse_router_plan(text: str) -> dict:
    """
    从模型输出中解析路由布尔开关，不依赖 json.loads。
    兼容 JSON 形态（如 "need_rag": false）以及 need_rag=false 等变体。
    """
    raw = text or ""
    plan = {
        "need_rag": False,
        "need_tool": False,
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
    """组装路由 LLM 单段提示：字段定义、示例与输出格式。"""
    cities = "、".join(TRAVEL_RAG_CITY_ALLOWLIST)
    return f"""你是任务路由器。判断用户是否需要下列能力（请结合下方多轮对话理解指代与意图）：
        {history_section}
        1) need_rag: **仅当**用户明确在问**旅游攻略/行程/怎么玩**且目的地涉及下列城市之一时为 true：
           {cities}。
           若问的是其它地区（如新疆、成都、深圳等）的攻略，或泛问旅游但未落在上述六城，一律 **false**。
        2) need_travel_itinerary: 用户要旅游攻略、行程规划、景点或美食推荐等（**含**非上述六城的目的地）时为 true；与旅游无关的闲聊、编程、解题等为 false。
           **例外**：用户只在问**两地交通路线/导航/怎么走**，且句子里能看成**两个具体地名之间的通勤**（如故宫到景山怎么走、从A打车到B多久）时，need_travel_itinerary 必须为 **false**（交给高德工具算路）。
           若对话里助手已给出行程/攻略：仅当用户是在问**行程结构**（几天怎么排、景点顺序要不要改、时间够不够、加减景点）时，才 need_travel_itinerary=true 且 need_tool=false。
           若用户问的是**通勤/交通**（怎么走、怎么去、地铁公交、打车、多远多久），即使用「上面」「这段」「刚才」指代前文里的某一段路，也必须 need_travel_itinerary=false、need_tool=true（起终点从对话里归纳，勿当成重新要一篇攻略）。
        3) need_tool: 需要外部工具/API/实时数据/执行动作；包括联网搜索(web_search)、**高德地图路线规划（两地怎么走、步行/打车/公交地铁）**；
        4) need_memory: 需要读取或写入用户长期记忆
        5) need_history: 终稿生成（response）是否需要附带多轮对话历史。
           true：用户依赖上文才能理解（指代「上面/刚才」、修改前文行程、同一任务的连续追问等）；
           false：仅凭当前一句即可独立作答（全新话题、与上文无关的技术问答等）。
           不确定时倾向 true。
           （说明：服务端会用规则在高置信场景覆盖本字段，你仍按语义照常输出即可。）

        判定规则：
        - 六城攻略示例：need_rag=true 且 need_travel_itinerary=true
        - 非六城攻略示例：need_rag=false，need_travel_itinerary=true
        - **两地通勤示例**：故宫到景山怎么走 → need_tool=true，need_rag=false，need_travel_itinerary=false
        - **延续上文行程结构示例**：对话里已有行程，用户问「第一天怎么安排比较顺」→ need_travel_itinerary=true，need_tool=false
        - **上文攻略后的通勤示例**：对话里已有行程，用户问「上面的路线怎么走/这段坐地铁怎么去」→ need_travel_itinerary=false，need_tool=true
        - 不确定时优先 false（need_history 除外：不确定时倾向 true）
        - 仅输出一行 JSON，不要解释，不要 Markdown 代码块
        - 字段必须完整且只包含这五个键

        （当前用户意图亦体现在对话最后一轮「用户：」中；若与下列单行冲突以对话为准）
        用户输入（便于对齐）：{query}
        输出：
        {{"need_rag": false, "need_travel_itinerary": false, "need_tool": false, "need_memory": false, "need_history": true}}"""


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
    return {
        "need_rag": need_rag,
        "need_tool": bool(plan.get("need_tool", False)),
        "need_memory": bool(plan.get("need_memory", False)),
        "need_history": need_history,
        "travel_itinerary_in_response": want_itinerary and not need_rag,
    }


def planner_node(state):
    query = state["query"]
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
