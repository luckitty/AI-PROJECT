import re

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

# 用户在上轮已收到「是否导入高德专属地图」追问后的肯定/拒绝短句。
AFFIRM_PERSONAL_MAP_EXACT = frozenset(
    {
        "可以",
        "好的",
        "好",
        "行",
        "要",
        "需要",
        "嗯",
        "好啊",
        "可以的",
        "ok",
        "OK",
    }
)

DECLINE_PERSONAL_MAP_SUBSTRINGS = (
    "不用",
    "不需要",
    "不必",
    "算了",
    "不要",
    "下次",
    "先不",
    "暂不",
)


# 首轮单轮且句极短时，高概率寒暄，可直接关 RAG/工具/攻略分支，避免为「你好」调路由模型。
GREETING_EXACT = frozenset(
    {
        "你好",
        "您好",
        "在吗",
        "在不在",
        "hi",
        "hello",
        "哈喽",
        "嗨",
        "早上好",
        "中午好",
        "晚上好",
    }
)

# 命中且句子里含下列词之一时，倾向「要攻略/行程」而非纯算路；与通勤正则搭配用于消歧。
# 注意：不用裸「几天」，避免「未来几天天气」等被误判为行程天数。
TRAVEL_GUIDE_INTENT_SUBSTRINGS = (
    "攻略",
    "行程",
    "怎么玩",
    "玩几天",
    "玩什么",
    "几日游",
    "安排",
    "推荐",
    "景点",
    "美食",
    "住哪",
    "酒店",
    "去哪玩",
    "好玩",
    "必去",
    "一日游",
    "二日游",
    "三日游",
    "自由行",
    "跟团",
    "游记",
    "规划",
)

# 出现这些词时更像泛攻略/多日安排，不交给「纯通勤」短路由，交给模型看指代与上下文。
TRAVEL_PLANNING_EXCLUDE_COMMUTE = (
    "攻略",
    "行程规划",
    "行程安排",
    "几天几夜",
    "几日",
    "自由行攻略",
    "旅游计划",
    "亲子游",
    "蜜月",
)

# 多日预报、气温趋势类问法：规则层优先走高德 MCP 独立节点。
WEATHER_MCP_HINT_SUBSTRINGS = (
    "未来",
    "几天",
    "多日",
    "预报",
    "趋势",
    "一周",
    "这周",
)

# 高德 MCP 公交路径规划（须坐标 + 起终点城市）。
AMAP_MCP_TRANSIT_SUBSTRINGS = (
    "公交路径",
    "地铁路线",
    "换乘方案",
    "火车换乘",
    "跨城公交",
)

# 高德 MCP 专属地图（行程导入高德、生成唤端链接）。
AMAP_MCP_PERSONAL_MAP_SUBSTRINGS = (
    "专属地图",
    "导入高德",
    "生成地图",
    "行程地图",
    "唤端链接",
    "高德地图链接",
)

# 高德 MCP POI / 地理编码类意图关键词。
AMAP_MCP_POI_GEO_SUBSTRINGS = (
    "经纬度",
    "坐标",
    "定位",
    "geocode",
    "逆地理",
    "周边搜",
    "附近有什么",
    "附近哪",
    "搜一下",
    "找一下",
    "POI",
    "景点推荐",
    "餐厅推荐",
    "酒店推荐",
)

# 问实况/预报/穿衣等气象信息；规则层统一走高德 MCP（amap_mcp_weather）。
WEATHER_INTENT_SUBSTRINGS = (
    "天气",
    "气温",
    "温度",
    "下雨",
    "降雨",
    "降雪",
    "台风",
    "预报",
    "空气质量",
    "湿度",
    "风力",
    "穿衣",
    "带伞",
)


def looksLikeWeatherQuery(query: str) -> bool:
    """判断当前句是否主要在问某城市天气/预报（含「未来几天天气」类）。"""
    q = (query or "").strip()
    if len(q) < 4:
        return False
    return any(w in q for w in WEATHER_INTENT_SUBSTRINGS)


def looksLikeAmapMcpWeatherQuery(query: str) -> bool:
    """多日预报、未来几天天气等高德 MCP 更合适的天气问法。"""
    q = (query or "").strip()
    if not looksLikeWeatherQuery(q):
        return False
    return any(w in q for w in WEATHER_MCP_HINT_SUBSTRINGS)


def looksLikeAmapMcpPoiGeoQuery(query: str) -> bool:
    """判断当前句是否偏向 POI 搜索、地理编码、周边搜等高德 MCP 能力。"""
    q = (query or "").strip()
    if len(q) < 4:
        return False
    if looksLikeCommuteOnly(q):
        return False
    return any(w in q for w in AMAP_MCP_POI_GEO_SUBSTRINGS)


def looksLikeAmapMcpTransitQuery(query: str) -> bool:
    """已知经纬度或明确要 MCP 公交/地铁/火车换乘方案时走高德 MCP。"""
    q = (query or "").strip()
    if len(q) < 4:
        return False
    # 含具体坐标且问公交/地铁/换乘，优先 MCP 公交路径规划。
    has_coords = bool(re.search(r"\d+\.\d+\s*,\s*\d+\.\d+", q))
    if has_coords and any(w in q for w in ("公交", "地铁", "换乘", "火车")):
        return True
    return any(w in q for w in AMAP_MCP_TRANSIT_SUBSTRINGS)


def looksLikeAmapMcpPersonalMapQuery(query: str) -> bool:
    """用户要把行程导入高德、生成专属地图链接。"""
    q = (query or "").strip()
    if len(q) < 4:
        return False
    return any(w in q for w in AMAP_MCP_PERSONAL_MAP_SUBSTRINGS)


def looksLikeAffirmPersonalMapOffer(state: dict, query: str) -> bool:
    """
    上一轮已追问是否导入高德专属地图，且用户本轮明确同意。
    """
    if not state.get("pending_amap_personal_map_offer"):
        return False
    q = (query or "").strip()
    if not q:
        return False
    if q in AFFIRM_PERSONAL_MAP_EXACT:
        return True
    if len(q) > 28:
        return False
    affirm_hints = ("可以", "好的", "帮我", "导入", "生成", "要生成", "要导入", "麻烦", "生成吧")
    return any(h in q for h in affirm_hints)


def looksLikeDeclinePersonalMapOffer(state: dict, query: str) -> bool:
    """用户拒绝生成高德专属地图。"""
    if not state.get("pending_amap_personal_map_offer"):
        return False
    q = (query or "").strip()
    if not q or len(q) > 24:
        return False
    return any(d in q for d in DECLINE_PERSONAL_MAP_SUBSTRINGS)


def looksLikeAmapMcpQuery(query: str) -> bool:
    """综合判断是否需要进入高德 MCP 独立节点。"""
    q = (query or "").strip()
    return (
        looksLikeAmapMcpWeatherQuery(q)
        or looksLikeAmapMcpPoiGeoQuery(q)
        or looksLikeAmapMcpTransitQuery(q)
        or looksLikeAmapMcpPersonalMapQuery(q)
    )


def looksLikeCommuteOnly(query: str) -> bool:
    """
    判断当前句是否高概率仅为「两地怎么走/公共交通/打车」类算路需求。
    故意偏保守：有强攻略/多日安排语料时不视为纯通勤，避免误跳过 LLM。
    """
    q = (query or "").strip()
    if len(q) < 4:
        return False
    if any(x in q for x in TRAVEL_PLANNING_EXCLUDE_COMMUTE):
        return False
    # 强攻略向且无明显「到…怎么走」结构时，不当作纯通勤。
    if any(x in q for x in TRAVEL_GUIDE_INTENT_SUBSTRINGS) and not re.search(
        r"(从|自).{1,22}到.{1,22}|.{2,18}到.{2,18}",
        q,
    ):
        return False

    if re.search(r"(导航|算路|路径规划|高德|地图).{0,6}(一下|吗|呢|？|\?)?$", q):
        return True
    if re.search(
        r"(从|自)[^，。！？\n]{1,22}到[^，。！？\n]{1,22}(怎么|如何)(走|去|坐车|乘车|搭车)",
        q,
    ):
        return True
    if re.search(
        r"[^，。！？\n]{2,18}到[^，。！？\n]{2,18}(怎么|如何)(走|去|坐车|乘车|搭车)",
        q,
    ):
        return True
    if re.search(
        r"去[^，。！？\n]{1,10}(怎么走|怎么去|地铁|公交|打车)",
        q,
    ):
        return True
    if re.search(
        r"(地铁|公交|步行|驾车|开车|打车).{0,10}(怎么|如何|哪|几路|多久|多远|线路)",
        q,
    ):
        return True
    # 多轮里指代前文路段 + 明确通勤问法：避免「这段行程里提到地铁」误进算路。
    if any(h in q for h in NEED_HISTORY_HINT_SUBSTRINGS) and re.search(
        r"(怎么走|怎么去|怎么坐|坐地铁|乘公交|打车|导航|多久到|多远|几站路)",
        q,
    ):
        return True
    return False


def looksLikeTravelGuideIntentGeneral(query: str) -> bool:
    """
    判断当前句是否明显在要「旅游攻略 / 行程 / 玩法推荐」类内容（不限定六城）。

    用于与六城 RAG 分支配合：命中时本轮应拉用户长期记忆，便于结合偏好做攻略。
    """
    q = (query or "").strip()
    if len(q) < 5:
        return False
    if looksLikeCommuteOnly(q):
        return False
    if looksLikeWeatherQuery(q):
        return False
    return any(k in q for k in TRAVEL_GUIDE_INTENT_SUBSTRINGS)


def looksLikeSixCityGuideIntent(query: str) -> bool:
    """六城白名单内且句意明显偏攻略/行程（非纯通勤、非天气）时，可规则给出 RAG+攻略体。"""
    q = (query or "").strip()
    if not hit_allowed_travel_city(q):
        return False
    return looksLikeTravelGuideIntentGeneral(q)


def infer_router_plan_by_rules(state: dict, query: str) -> dict | None:
    """
    高置信路由：返回含五键的 plan 字典时可跳过路由 LLM；返回 None 表示交给模型判断。
    与 build_router_prompt 的边界尽量对齐：仅覆盖明显可模式化的类。
    """
    q = (query or "").strip()
    msgs = state.get("messages") or []

    # 攻略后追问专属地图：用户短句确认或拒绝，优先于其它规则。
    if looksLikeDeclinePersonalMapOffer(state, q):
        return {
            "need_rag": False,
            "need_tool": False,
            "need_amap_mcp": False,
            "need_memory": False,
            "need_travel_itinerary": False,
            "need_history": False,
            "decline_amap_personal_map": True,
        }
    if looksLikeAffirmPersonalMapOffer(state, q):
        return {
            "need_rag": False,
            "need_tool": False,
            "need_amap_mcp": True,
            "need_memory": False,
            "need_travel_itinerary": False,
            "need_history": False,
            "confirm_amap_personal_map": True,
        }

    # 首轮寒暄：不拉 RAG、不算路、不要攻略长模板。
    if len(msgs) <= 1 and q in GREETING_EXACT:
        return {
            "need_rag": False,
            "need_tool": False,
            "need_amap_mcp": False,
            "need_memory": False,
            "need_travel_itinerary": False,
            "need_history": False,
        }

    # 纯通勤算路：高德 MCP（geo + transit 等），勿进旅游 RAG 与攻略长模板。
    if looksLikeCommuteOnly(q):
        return {
            "need_rag": False,
            "need_tool": False,
            "need_amap_mcp": True,
            "need_memory": False,
            "need_travel_itinerary": False,
            "need_history": True,
        }

    # 高德 MCP：天气、POI 搜、地理编码、公交路径规划等。
    if looksLikeAmapMcpQuery(q):
        return {
            "need_rag": False,
            "need_tool": False,
            "need_amap_mcp": True,
            "need_memory": False,
            "need_travel_itinerary": False,
            "need_history": False,
        }

    # 天气（含实况与多日预报）：统一走高德 MCP，勿进旅游 RAG 与攻略长模板。
    if looksLikeWeatherQuery(q):
        return {
            "need_rag": False,
            "need_tool": False,
            "need_amap_mcp": True,
            "need_memory": False,
            "need_travel_itinerary": False,
            "need_history": False,
        }

    if looksLikeSixCityGuideIntent(q):
        return {
            "need_rag": True,
            "need_tool": False,
            "need_amap_mcp": False,
            # 攻略类问题结合用户长期偏好，与 need_rag 同时为真时由图编排先 memory 再 rag。
            "need_memory": True,
            "need_travel_itinerary": True,
            "need_history": True,
        }

    # 非六城但明显要攻略/行程：不走本地旅游 Milvus，仍拉长期记忆并走终稿攻略分支。
    if looksLikeTravelGuideIntentGeneral(q) and not hit_allowed_travel_city(q):
        return {
            "need_rag": False,
            "need_tool": False,
            "need_amap_mcp": False,
            "need_memory": True,
            "need_travel_itinerary": True,
            "need_history": True,
        }

    return None


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
