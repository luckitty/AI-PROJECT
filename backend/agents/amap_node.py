import json

from amap.amap_travel_service import enrich_itinerary_with_amap_transit


def amap_node(state):
    """
    读取 travel_node 产出的结构化草稿，独立补充高德交通方案并回填 travel_context。
    """
    query = state["query"]
    travel_context = state.get("travel_context") or ""
    try:
        travel_payload = json.loads(travel_context)
    except (json.JSONDecodeError, TypeError, ValueError):
        # 若上游未返回合法 JSON，直接透传，后续由 response_node 统一兜底提示。
        return {
            **state,
            "travel_context": travel_context,
        }

    itinerary_structured = str(travel_payload.get("itinerary_structured") or "").strip()
    # print("amap_node===========itinerary_structured \n", itinerary_structured, "\n")
    if itinerary_structured:
        # 只增强结构化行程，素材字段保持原值，避免重复拼装规则文本。
        travel_payload["itinerary_structured"] = enrich_itinerary_with_amap_transit(
            itinerary_structured,
            query,
        )
    # print("amap_node===========travel_payload \n\n", travel_payload, "\n")

    return {
        **state,
        "travel_context": json.dumps(travel_payload, ensure_ascii=False),
    }
