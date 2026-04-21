import json
import re
from typing import List
from pathlib import Path

from core.llm import get_llm
from rag.travel_cache_retriever import is_food_focus_query

travel_poi_cache_path = (
    Path(__file__).resolve().parents[1] / "data" / "travel_poi_cache.json"
)


def load_travel_poi_cache_names():
    """
    加载本地旅游 POI 缓存，并返回景点/餐厅名称集合用于候选过滤。
    """
    attraction_names = set()
    restaurant_names = set()
    if not travel_poi_cache_path.exists():
        return attraction_names, restaurant_names

    with travel_poi_cache_path.open("r", encoding="utf-8") as cache_file:
        cache_data = json.load(cache_file)

    # 使用缓存里的名称列表作为白名单，只有命中白名单的候选才会进入最终列表。
    attraction_names = {item for item in cache_data["attraction"]}
    restaurant_names = {item for item in cache_data["restaurant"]}
    return attraction_names, restaurant_names


cached_attraction_names, cached_restaurant_names = load_travel_poi_cache_names()


def build_itinerary_format_instruction(query: str) -> str:
    """
    生成旅游攻略输出格式约束，要求回答按天给出可执行路线与餐饮建议。
    """
    if is_food_focus_query(query):
        return (
            "【美食输出要求（必须遵守）】\n"
            "你现在是“美食推荐博主”，请严格执行以下规则，不要输出与规则无关的泛化攻略。\n\n"
            "请先根据目的地与下方参考材料整理当地美食。\n"
            "把菜品分类，例如当地特色菜、京菜、川菜、粤菜等（按材料里实际出现的内容选，不要空泛罗列菜系名）。\n"
            "然后根据口味、价格区间、地理位置、口碑等把材料里出现的美食尽量列全，并做推荐度排序（⭐️ 越多越推荐）。\n"
            "若有具体餐厅名请写出餐厅名；若无则给可执行的美食建议。\n"
            "若有具体菜名请写出菜名；若无则给品类或方向建议。\n"
        )
    return (
        "你现在是“旅游路线编排助手”，请严格执行以下规则，不要输出与规则无关的泛化攻略。\n\n"
        "【行程输出要求（必须遵守）】\n"
        "请先根据目的地规模、景点密度和检索材料，给出“游玩天数”（例如 2-3 天 / 4-5 天），"
        "根据游玩路线选择1-2个最适合住宿的地方，给出酒店名称和价格区间。"
        "并给大致游玩概览；然后按这个建议天数输出逐日路线。\n\n"
        "【每日行程结构（每一天必须严格遵守）】\n"
        "先来解锁今天的美食，开启元气满满的一天\n"
        "- 给出 2 家餐厅\n"
        "游玩安排：\n"
        "- 给出 1-3 个景点，说明原因，游玩时长，景点特色，是否需要门票，门票价格。\n"
        "晚餐安排：补充能量\n"
        "- 给出 2 家餐厅，说明是否需要排队、最佳到店时间。\n"
        "餐后漫游 推荐一个闲逛地方"
        "每天出片点提醒：至少给 1-2 个适合拍照出片的位置和推荐时间段。\n\n"
        "行程结束：总结今天行程，如果体力消耗严重则取消餐后漫游。\n"
        "【全局要求】\n"
        "- 默认9点半出门，若要必须早出门请说明理由\n"
        "- 每天的餐食要做到不重复 比如第一天吃烤鸭第二天的推荐里面就不要再有烤鸭了\n"
        "- 每一天末尾补充：说明景点数量、步行/交通时长、就餐节奏是否过紧。若过紧则减少景点\n"
        "- 餐厅推荐必须解释“推荐理由”（口味、位置、口碑、与路线顺路程度）。\n"
        "- 不要只给景点清单，必须给可执行的时间段与顺序。"
    )


travel_propmt = """
你是一个旅游行程规划助手。
"请先根据目的地规模、景点密度和检索材料，给出“游玩天数”（例如 2-3 天 / 4-5 天），"
"并给大致游玩概览；然后按这个建议天数输出逐日路线。\n\n"
请根据提供的候选景点和餐厅，为用户生成一个“结构化且有顺序”的旅游行程。

【要求】
1. 必须按天输出，每天一个游玩主题
2. 每一天的行程必须是有序 sequence（表示游玩顺序）
3. 排序规则：
   - 先吃首餐，再游玩景点，最后吃晚餐
   - 路线应尽量连续，不走回头路
4. 只能使用候选列表中的POI，不允许编造

【输出格式】
严格输出JSON：

{
  "days": [
    {
      "day": 1,
      "theme": "游玩主题",
      "sequence": [
        {
            "name": "餐厅名",
            "type": "restaurant",
            "start_time": ""
        },
        {
            "name": "景点名",
            "type": "attraction",
            "duration": "小时h"
        },
        {
            "name": "餐厅名",
            "type": "restaurant",
            "start_time": ""
        },
        {
            "name": "餐后漫游地点",
            "type": "attraction",
        }
      ]
    }
  ]
}

【候选景点】
{{retrieved_attractions}}

【候选餐厅】
{{retrieved_restaurants}}"""


def split_candidate_terms(raw_text: str) -> List[str]:
    """
    把 spots_text / foods_text 这类字符串切成候选词列表，去重并保持原始顺序。
    """
    text = (raw_text or "").strip()
    if not text:
        return []
    segments = re.split(r"[、，,；;。/\s]+", text)
    cleaned: List[str] = []
    seen = set()
    for segment in segments:
        item = segment.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def build_travel_poi_candidates(docs):
    """
    从 RAG 文档里提取结构化候选：景点列表 + 餐厅列表。
    """
    attraction_names: List[str] = []
    restaurant_names: List[str] = []
    seen_attraction = set()
    seen_restaurant = set()

    for doc in docs:
        metadata = doc.metadata or {}
        spots_text = str(metadata.get("spots_text") or "").strip()
        foods_text = str(metadata.get("foods_text") or "").strip()
        print("build_travel_poi_candidates===========spots_text \n\n", spots_text, "\n")
        print("build_travel_poi_candidates===========foods_text \n\n", foods_text, "\n")

        for name in split_candidate_terms(spots_text):
            if name not in cached_attraction_names:
                continue
            key = ( name)
            if key in seen_attraction:
                continue
            seen_attraction.add(key)
            attraction_names.append(
                json.dumps(
                    {
                        "name": name,
                        "type": "attraction",
                    },
                    ensure_ascii=False,
                )
            )
        for name in split_candidate_terms(foods_text):
            if name not in cached_restaurant_names:
                continue
            key = (name)
            if key in seen_restaurant:
                continue
            seen_restaurant.add(key)
            restaurant_names.append(
                json.dumps(
                    {
                        "name": name,
                        "type": "restaurant",
                    },
                    ensure_ascii=False,
                )
            )
    print("build_travel_poi_candidates===========attraction_names \n\n", attraction_names, "\n")
    print("build_travel_poi_candidates===========restaurant_names \n\n", restaurant_names, "\n")
    return {
        "retrieved_attractions": attraction_names[:40],
        "retrieved_restaurants": restaurant_names[:40],
    }


def build_travel_prompt_with_candidates(docs) -> str:
    """
    把 travel_prompt 里的候选占位符替换成 RAG 提取出的景点/餐厅列表。
    """
    candidates = build_travel_poi_candidates(docs)
    print("build_travel_prompt_with_candidates===========candidates \n\n", candidates, "\n")
    attractions_text = "\n".join(candidates["retrieved_attractions"]) or "[]"
    restaurants_text = "\n".join(candidates["retrieved_restaurants"]) or "[]"
    return (
        travel_propmt.replace("{{retrieved_attractions}}", attractions_text).replace(
            "{{retrieved_restaurants}}", restaurants_text
        )
    )


def generate_travel_draft(query: str, travel_material: str, docs) -> str:
    """
    基于旅游规则与素材生成结构化行程。
    """
    travel_prompt_with_candidates = build_travel_prompt_with_candidates(docs)
    prompt = f"""
{travel_prompt_with_candidates}

执行要求：
1) 所有行程结构与硬性规则只以“旅游工具素材”中的要求为准，不新增第二套规则。

用户问题：
{query}

旅游工具素材：
{travel_material}
"""
    llm = get_llm(streaming=False)
    msg = llm.invoke(prompt)
    answer = msg.content if hasattr(msg, "content") else str(msg)
    print("generate_travel_draft===========answer \n", answer, "\n")
    return (answer or "").strip()
