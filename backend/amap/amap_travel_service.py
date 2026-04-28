import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from core.config import AMAP_KEY
from rag.travel_cache_retriever import detect_city_from_query

amap_poi_location_cache_path = (
    Path(__file__).resolve().parents[1] / "data" / "amap_poi_location_cache.json"
)


def load_amap_poi_location_cache() -> dict:
    """
    加载本地 POI 坐标缓存，减少重复 geocode 请求。
    """
    if not amap_poi_location_cache_path.exists():
        return {}
    try:
        with amap_poi_location_cache_path.open("r", encoding="utf-8") as cache_file:
            cache_data = json.load(cache_file)
        return cache_data if isinstance(cache_data, dict) else {}
    except Exception:
        return {}


def save_amap_poi_location_cache(cache_data: dict) -> None:
    """
    保存 POI 坐标缓存到本地文件，供后续请求离线复用。
    """
    try:
        amap_poi_location_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with amap_poi_location_cache_path.open("w", encoding="utf-8") as cache_file:
            json.dump(cache_data, cache_file, ensure_ascii=False, indent=2)
    except Exception:
        # 缓存写入失败不影响主流程，直接忽略。
        pass


def build_poi_cache_key(city_name: str, poi_name: str) -> str:
    """
    生成城市+POI 的唯一键，避免跨城同名地点互相污染。
    """
    return f"{str(city_name or '').strip()}::{str(poi_name or '').strip()}"



def extract_json_text(raw_text: str) -> str:
    """
    从模型输出中提取首个 JSON 对象文本，兼容前后混有解释文字的情况。
    """
    text = (raw_text or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    return (match.group(0) if match else "").strip()


def search_poi_location(keyword: str, city_name: str) -> dict | None:
    """
    用高德关键词搜索拿 POI 坐标，后续用于路径规划。
    """
    if not keyword or not AMAP_KEY:
        return None
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": AMAP_KEY,
        "keywords": keyword,
        "city": city_name,
        "citylimit": "true",
        "offset": 1,
        "page": 1,
        "extensions": "base",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception:
        return None
    if str(data.get("status")) != "1":
        return None
    pois = data.get("pois") or []
    if not pois:
        return None
    top_poi = pois[0]
    location = str(top_poi.get("location") or "").strip()
    if not location or "," not in location:
        return None
    return {
        "name": top_poi.get("name") or keyword,
        "location": location,
        "address": top_poi.get("address") or "",
    }


def batch_geocode_poi_locations(poi_names: list[str], city_name: str) -> dict:
    """
    批量把 POI 名称转换成坐标，优先减少网络请求次数。
    """
    clean_names = []
    seen = set()
    for item in poi_names or []:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        clean_names.append(name)
    if not clean_names or not AMAP_KEY:
        return {}

    persistent_cache = load_amap_poi_location_cache()
    updated_cache = dict(persistent_cache)
    found_from_cache = {}
    uncached_names = []
    for name in clean_names:
        cache_key = build_poi_cache_key(city_name, name)
        cached_poi = persistent_cache.get(cache_key)
        location = str((cached_poi or {}).get("location") or "").strip()
        if isinstance(cached_poi, dict) and location and "," in location:
            found_from_cache[name] = cached_poi
        else:
            uncached_names.append(name)

    # 高德 geocode 支持 batch=true + 管道符拼接 address，可一次请求多条地址。
    # 这里按 10 个一批切分，避免 URL 过长导致请求失败。
    location_map = dict(found_from_cache)
    chunk_size = 10
    url = "https://restapi.amap.com/v3/geocode/geo"
    for start in range(0, len(uncached_names), chunk_size):
        chunk = uncached_names[start : start + chunk_size]
        params = {
            "key": AMAP_KEY,
            "address": "|".join(chunk),
            "city": city_name,
            "batch": "true",
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception:
            data = {}
        geocodes = data.get("geocodes") if isinstance(data, dict) else None
        if str(data.get("status")) == "1" and isinstance(geocodes, list):
            for index, geocode in enumerate(geocodes):
                if index >= len(chunk):
                    break
                location = str((geocode or {}).get("location") or "").strip()
                if not location or "," not in location:
                    continue
                poi_item = {
                    "name": chunk[index],
                    "location": location,
                    "address": (geocode or {}).get("formatted_address") or "",
                }
                location_map[chunk[index]] = poi_item
                updated_cache[build_poi_cache_key(city_name, chunk[index])] = poi_item

    # 批量漏掉的再单点补齐，兼顾速度和命中率。
    for name in uncached_names:
        if name in location_map:
            continue
        poi_info = search_poi_location(name, city_name)
        if poi_info:
            location_map[name] = poi_info
            updated_cache[build_poi_cache_key(city_name, name)] = poi_info
    if updated_cache != persistent_cache:
        save_amap_poi_location_cache(updated_cache)
    return location_map


def plan_transit_between_locations(origin: str, destination: str, city_name: str) -> dict | None:
    """
    用高德公交综合规划计算两点通勤方案，并直接归一为打车/步行/地铁三类结果。
    """
    if not origin or not destination or not AMAP_KEY:
        return None
    url = "https://restapi.amap.com/v3/direction/transit/integrated"
    params = {
        "key": AMAP_KEY,
        "origin": origin,
        "destination": destination,
        "city": city_name,
        "cityd": city_name,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception:
        return None
    if str(data.get("status")) != "1":
        return None
    route = data.get("route") or {}
    transits = route.get("transits") or []
    if not transits:
        return None
    best = transits[0]
    # 只用 transit 接口结果做模式判断，不再额外请求步行/驾车接口。
    segments = best.get("segments") or []
    has_subway = False
    has_taxi = False
    walking_distance_m = 0.0
    total_distance_m = 0.0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        walking = segment.get("walking") or {}
        try:
            walking_distance_m += float(walking.get("distance") or 0)
        except (TypeError, ValueError):
            pass
        bus = segment.get("bus") or {}
        buslines = bus.get("buslines") or []
        for line in buslines:
            line_item = line or {}
            line_name = str(line_item.get("name") or "").strip()
            if any(token in line_name for token in ("地铁", "号线")):
                has_subway = True
        taxi = segment.get("taxi") or {}
        try:
            has_taxi = has_taxi or float(taxi.get("distance") or 0) > 0
        except (TypeError, ValueError):
            pass
    try:
        total_distance_m = float(best.get("distance") or 0)
    except (TypeError, ValueError):
        total_distance_m = 0.0

    # 优先级：短距离步行 > 地铁 > 步行（大部分路程都在走）> 打车。
    mode = "taxi"
    # 城市核心景区里很多点位本身就很近，transit 接口也可能给出 taxi 结果；
    # 这里对短链路直接判定为步行，避免出现“打车xx分钟（其实可步行）”的反直觉文案。
    if total_distance_m > 0 and total_distance_m <= 1800:
        mode = "walking"
    elif has_subway:
        mode = "subway"
    elif total_distance_m > 0 and walking_distance_m >= total_distance_m * 0.7:
        mode = "walking"
    elif has_taxi:
        mode = "taxi"

    return {
        "mode": mode,
        "distance_m": best.get("distance"),
        "duration_s": best.get("duration"),
        "cost_cny": best.get("cost"),
        "walking_distance_m": best.get("walking_distance"),
    }


def parse_seconds_to_minutes(duration_s) -> int:
    """
    将秒数转成分钟并四舍五入，解析失败时返回 0。
    """
    try:
        return int(round(float(duration_s) / 60.0))
    except (TypeError, ValueError):
        return 0


def build_transport_text(transit_plan: dict) -> str:
    """
    把高德结构化交通结果整理成可直接写入 sequence[].transport 的文本。
    当前策略返回「交通方式 + 大约时长」，但不暴露线路与换乘细节。
    """
    if not isinstance(transit_plan, dict):
        return ""
    mode = str(transit_plan.get("mode") or "").strip()
    duration_minutes = parse_seconds_to_minutes(transit_plan.get("duration_s"))
    if duration_minutes <= 0:
        return ""
    if mode == "taxi":
        return f"打车约{duration_minutes}分钟"
    if mode == "walking":
        return f"步行约{duration_minutes}分钟"
    if mode == "subway":
        return f"地铁约{duration_minutes}分钟"
    return f"约{duration_minutes}分钟"


def normalize_day_pois(day_item: dict) -> list[str]:
    """
    统一每日点位结构：仅使用 days[].pois，返回去空后的景点名列表。
    """
    if not isinstance(day_item, dict):
        return []
    pois = day_item.get("pois")
    if not isinstance(pois, list):
        return []
    cleaned = []
    for poi_name in pois:
        name = str(poi_name or "").strip()
        if name:
            cleaned.append(name)
    day_item["pois"] = cleaned
    return cleaned


def enrich_itinerary_with_amap_transit(raw_answer: str, query: str) -> str:
    """
    解析结构化行程并补充高德交通方案，失败时回退原文，避免影响主流程可用性。
    """
    json_text = extract_json_text(raw_answer)
    if not json_text:
        return raw_answer
    try:
        itinerary = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw_answer

    if not isinstance(itinerary, dict):
        return raw_answer
    days = itinerary.get("days")
    print("enrich_itinerary_with_amap_transit===========days \n", days, "\n")
    if not isinstance(days, list) or not days:
        return raw_answer

    city_name = detect_city_from_query(query) or ""
    if not city_name:
        # 城市缺失时不强行调用高德，避免跨城同名 POI 带来错误路线。
        return json.dumps(itinerary, ensure_ascii=False)

    # 先把每日结构统一成 pois，并批量准备景点坐标，减少逐条串行请求。
    all_poi_names = []
    for day in days:
        for item_name in normalize_day_pois(day if isinstance(day, dict) else {}):
            all_poi_names.append(item_name)
    poi_location_cache = batch_geocode_poi_locations(all_poi_names, city_name)
    route_tasks = []
    for day_index, day in enumerate(days):
        pois = normalize_day_pois(day if isinstance(day, dict) else {})
        if len(pois) < 2:
            continue
        for index in range(len(pois) - 1):
            start_name = pois[index]
            end_name = pois[index + 1]
            start_poi = poi_location_cache.get(start_name)
            end_poi = poi_location_cache.get(end_name)
            if not start_poi or not end_poi:
                continue
            route_tasks.append(
                {
                    "day_index": day_index,
                    "from_name": start_name,
                    "to_name": end_name,
                    "from": start_name,
                    "to": end_name,
                    "from_location": start_poi["location"],
                    "to_location": end_poi["location"],
                }
            )

    # 这里缓存的是「已归一后的交通方式结果」（打车/步行/地铁），不是原始分段明细。
    transport_plan_cache = {}
    if route_tasks:
        # 路线规划网络请求并发执行，避免多段行程串行等待导致总时延线性增长。
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {}
            for task in route_tasks:
                future = executor.submit(
                    plan_transit_between_locations,
                    task["from_location"],
                    task["to_location"],
                    city_name,
                )
                future_map[future] = id(task)
            for future in as_completed(future_map):
                task_key = future_map[future]
                try:
                    transport_plan_cache[task_key] = future.result()
                except Exception:
                    transport_plan_cache[task_key] = None

    # 按当前 JSON 结构回填：days[].transports 保存「上一站 -> 下一站」交通描述。
    for task in route_tasks:
        transport_plan = transport_plan_cache.get(id(task))
        if not transport_plan:
            continue
        day = days[task["day_index"]]
        transport_text = build_transport_text(transport_plan)
        if not transport_text:
            continue
        transports = day.get("transports")
        if not isinstance(transports, list):
            transports = []
            day["transports"] = transports
        transports.append(
            {
                "from": task["from_name"],
                "to": task["to_name"],
                "transport": transport_text,
            }
        )

    return json.dumps(itinerary, ensure_ascii=False)
