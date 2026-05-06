import json
from pathlib import Path

import requests

from core.config import AMAP_KEY

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


def search_poi_location(keyword: str, city_name: str) -> dict | None:
    """
    用高德关键词搜索拿 POI 坐标；供本地缓存构建脚本等工具链使用（不再参与旅游正文算路）。
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
