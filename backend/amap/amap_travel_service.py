import json
import math
from concurrent.futures import ThreadPoolExecutor
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


def normalize_city_token(cityname: str) -> str:
    """比较 POI 是否同城：去掉常见「市」后缀。"""
    s = str(cityname or "").strip()
    if s.endswith("市"):
        return s[:-1].strip() or s
    return s


def transit_city_param(cityname: str) -> str:
    """公交换乘接口常用「北京」这类市名。"""
    return normalize_city_token(cityname)


def straight_line_km_coords(loc_a: str, loc_b: str) -> float | None:
    """两点球面距离（公里）。"""
    parts_a = str(loc_a or "").strip().split(",")
    parts_b = str(loc_b or "").strip().split(",")
    if len(parts_a) != 2 or len(parts_b) != 2:
        return None
    try:
        lon1, lat1 = float(parts_a[0]), float(parts_a[1])
        lon2, lat2 = float(parts_b[0]), float(parts_b[1])
    except ValueError:
        return None
    rad = math.pi / 180.0
    lon1, lat1 = lon1 * rad, lat1 * rad
    lon2, lat2 = lon2 * rad, lat2 * rad
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(max(0.0, h))))
    return 6371.0 * c


def poi_result_core(poi: dict) -> dict:
    """与 search_poi_location 返回字段对齐。"""
    return {
        "name": str(poi.get("name") or "").strip(),
        "location": str(poi.get("location") or "").strip(),
        "address": str(poi.get("address") or "").strip(),
    }


def search_poi_candidates_nationwide(keyword: str, limit: int) -> list[dict]:
    """
    全国关键词检索 POI（不限城市），用于无 city 时在候选里找「同城且相距最近」的一对。
    """
    keyword = str(keyword or "").strip()
    if not keyword or not AMAP_KEY:
        return []
    url = "https://restapi.amap.com/v3/place/text"
    safe_limit = min(max(limit, 1), 25)
    params = {
        "key": AMAP_KEY,
        "keywords": keyword,
        "citylimit": "false",
        "offset": safe_limit,
        "page": 1,
        "extensions": "base",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception:
        return []
    if str(data.get("status")) != "1":
        return []
    pois = data.get("pois") or []
    out: list[dict] = []
    for item in pois:
        if not isinstance(item, dict):
            continue
        location = str(item.get("location") or "").strip()
        if not location or "," not in location:
            continue
        cityname = str(item.get("cityname") or "").strip()
        out.append(
            {
                "name": item.get("name") or keyword,
                "location": location,
                "address": item.get("address") or "",
                "cityname": cityname,
            }
        )
        if len(out) >= safe_limit:
            break
    return out


def pick_best_poi_pair_same_city(origin_kw: str, dest_kw: str, limit: int = 12):
    """
    在全国检索结果中，只保留两市名字段（规范化后）一致的候选对，取直线距离最短的一对。
    全国地名不靠枚举维护；依赖高德返回的 cityname + 几何约束消歧。
    """
    o_list = search_poi_candidates_nationwide(origin_kw, limit)
    d_list = search_poi_candidates_nationwide(dest_kw, limit)
    if not o_list or not d_list:
        return None, None, ""
    best_o: dict | None = None
    best_d: dict | None = None
    best_dist = float("inf")
    raw_city_label = ""
    for o in o_list:
        for d in d_list:
            oc = normalize_city_token(o.get("cityname") or "")
            dc = normalize_city_token(d.get("cityname") or "")
            if not oc or not dc or oc != dc:
                continue
            dist = straight_line_km_coords(o["location"], d["location"])
            if dist is None:
                continue
            if dist < best_dist:
                best_dist = dist
                best_o, best_d = o, d
                raw_city_label = str(o.get("cityname") or d.get("cityname") or "").strip()
    if best_o and best_d and raw_city_label:
        return poi_result_core(best_o), poi_result_core(best_d), raw_city_label
    return None, None, ""


def geocode_structured_address(address: str, city_name: str) -> dict | None:
    """
    将结构化地址或地名转为经纬度；可选 city_name 缩小同城歧义。
    返回与 search_poi_location 相近的字典，便于路径规划节点统一读取 location。
    """
    addr = str(address or "").strip()
    if not addr or not AMAP_KEY:
        return None
    url = "https://restapi.amap.com/v3/geocode/geo"
    params: dict = {"key": AMAP_KEY, "address": addr}
    city = str(city_name or "").strip()
    if city:
        params["city"] = city
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception:
        return None
    if str(data.get("status")) != "1":
        return None
    geocodes = data.get("geocodes") or []
    if not geocodes:
        return None
    top = geocodes[0]
    location = str((top or {}).get("location") or "").strip()
    if not location or "," not in location:
        return None
    return {
        "name": addr,
        "location": location,
        "address": (top or {}).get("formatted_address") or "",
    }


def resolve_place_location(place: str, city_name: str) -> dict | None:
    """
    把用户口中的地名解析成「经度,纬度」：优先同城 POI 关键词（景点命中率更高），失败再走地理编码。
    """
    keyword = str(place or "").strip()
    city = str(city_name or "").strip()
    if not keyword:
        return None
    if city:
        poi_hit = search_poi_location(keyword, city)
        if poi_hit:
            return poi_hit
    return geocode_structured_address(keyword, city)


def resolve_route_endpoints(origin_place: str, destination_place: str, city: str, destination_city: str):
    """
    解析起点、终点 POI 与城市上下文。
    用户未写 city：用全国 POI 候选按「同城 + 直线距离最短」配对，避免仅用 geocode 跨省串点。
    用户写了 city 仍异常（直线距离过大）：再尝试同城配对纠正。
    返回 (origin_info, dest_info, transit_city, transit_destination_city)。
    """
    o_kw = str(origin_place or "").strip()
    d_kw = str(destination_place or "").strip()
    city_o = str(city or "").strip()
    city_d_raw = str(destination_city or "").strip()
    city_d_resolve = city_d_raw or city_o

    def maybe_fix_far_points(o_inf, d_inf, c_out: str, d_out: str):
        """跨省误解析时常表现为两点直线距离过大；用同城 POI 配对若能显著拉近则替换。"""
        if not o_inf or not d_inf:
            return o_inf, d_inf, c_out, d_out
        oloc = str(o_inf.get("location") or "").strip()
        dloc = str(d_inf.get("location") or "").strip()
        dist = straight_line_km_coords(oloc, dloc)
        if dist is None or dist <= 85:
            return o_inf, d_inf, c_out, d_out
        po, pd, raw_city = pick_best_poi_pair_same_city(o_kw, d_kw)
        if not po or not pd or not raw_city:
            return o_inf, d_inf, c_out, d_out
        alt = straight_line_km_coords(po["location"], pd["location"])
        if alt is None or alt >= dist:
            return o_inf, d_inf, c_out, d_out
        tc = transit_city_param(raw_city)
        return po, pd, tc, tc

    if city_o:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fo = pool.submit(resolve_place_location, o_kw, city_o)
            fd = pool.submit(resolve_place_location, d_kw, city_d_resolve or city_o)
            o_inf, d_inf = fo.result(), fd.result()
        c_out = transit_city_param(city_o)
        d_out = transit_city_param(city_d_resolve or city_o)
        return maybe_fix_far_points(o_inf, d_inf, c_out, d_out)

    po, pd, raw_city = pick_best_poi_pair_same_city(o_kw, d_kw)
    if po and pd and raw_city:
        tc = transit_city_param(raw_city)
        return maybe_fix_far_points(po, pd, tc, tc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fo = pool.submit(resolve_place_location, o_kw, "")
        fd = pool.submit(resolve_place_location, d_kw, "")
        o_inf, d_inf = fo.result(), fd.result()
    return maybe_fix_far_points(o_inf, d_inf, "", "")


def search_poi_location(keyword: str, city_name: str = "", citylimit: bool = True) -> dict | None:
    """
    用高德关键词搜索拿 POI 坐标与 poi_id；专属地图等场景依赖 id 字段，勿仅用 geocode。
    citylimit=False 时放宽同城限制，用于有城市仍搜不到时的兜底。
    """
    keyword = str(keyword or "").strip()
    if not keyword or not AMAP_KEY:
        return None
    url = "https://restapi.amap.com/v3/place/text"
    params: dict = {
        "key": AMAP_KEY,
        "keywords": keyword,
        "citylimit": "true" if citylimit else "false",
        "offset": 1,
        "page": 1,
        "extensions": "base",
    }
    city = str(city_name or "").strip()
    if city:
        params["city"] = city
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
    poi_id = str(top_poi.get("id") or "").strip()
    return {
        "name": top_poi.get("name") or keyword,
        "location": location,
        "address": top_poi.get("address") or "",
        "poi_id": poi_id,
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
