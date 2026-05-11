import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from langchain.tools import tool
from requests.adapters import HTTPAdapter

from amap.amap_travel_service import resolve_route_endpoints
from core.config import AMAP_KEY

# 连接超时与读超时分离：避免单次高德接口拖住整轮对话过久。
REQUEST_TIMEOUT = (2.0, 6.0)

_tls = threading.local()


def thread_amap_session():
    """线程局部 Session，配合线程池复用连接；requests.Session 不建议跨线程共享。"""
    session = getattr(_tls, "amap_session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=6, pool_maxsize=6, max_retries=0)
        session.mount("https://", adapter)
        _tls.amap_session = session
    return session


def parse_lng_lat(loc: str):
    """高德 location 字符串「经度,纬度」解析为浮点元组。"""
    parts = str(loc or "").strip().split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def straight_line_km(loc_a: str, loc_b: str):
    """两点球面距离（公里），用于发现跨省误 geocode。"""
    pa = parse_lng_lat(loc_a)
    pb = parse_lng_lat(loc_b)
    if not pa or not pb:
        return None
    rad = math.pi / 180.0
    lon1, lat1 = pa[0] * rad, pa[1] * rad
    lon2, lat2 = pb[0] * rad, pb[1] * rad
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(max(0.0, h))))
    return 6371.0 * c


def meters_to_km_label(meters_raw: str | int | float | None) -> str:
    """把高德返回的米数转成中文可读距离。"""
    try:
        m = float(meters_raw)
    except (TypeError, ValueError):
        return "未知距离"
    if m >= 1000:
        return f"{m / 1000:.1f} 公里"
    return f"{int(m)} 米"


def seconds_to_min_label(seconds_raw: str | int | float | None) -> str:
    """把秒数转成「约 X 分钟」。"""
    try:
        sec = int(float(seconds_raw))
    except (TypeError, ValueError):
        return "未知时长"
    if sec < 60:
        return f"约 {sec} 秒"
    minutes = max(1, round(sec / 60))
    return f"约 {minutes} 分钟"


def direction_request_json(url: str, params: dict) -> dict | None:
    """请求高德路径类接口，成功返回整份 JSON，失败返回 None。"""
    try:
        response = thread_amap_session().get(url, params=params, timeout=REQUEST_TIMEOUT)
        data = response.json()
    except Exception:
        return None
    if str(data.get("status")) != "1":
        return None
    return data if isinstance(data, dict) else None


def summarize_walk_route(route_obj: dict | None) -> str:
    """步行路径规划：从 route.paths[0] 取距离与时长摘要。"""
    prefix = "【步行】"
    if not isinstance(route_obj, dict):
        return f"{prefix}未返回可行路径。"
    paths = route_obj.get("paths") or []
    if not paths:
        info = str(route_obj.get("info") or "").strip()
        hint = f"（{info}）" if info else ""
        return f"{prefix}无可行路径{hint}。"
    top = paths[0] if isinstance(paths[0], dict) else {}
    dist = meters_to_km_label(top.get("distance"))
    dur = seconds_to_min_label(top.get("duration"))
    return f"{prefix}全程 {dist}，预计 {dur}。"


def build_taxi_hint_block(origin_loc: str, dest_loc: str) -> str:
    """
    打车不做驾车路径 API（与用户要求一致）；用直线距离 + 粗略路况系数给用户量级感知。
    """
    dist_km = straight_line_km(origin_loc, dest_loc)
    if dist_km is None:
        return "【打车】请在网约车或出租车 App 输入起终点叫车；费用与耗时以平台实时计价为准。"
    road_scale = 1.35
    est_km = dist_km * road_scale
    est_min = max(5, round(est_km / 22 * 60))
    dist_label = f"{dist_km:.1f} 公里" if dist_km >= 1 else f"{max(1, int(dist_km * 1000))} 米"
    return (
        f"【打车】请在滴滴、高德打车等输入起终点叫车；两地直线距离约 {dist_label}，"
        f"市内车程粗略约 {est_min} 分钟（路况与计费以实际为准）。"
    )


def is_metro_busline(bl: dict) -> bool:
    """根据线路名称或类型判断是否为地铁（轨交）段。"""
    name = str(bl.get("name") or "")
    bl_type = str(bl.get("type") or "")
    blob = name + bl_type
    if "公交" in blob and "地铁" not in blob and "轨" not in blob:
        return False
    if "地铁" in blob or "轨交" in blob or "轨道交通" in blob:
        return True
    if "号线" in name:
        return True
    return False


def short_line_name(raw: str) -> str:
    """去掉括号内副标题，如 386路(xxx) → 386路。"""
    s = str(raw or "").strip()
    if not s:
        return ""
    return s.split("(")[0].strip()


def summarize_transit_with_modes(route_obj: dict | None) -> str:
    """
    公交/地铁摘要：总览 + 地铁线路序列 + 公交线路序列（来自首推换乘方案）。
    """
    if not isinstance(route_obj, dict):
        return "【公交/地铁】未返回可行方案。"
    transits = route_obj.get("transits") or []
    if not transits:
        info = str(route_obj.get("info") or "").strip()
        return f"【公交/地铁】无可行换乘方案。{info}".strip()
    top = transits[0] if isinstance(transits[0], dict) else {}
    dist = meters_to_km_label(top.get("distance"))
    dur = seconds_to_min_label(top.get("duration"))
    cost = top.get("cost")
    cost_part = ""
    if cost is not None and str(cost).strip() != "":
        cost_part = f"，预估票价约 {cost} 元"
    metro_names: list[str] = []
    bus_names: list[str] = []
    for seg in top.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        bus = seg.get("bus") or {}
        for bl in bus.get("buslines") or []:
            if not isinstance(bl, dict):
                continue
            short = short_line_name(str(bl.get("name") or ""))
            if not short:
                continue
            if is_metro_busline(bl):
                if short not in metro_names:
                    metro_names.append(short)
            else:
                if short not in bus_names:
                    bus_names.append(short)
    lines = [
        f"【公交/地铁】全程 {dist}，预计 {dur}{cost_part}。",
    ]
    if metro_names:
        lines.append(f"· 地铁：{' → '.join(metro_names)}")
    else:
        lines.append("· 地铁：本方案推荐路线中无地铁乘车段。")
    if bus_names:
        lines.append(f"· 公交：{' → '.join(bus_names)}")
    else:
        lines.append("· 公交：本方案推荐路线中无公交线路。")
    return "\n".join(lines)


def fetch_walk_block(origin_loc: str, dest_loc: str) -> str:
    url = "https://restapi.amap.com/v3/direction/walking"
    data = direction_request_json(url, {"key": AMAP_KEY, "origin": origin_loc, "destination": dest_loc})
    if not data:
        return "【步行】查询失败（可能超出步行规划范围或服务异常）。"
    return summarize_walk_route(data.get("route"))


def fetch_transit_block(origin_loc: str, dest_loc: str, city: str, cityd: str) -> str:
    url = "https://restapi.amap.com/v3/direction/transit/integrated"
    data = direction_request_json(
        url,
        {
            "key": AMAP_KEY,
            "origin": origin_loc,
            "destination": dest_loc,
            "city": city,
            "cityd": cityd,
        },
    )
    if not data:
        return "【公交/地铁】查询失败或服务不可用。"
    return summarize_transit_with_modes(data.get("route"))


def build_multimodal_report(
    origin_loc: str,
    dest_loc: str,
    city: str,
    destination_city: str,
    oname: str,
    dname: str,
) -> str:
    """并行拉取步行与公交/地铁；打车为本地提示（不调驾车规划接口）。"""
    cityo = str(city or "").strip()
    cityd = str(destination_city or "").strip() or cityo
    header = f"从「{oname}」到「{dname}」\n\n"
    taxi_line = build_taxi_hint_block(origin_loc, dest_loc)

    def run_walk():
        return fetch_walk_block(origin_loc, dest_loc)

    def run_transit():
        if not cityo:
            return "【公交/地铁】未查询（缺少起点城市 city，无法规划公交/地铁）。"
        return fetch_transit_block(origin_loc, dest_loc, cityo, cityd)

    ordered: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_map = {
            pool.submit(run_walk): "walk",
            pool.submit(run_transit): "transit",
        }
        for fut in as_completed(future_map):
            key = future_map[fut]
            try:
                ordered[key] = fut.result()
            except Exception as exc:
                ordered[key] = f"【{key}】查询异常: {exc}"
    blocks = [
        ordered.get("walk", ""),
        taxi_line,
        ordered.get("transit", ""),
    ]
    body = "\n".join(blocks)
    tail = (
        "\n\n【回复要求】综合以下步行、打车、公交/地铁信息给用户作简短说明；"
        "勿复述逐步转向或米级导航。"
    )
    return header + body + tail


@tool
def amap_route(
    origin_place: str,
    destination_place: str,
    city: str,
    destination_city: str,
) -> str:
    """
    高德多方式路线：步行规划、打车提示、公交/地铁换乘摘要（不调用驾车路径接口）。
    用户问两地怎么走、怎么去、交通方式时调用；须给出起点/终点地名及城市（同城时 destination_city 可与 city 相同）。
    """
    if not AMAP_KEY:
        return "高德密钥未配置，无法算路。"
    origin_place = str(origin_place or "").strip()
    destination_place = str(destination_place or "").strip()
    print("amap_route===========amap_route \n", origin_place, destination_place, city, destination_city, "\n")
    city = str(city or "").strip()
    destination_city = str(destination_city or "").strip()
    if not origin_place or not destination_place:
        return "起点或终点为空，无法算路。"
    origin_info, dest_info, city, destination_city = resolve_route_endpoints(
        origin_place, destination_place, city, destination_city
    )
    if not origin_info:
        return f"未能解析起点「{origin_place}」的坐标，请尝试补全城市或更具体地名。"
    if not dest_info:
        return f"未能解析终点「{destination_place}」的坐标，请尝试补全城市或更具体地名。"
    oloc = str(origin_info.get("location") or "").strip()
    dloc = str(dest_info.get("location") or "").strip()
    oname = str(origin_info.get("name") or origin_place).strip()
    dname = str(dest_info.get("name") or destination_place).strip()
    cityd = destination_city or city
    return build_multimodal_report(oloc, dloc, city, cityd, oname, dname)
