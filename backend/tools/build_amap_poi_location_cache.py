import argparse
import json
from pathlib import Path

from amap.amap_travel_service import (
    batch_geocode_poi_locations,
    build_poi_cache_key,
    load_amap_poi_location_cache,
)


travel_poi_cache_path = Path(__file__).resolve().parents[1] / "data" / "travel_poi_cache.json"


def load_all_poi_names(poi_cache_file: Path) -> list[str]:
    """
    从 travel_poi_cache.json 读取景点和餐厅名称，合并为去重后的 POI 列表。
    """
    if not poi_cache_file.exists():
        print("load_all_poi_names===========poi cache file not found:", poi_cache_file)
        return []
    with poi_cache_file.open("r", encoding="utf-8") as cache_file:
        cache_data = json.load(cache_file)
    attraction_names = cache_data.get("attraction") or []
    restaurant_names = cache_data.get("restaurant") or []
    all_names = []
    seen = set()
    for name in list(attraction_names) + list(restaurant_names):
        item = str(name or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        all_names.append(item)
    return all_names


def build_amap_poi_location_cache(city_name: str, poi_cache_file: Path) -> None:
    """
    按城市批量预热 POI 坐标缓存，减少线上行程生成阶段的高德查询耗时。
    """
    all_poi_names = load_all_poi_names(poi_cache_file)
    if not all_poi_names:
        print("build_amap_poi_location_cache===========no poi names loaded")
        return

    before_cache = load_amap_poi_location_cache()
    before_hit = 0
    for poi_name in all_poi_names:
        cache_key = build_poi_cache_key(city_name, poi_name)
        if cache_key in before_cache:
            before_hit += 1

    print("build_amap_poi_location_cache===========city_name:", city_name)
    print("build_amap_poi_location_cache===========poi_total:", len(all_poi_names))
    print("build_amap_poi_location_cache===========cached_before:", before_hit)

    result_map = batch_geocode_poi_locations(all_poi_names, city_name)
    success_count = len(result_map)

    after_cache = load_amap_poi_location_cache()
    after_hit = 0
    for poi_name in all_poi_names:
        cache_key = build_poi_cache_key(city_name, poi_name)
        if cache_key in after_cache:
            after_hit += 1

    print("build_amap_poi_location_cache===========success_count:", success_count)
    print("build_amap_poi_location_cache===========cached_after:", after_hit)
    print("build_amap_poi_location_cache===========failed_count:", max(0, len(all_poi_names) - success_count))


def main():
    """
    解析命令行参数并执行 POI 坐标全量预生成任务。
    """
    parser = argparse.ArgumentParser(description="全量预生成高德 POI 坐标缓存")
    parser.add_argument("--city", default="北京", help="POI 所属城市名（默认: 北京）")
    parser.add_argument(
        "--poi-cache",
        default=str(travel_poi_cache_path),
        help="POI 名称缓存文件路径（默认: backend/data/travel_poi_cache.json）",
    )
    args = parser.parse_args()
    build_amap_poi_location_cache(args.city, Path(args.poi_cache))


if __name__ == "__main__":
    main()
