import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# 兼容从仓库根目录或 backend 目录直接执行脚本。
current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
backend_root = current_file.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

try:
    from rag.travel_loader import extract_spots
except ModuleNotFoundError:
    from backend.rag.travel_loader import extract_spots

GENERIC_WORDS = {
    "attraction": {"景点", "景区", "路线", "攻略", "旅行", "旅游", "打卡", "中心", "线路", "步行街", "公园", "广场", "古城", "大街"},
    "restaurant": {"美食", "小吃", "餐厅", "咖啡", "必吃", "早餐", "午餐", "晚餐", "夜宵", "口味", "菜品"},
}
RESTAURANT_NAME_PATTERN = re.compile(
    r"([\u4e00-\u9fa5A-Za-z]{2,20}(?:店|餐厅|饭店|饭馆|酒楼|小馆|咖啡馆|茶馆|酒吧))"
)
KNOWN_BEIJING_RESTAURANTS = {
    "四季民福",
    "南门涮肉",
    "姚记炒肝",
    "天兴居",
    "聚宝源",
    "大董烤鸭",
    "玺源居涮肉",
    "东兴顺铜锅涮肉",
    "菊儿人家",
    "刘阿妹鸡公煲",
}
RESTAURANT_VALID_SUFFIXES = (
    "店",
    "餐厅",
    "饭店",
    "饭馆",
    "酒楼",
    "小馆",
    "咖啡馆",
    "茶馆",
    "酒吧",
    "烤鸭",
    "涮肉",
    "炒肝",
    "炸酱面",
    "鸡公煲",
    "卤煮",
    # 老字号/品牌常见尾字，减少「便宜坊、天兴居、聚宝源」等被后缀规则误杀。
    "坊",
    "居",
    "源",
    "记",
    "斋",
    "德",
    "楼",
    "府",
    "春",
)
RESTAURANT_NOISE_WORDS = (
    "探店",
    "推荐",
    "攻略",
    "合集",
    "打车",
    "入住",
    "沿途",
    "很多",
    "还有",
    "这里",
    "这条",
    "这家",
    "有家",
    "顺便",
    "力推",
    "藏在",
    "藏着",
    "选择了",
    "什么时候",
    "酒店",
    "书店",
    "礼品店",
    "文创店",
    "精品店",
)
RESTAURANT_LOCATION_PREFIXES = (
    "故宫",
    "王府井",
    "前门",
    "南门",
    "北新桥",
    "簋街",
    "天坛",
    "天安门",
    "银锭桥",
    "锡拉胡同",
    "三元桥",
)
ATTRACTION_NOISE_WORDS = (
    "景点",
    "路线",
    "攻略",
    "预约",
    "大学",
    "胡同",
)


def split_candidate_terms(text: str) -> List[str]:
    """切分候选词文本，统一兼容常见分隔符。"""
    return [item.strip() for item in re.split(r"[、，,；;。/\s|]+", (text or "").strip()) if item.strip()]


def normalize_poi_name(text: str) -> str:
    """统一 POI 名称格式，减少空格和括号说明造成的重复。"""
    value = re.sub(r"[（(][^）)]{1,24}[）)]", "", (text or "").strip())
    value = re.sub(r"[\s\-_/|·•]+", "", value)
    return value.strip()


def load_note_list_from_file(cache_file: Path) -> List[dict]:
    """读取城市缓存文件并返回笔记数组。"""
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"[WARN] 读取失败: {cache_file} -> {error}")
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    notes = payload.get("notes") if isinstance(payload, dict) else None
    if isinstance(notes, list):
        return [item for item in notes if isinstance(item, dict)]
    return []


def collect_note_candidates(note: dict, city_name: str = "") -> Dict[str, List[str]]:
    """提取单条笔记候选：先结构化字段，再正文补抽。"""
    title = str(note.get("title") or "").strip()
    desc = str(note.get("desc") or "").strip()
    ocr_text = str(note.get("ocr_text") or "").strip()
    merged_text = f"{title}\n{desc}\n{ocr_text}".strip()

    attraction_names = split_candidate_terms(str(note.get("spots_text") or ""))
    restaurant_names = split_candidate_terms(str(note.get("foods_text") or ""))
    if merged_text:
        # 正文补抽用于补齐结构化字段缺失时的高频实体。
        # 传入城市名才会加载 data/travel_spot_rules/{城市}.json，与线上建库/抽取一致。
        attraction_names.extend(extract_spots(merged_text, title, city_name=city_name))
        restaurant_names.extend([name.strip() for name in RESTAURANT_NAME_PATTERN.findall(merged_text)])
        # 这批是人工确认过的高频北京餐厅，正文命中时优先补齐。
        for name in KNOWN_BEIJING_RESTAURANTS:
            if name in merged_text:
                restaurant_names.append(name)
    return {"attraction": attraction_names, "restaurant": restaurant_names}


def is_valid_poi_name(name: str, poi_type: str) -> bool:
    """做通用实体过滤，尽量只保留“可作为 POI 的实体名”而不是句子片段。"""
    if poi_type == "attraction":
        if name in GENERIC_WORDS["attraction"]:
            return False
        if len(name) <= 2 and name in {"大学", "胡同", "城楼"}:
            return False
        if any(noise in name for noise in ATTRACTION_NOISE_WORDS) and len(name) <= 4:
            return False
        return True

    if name in KNOWN_BEIJING_RESTAURANTS:
        return True
    # OCR/正文里常见「coffeeXX店」类噪声，不是中文店名实体。
    if re.search(r"[A-Za-z]{3,}", name):
        return False
    if name in GENERIC_WORDS["restaurant"]:
        return False
    if any(noise in name for noise in RESTAURANT_NOISE_WORDS):
        return False
    if any(token in name for token in ("和", "与", "及", "以及")):
        return False
    if len(name) > 10 and any(token in name for token in ("的", "了", "去", "也有", "可以", "就", "和")):
        return False
    # 过滤“王府井店/前门店/故宫店”这类位置店名占位词，不是可复用餐厅实体。
    if name.endswith("店") and any(name.startswith(prefix) for prefix in RESTAURANT_LOCATION_PREFIXES):
        return False
    if not any(name.endswith(suffix) for suffix in RESTAURANT_VALID_SUFFIXES):
        return False
    return True


def load_city_seed_restaurants(city_name: str) -> List[str]:
    """
    从 data/travel_spot_rules/{城市}.json 读取 seed_restaurants（人工维护的餐厅种子）。
    与 must_include_names 等景点规则同文件、不同字段，避免和景点列表混写。
    """
    city = (city_name or "").strip()
    if not city:
        return []
    rule_path = backend_root / "data" / "travel_spot_rules" / f"{city}.json"
    if not rule_path.is_file():
        return []
    try:
        payload = json.loads(rule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    raw = payload.get("seed_restaurants") or payload.get("restaurant_seed_names")
    if not isinstance(raw, list):
        return []
    names = []
    for item in raw:
        name = normalize_poi_name(str(item or ""))
        if len(name) < 2 or len(name) > 30 or name in GENERIC_WORDS["restaurant"]:
            continue
        names.append(name)
    return names


def merge_seed_restaurants_into_pool(
    pools: Dict[str, Dict[str, int]], city_name: str, min_count_restaurant: int
) -> None:
    """
    把城市规则里的餐厅种子并入计数池，保证至少达到 min_count 以便进入最终 JSON。
    种子单独保证频次，正文统计仍可用更高 min_count 抑制噪声。
    """
    min_req = max(1, int(min_count_restaurant or 1))
    # 种子名单人工维护，至少给 2 次计数，避免与正文 min_count=2 时整表被清空。
    seed_floor = max(min_req, 2)
    for name in load_city_seed_restaurants(city_name):
        current = int(pools["restaurant"].get(name) or 0)
        pools["restaurant"][name] = max(current, seed_floor)


def collect_city_pools(cache_files: List[Path]) -> Dict[str, Dict[str, int]]:
    """扫描城市缓存并构建景点/餐厅计数池。"""
    pools = {"attraction": {}, "restaurant": {}}
    city_name_set = {normalize_poi_name(file.stem) for file in cache_files}

    for cache_file in cache_files:
        # 与 data/cache/北京.json 等文件名对齐，供 extract_spots 加载同城规则。
        file_city = normalize_poi_name(cache_file.stem)
        for note in load_note_list_from_file(cache_file):
            note_candidates = collect_note_candidates(note, city_name=file_city)
            for poi_type in ("attraction", "restaurant"):
                for candidate in note_candidates[poi_type]:
                    name = normalize_poi_name(candidate)
                    if len(name) < 2 or len(name) > 40 or name in city_name_set:
                        continue
                    # 短品牌名（如「大董」）允许入库，便于与白名单/高德检索对齐。
                    if poi_type == "restaurant" and (len(name) < 2 or len(name) > 20):
                        continue
                    if not is_valid_poi_name(name, poi_type):
                        continue
                    pools[poi_type][name] = int(pools[poi_type].get(name) or 0) + 1
    return pools


def build_output_list(pool: Dict[str, int], min_count: int, top_k: int) -> List[str]:
    """把计数池转成名称数组并按频次排序截断（输出为 JSON 数组结构）。"""
    rows = [(name, count) for name, count in pool.items() if int(count) >= min_count]
    rows.sort(key=lambda item: (-int(item[1]), -len(item[0]), item[0]))
    if top_k > 0:
        rows = rows[:top_k]
    return [name for name, _ in rows]


def build_travel_poi_cache(
    cache_dir: Path,
    output_file: Path,
    city_name: str,
    min_count_attraction: int,
    min_count_restaurant: int,
    top_attractions: int,
    top_restaurants: int,
) -> None:
    """扫描本地缓存，生成离线 POI 名称列表（JSON 中为 attraction / restaurant 两个数组）。"""
    city_name = (city_name or "").strip()
    cache_files = sorted(cache_dir.glob("*.json"))
    if city_name:
        cache_files = [file for file in cache_files if file.stem == city_name]
    if not cache_files:
        raise FileNotFoundError(f"未找到缓存文件: {cache_dir} city={city_name}")

    scanned_note_count = sum(len(load_note_list_from_file(file)) for file in cache_files)
    pools = collect_city_pools(cache_files)
    merge_seed_restaurants_into_pool(pools, city_name, min_count_restaurant)
    # 与 travel_itinerary_builder 等消费方约定：景点/餐厅均为字符串数组。
    output_payload = {
        "attraction": build_output_list(
            pools["attraction"], min_count_attraction, top_attractions
        ),
        "restaurant": build_output_list(
            pools["restaurant"], min_count_restaurant, top_restaurants
        ),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    attraction_total = len(output_payload["attraction"])
    restaurant_total = len(output_payload["restaurant"])
    print("build_travel_poi_cache===========scanned_note_count:", scanned_note_count)
    print("build_travel_poi_cache===========total_poi:", attraction_total + restaurant_total)
    print("build_travel_poi_cache===========attraction_total:", attraction_total)
    print("build_travel_poi_cache===========restaurant_total:", restaurant_total)
    print("build_travel_poi_cache===========output_file:", output_file)


def main():
    """命令行入口：离线生成旅游 POI 缓存。"""
    parser = argparse.ArgumentParser(description="构建离线旅游 POI 缓存")
    parser.add_argument("--cache-dir", default="backend/data/cache", help="旅游笔记缓存目录（默认: backend/data/cache）")
    parser.add_argument("--output", default="backend/data/travel_poi_cache.json", help="POI 缓存输出文件（默认: backend/data/travel_poi_cache.json）")
    parser.add_argument("--city-name", default="北京", help="只构建指定城市的缓存（默认: 北京）")
    parser.add_argument("--min-count-attraction", type=int, default=2, help="景点最小出现频次（默认: 2）")
    # 餐厅抽取噪声普遍高于景点，默认频次设为 2 可明显提升词典准确率。
    parser.add_argument(
        "--min-count-restaurant",
        type=int,
        default=2,
        help="餐厅最小出现频次（默认: 2，抑制误匹配；种子餐厅见 travel_spot_rules 里 seed_restaurants）",
    )
    parser.add_argument("--top-attractions", type=int, default=300, help="景点最多保留多少条，0 表示不限制（默认: 300）")
    parser.add_argument("--top-restaurants", type=int, default=300, help="餐厅最多保留多少条，0 表示不限制（默认: 300）")
    args = parser.parse_args()
    build_travel_poi_cache(
        cache_dir=Path(args.cache_dir),
        output_file=Path(args.output),
        city_name=args.city_name,
        min_count_attraction=args.min_count_attraction,
        min_count_restaurant=args.min_count_restaurant,
        top_attractions=args.top_attractions,
        top_restaurants=args.top_restaurants,
    )


if __name__ == "__main__":
    main()
