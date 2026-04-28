from pathlib import Path
import json
import re

BUDGET_KEYWORDS = {
    "low": ["穷游", "学生党", "人均100", "人均200", "平价", "低预算", "省钱"],
    "medium": ["人均300", "人均500", "性价比", "预算适中", "中等预算"],
    "high": ["高端", "奢华", "五星", "米其林", "人均1000", "高预算"],
}
TRAVEL_STYLE_KEYWORDS = {
    "特种兵": ["暴走", "一天打卡", "高强度", "极限", "赶行程", "早起冲"],
    "休闲": ["慢游", "松弛", "轻松", "悠闲", "不赶路"],
    "深度": ["深度游", "人文", "博物馆", "在地", "小众", "本地人"],
}
FOOD_KEYWORDS = [
    "烤鸭",
    "火锅",
    "小吃",
    "早茶",
    "烧烤",
    "面",
    "粉",
    "甜品",
    "咖啡",
    "夜宵",
    "美食",
    "餐厅",
]
TRANSPORT_KEYWORDS = [
    "地铁",
    "公交",
    "打车",
    "高铁",
    "机场",
    "骑行",
    "步行",
    "1号线",
    "2号线",
    "3号线",
]
SPOT_PATTERN = re.compile(r"(?:去|逛|打卡|游玩|拍照)([\u4e00-\u9fa5]{2,8})")
DAY_PATTERN = re.compile(r"(?:([一二三四五六七八九十\d]+)\s*天|([一二三四五六七八九十\d]+)\s*日游)")
SPOT_NOISE_SUBSTRINGS = ("小时", "分钟", "攻略", "方案", "什么", "直接", "吃饭", "就没", "感受", "单买", "古代中国")
SPOT_NOISE_WORDS = frozenset(
    {
        "景点",
        "热门景点",
        "核心景点",
        "具体景点",
        "路线",
        "线路",
        "攻略",
        "分享",
        "建议",
        "打卡",
        "交通",
        "住宿",
        "酒店",
        "出门",
        "路",
        "沿路",
        "沿途路",
        "主街",
        "园林",
        "城",
        "中心",
        "大学",
        "胡同",
        "公园",
        "博物馆",
        "大街",
    }
)
SINGLE_CHAR_POI_SUFFIXES = frozenset(
    {
        "塔", "楼", "阁", "亭", "台", "桥", "洞", "宫", "殿", "院", "苑", "园", "寺", "庙", "庵", "观", "祠", "坛", "坊", "门",
        "廊", "街", "巷", "弄", "屯", "寨", "村", "庄", "堡", "城", "镇", "乡", "岛", "礁", "滩", "湾", "港", "池", "潭",
        "泉", "瀑", "溪", "江", "河", "湖", "海", "洋", "泊", "山", "岭", "峰", "岳", "丘", "岗", "坡", "峡", "谷", "坪",
        "峪", "关", "洲", "路",
    }
)
SHORT_POI_ALLOWLIST = frozenset({"故宫", "天坛", "北海", "后海", "前门", "鼓楼", "国博", "清华", "北大", "鸟巢", "水立方", "长城", "圆明园"})
POI_NAME_SUFFIXES = tuple(
    sorted(
        {
            "国家森林公园", "国家博物馆", "遗址公园", "博物馆", "纪念馆", "陈列馆", "美术馆", "科技馆", "文化馆", "海洋馆", "水族馆", "馆",
            "植物园", "动物园", "湿地公园", "森林公园", "主题公园", "影视城", "电影城", "游乐园", "度假区", "度假村", "旅游区", "风景区",
            "名胜区", "步行街", "商业街", "美食街", "古镇", "古城", "古村", "大峡谷", "大教堂", "清真寺", "大学", "学院", "中学", "小学",
            "道观", "滑雪场", "文化园", "生态园", "遗址", "乐园", "景区", "景点", "名胜", "寺庙", "禅寺", "教堂", "园林", "公园", "广场",
            "胡同", "峡谷", "溶洞", "瀑布", "山脉", "高原", "草原", "沙漠", "湿地", "运河", "大桥", "码头", "港口", "航站楼", "陵园", "陵墓",
            "陵寝", "塔", "楼", "阁", "亭", "台", "桥", "隧", "洞", "宫", "殿", "院", "苑", "园", "寺", "庙", "庵", "观", "祠", "坛",
            "坊", "门", "廊", "街", "巷", "弄", "屯", "寨", "村", "庄", "堡", "城", "镇", "乡", "岛", "礁", "滩", "湾", "港", "池",
            "潭", "泉", "瀑", "溪", "江", "河", "湖", "海", "洋", "泊", "山", "岭", "峰", "岳", "丘", "岗", "坡", "峡", "谷", "坪",
            "峪", "关", "洲", "中心", "大道", "路",
        },
        key=len,
        reverse=True,
    )
)
POI_EXTRA_FIXED_NAMES = frozenset({"兵马俑", "小蛮腰", "橘子洲", "什刹海", "环球影城"})
POI_LEADING_FORBIDDEN_FIRST = frozenset("买卖再去就又把给单叫让")
POI_SHORT_ABBREVS = frozenset({"北大", "清华", "复旦", "交大", "浙大", "武大", "厦大", "中大", "南大", "南开", "华科", "人大"})
POI_LINKER_PATTERN = re.compile(r"[和与及]")

travel_poi_cache_path = Path(__file__).resolve().parents[1] / "data" / "travel_poi_cache.json"
travel_spot_rule_dir = Path(__file__).resolve().parents[1] / "data" / "travel_spot_rules"
city_spot_rule_cache = {}


def normalize_text(value) -> str:
    """统一清洗文本：把 None 等值转换成去空白字符串。"""
    return str(value or "").strip()


def unique_keep_order(values: list[str]) -> list[str]:
    """按出现顺序去重，避免列表里重复关键词。"""
    seen = set()
    deduped = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def load_cached_attraction_names() -> set[str]:
    """读取景点白名单用于候选归一。"""
    if not travel_poi_cache_path.is_file():
        return set()
    try:
        with open(travel_poi_cache_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print("load_cached_attraction_names===========data \n", data, "\n")
    except (OSError, json.JSONDecodeError):
        return set()
    attraction_list = data.get("attraction") if isinstance(data, dict) else []
    if not isinstance(attraction_list, list):
        return set()
    return {str(item or "").strip() for item in attraction_list if str(item or "").strip()}


cached_attraction_names = None


def get_cached_attraction_names() -> set[str]:
    """懒加载景点白名单，避免模块导入时触发 IO。"""
    global cached_attraction_names
    # 仅在真正需要做景点归一时加载一次缓存，减少保存触发的日志噪音。
    if cached_attraction_names is None:
        cached_attraction_names = load_cached_attraction_names()
    return cached_attraction_names


def load_city_spot_rule(city_name: str) -> dict:
    """读取城市级景点规则。"""
    city = str(city_name or "").strip()
    if not city:
        return {}
    cached_rule = city_spot_rule_cache.get(city)
    if isinstance(cached_rule, dict):
        return cached_rule
    rule_path = travel_spot_rule_dir / f"{city}.json"
    if not rule_path.is_file():
        city_spot_rule_cache[city] = {}
        return {}
    try:
        with open(rule_path, "r", encoding="utf-8") as file:
            raw_rule = json.load(file)
    except (OSError, json.JSONDecodeError):
        city_spot_rule_cache[city] = {}
        return {}
    rule = raw_rule if isinstance(raw_rule, dict) else {}
    city_spot_rule_cache[city] = rule
    return rule


def get_spot_rule_values(city_name: str) -> dict:
    """合并默认规则和城市规则。"""
    city_rule = load_city_spot_rule(city_name)
    merged = {
        "noise_words": set(SPOT_NOISE_WORDS),
        "noise_substrings": set(SPOT_NOISE_SUBSTRINGS),
        "single_char_suffixes": set(SINGLE_CHAR_POI_SUFFIXES),
        "allowlist_names": set(SHORT_POI_ALLOWLIST),
        "must_include_names": set(POI_EXTRA_FIXED_NAMES),
        "short_abbrevs": set(POI_SHORT_ABBREVS),
        "leading_forbidden_first": set(POI_LEADING_FORBIDDEN_FIRST),
        "single_char_min_len": 4,
    }
    for key in ("noise_words", "noise_substrings", "single_char_suffixes", "allowlist_names", "must_include_names", "short_abbrevs"):
        values = city_rule.get(key)
        if isinstance(values, list):
            merged[key].update(str(item).strip() for item in values if str(item).strip())
    merged["allowlist_names"].update(merged["must_include_names"])
    custom_forbidden_first = city_rule.get("leading_forbidden_first")
    if isinstance(custom_forbidden_first, str) and custom_forbidden_first:
        merged["leading_forbidden_first"] = set(custom_forbidden_first)
    min_len = city_rule.get("single_char_min_len")
    if isinstance(min_len, int) and min_len >= 2:
        merged["single_char_min_len"] = min_len
    return merged


default_spot_rule = get_spot_rule_values("")


def normalize_poi_token(value: str) -> str:
    """统一 POI 片段格式，便于白名单匹配。"""
    text = re.sub(r"[（(][^）)]{1,24}[）)]", "", normalize_text(value))
    text = re.sub(r"[\s\-_/|·•]+", "", text)
    return text.strip()


def match_cached_attraction_name(token: str, cache_names: set[str]) -> str:
    """把候选名映射到白名单里的规范名称。"""
    name = normalize_text(token)
    if not name or not cache_names:
        return ""
    if name in cache_names:
        return name
    normalized = normalize_poi_token(name)
    if not normalized:
        return ""
    for cached_name in cache_names:
        if normalize_poi_token(cached_name) == normalized:
            return cached_name
    contained = [cached_name for cached_name in cache_names if len(cached_name) >= 2 and cached_name in name]
    return max(contained, key=len) if contained else ""


def is_plausible_poi_name(name: str, spot_rule: dict | None = None) -> bool:
    """判断字符串是否像景点名。"""
    rule = spot_rule or default_spot_rule
    name = normalize_text(name)
    if not name or len(name) < 2 or name in rule["noise_words"] or "的" in name:
        return False
    if any(fragment in name for fragment in rule["noise_substrings"]) or any(ch.isdigit() for ch in name):
        return False
    if len(name) > 10 or name[0] in rule["leading_forbidden_first"] or any(link in name for link in ("和", "与", "及")):
        return False
    if name in rule["must_include_names"] or name in rule["short_abbrevs"] or name in rule["allowlist_names"]:
        return True
    matched_suffix = next((suffix for suffix in POI_NAME_SUFFIXES if name.endswith(suffix)), "")
    if not matched_suffix:
        return False
    if matched_suffix in rule["single_char_suffixes"] and len(name) < rule["single_char_min_len"]:
        return False
    return True


def scan_text_for_poi_like_names(text: str, spot_rule: dict | None = None) -> list[str]:
    """从全文滑窗扫描补全景点候选。"""
    rule = spot_rule or default_spot_rule
    results = []
    for chunk in re.findall(r"[\u4e00-\u9fa5]+", text):
        for segment in POI_LINKER_PATTERN.split(chunk):
            if not segment:
                continue
            index = 0
            while index < len(segment):
                picked = None
                max_size = min(8, len(segment) - index)
                for size in range(max_size, 1, -1):
                    window = segment[index : index + size]
                    if is_plausible_poi_name(window, rule):
                        picked = window
                        break
                if picked:
                    results.append(picked)
                    index += len(picked)
                else:
                    index += 1
    return results


def refine_spots_with_cache(candidates: list[str]) -> list[str]:
    """用白名单做最终归一。"""
    cache_names = get_cached_attraction_names()
    if not cache_names:
        return candidates
    refined = []
    seen = set()
    for item in candidates:
        matched = match_cached_attraction_name(item, cache_names)
        if not matched or matched in seen:
            continue
        seen.add(matched)
        refined.append(matched)
    return refined


def extract_keywords_by_dictionary(text: str, keywords: list[str], limit: int) -> list[str]:
    """按词典提取关键词。"""
    hits = [keyword for keyword in keywords if keyword in text]
    return unique_keep_order(hits)[:limit]


def extract_spots(text: str, title: str, city_name: str = "") -> list[str]:
    """从标题与正文中抽取景点候选词。"""
    spot_rule = get_spot_rule_values(city_name)
    combined = f"{title}\n{text}"
    candidates = SPOT_PATTERN.findall(combined)
    candidates.extend(scan_text_for_poi_like_names(combined, spot_rule))
    for token in sorted(spot_rule["must_include_names"], key=len, reverse=True):
        if token in text or token in title:
            candidates.append(token)
    candidates = [item for item in candidates if is_plausible_poi_name(item, spot_rule)]
    unique_candidates = unique_keep_order(candidates)
    fixed_hits = [token for token in sorted(spot_rule["must_include_names"], key=len, reverse=True) if token in combined]
    prioritized = []
    seen = set()
    for item in fixed_hits + unique_candidates:
        if item in seen:
            continue
        seen.add(item)
        prioritized.append(item)
    refined = refine_spots_with_cache(prioritized)
    return (refined or prioritized)[:12]


def extract_duration(text: str) -> str:
    """从融合文本中提取游玩时长。"""
    match = DAY_PATTERN.search(text)
    if not match:
        return ""
    token = normalize_text(match.group(1) or match.group(2))
    return f"{token}天" if token else ""


def extract_budget_level(text: str) -> str:
    """根据预算关键词推断预算档位。"""
    for budget_level, keywords in BUDGET_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return budget_level
    return "medium"


def extract_travel_style(text: str) -> str:
    """根据描述关键词推断旅行风格。"""
    for travel_style, keywords in TRAVEL_STYLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return travel_style
    return "休闲"


def extract_itinerary_text(text: str, city_name: str = "") -> list[dict]:
    """提取按天信息，生成简化 itinerary。"""
    itinerary = []
    day_matches = list(re.finditer(r"第\s*([一二三四五六七八九十\d]+)\s*天[:：]?", text))
    if not day_matches:
        return itinerary
    for index, day_match in enumerate(day_matches):
        start = day_match.end()
        end = day_matches[index + 1].start() if index + 1 < len(day_matches) else len(text)
        day_block = text[start:end]
        itinerary.append({"day": day_match.group(1), "activities": extract_spots(day_block, "", city_name=city_name)[:6]})
    return itinerary[:7]


def extract_tags(text: str) -> list[str]:
    """根据内容语义打标签。"""
    tag_keyword_map = {
        "拍照": ["拍照", "出片", "机位"],
        "历史": ["历史", "博物馆", "古迹"],
        "美食": ["美食", "餐厅", "小吃"],
        "夜景": ["夜景", "灯光"],
        "亲子": ["亲子"],
        "徒步": ["徒步", "登山"],
    }
    tags = []
    for tag_name, keywords in tag_keyword_map.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag_name)
    return unique_keep_order(tags)[:8]


def build_structured_profile(city_name: str, title: str, desc: str, ocr_text: str) -> dict:
    """融合正文与 OCR，输出结构化旅游画像。"""
    merged_text = f"{title}\n{desc}\n{ocr_text}".strip()

    return {
        "city": city_name,
        "spots": extract_spots(merged_text, title, city_name=city_name),
        "foods": extract_keywords_by_dictionary(merged_text, FOOD_KEYWORDS, limit=12),
        "transport": extract_keywords_by_dictionary(merged_text, TRANSPORT_KEYWORDS, limit=12),
        "duration": extract_duration(merged_text),
        "budget_level": extract_budget_level(merged_text),
        "tags": extract_tags(merged_text),
        "travel_style": extract_travel_style(merged_text),
        "itinerary": extract_itinerary_text(merged_text, city_name=city_name),
        "raw_summary": merged_text[:220],
    }
