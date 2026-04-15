from pathlib import Path
import json
import re

from langchain_core.documents import Document

MAX_IMAGES_PER_NOTE_FOR_OCR = 8
MAX_OCR_CHARS_FOR_EMBED = 1200
ocr_engine = None
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
    "烤鸭", "火锅", "小吃", "早茶", "烧烤", "面", "粉", "甜品", "咖啡", "夜宵", "美食", "餐厅"
]
TRANSPORT_KEYWORDS = [
    "地铁", "公交", "打车", "高铁", "机场", "骑行", "步行", "1号线", "2号线", "3号线"
]
# 动词后只抓连续汉字，避免把「游玩1小时」里的时长、方案编号等吃进景点字段。
SPOT_PATTERN = re.compile(r"(?:去|逛|打卡|游玩|拍照)([\u4e00-\u9fa5]{2,8})")
DAY_PATTERN = re.compile(r"(?:([一二三四五六七八九十\d]+)\s*天|([一二三四五六七八九十\d]+)\s*日游)")
# 游记口语里跟在动词后面常被误抽成「景点」的片段：时间、方案、动作描述等。
SPOT_NOISE_SUBSTRINGS = (
    "小时",
    "分钟",
    "攻略",
    "方案",
    "什么",
    "直接",
    "吃饭",
    "就没",
    "感受",
    "单买",
    "古代中国",
)
# 通用汉语地名/景点名后缀（语言学规律，全国适用）；用 set 去重后按长度降序，匹配时先试长后缀。
POI_NAME_SUFFIXES = tuple(
    sorted(
        {
            "国家森林公园",
            "国家博物馆",
            "遗址公园",
            "博物馆",
            "纪念馆",
            "陈列馆",
            "美术馆",
            "科技馆",
            "文化馆",
            "海洋馆",
            "水族馆",
            "馆",
            "植物园",
            "动物园",
            "湿地公园",
            "森林公园",
            "主题公园",
            "影视城",
            "电影城",
            "游乐园",
            "度假区",
            "度假村",
            "旅游区",
            "风景区",
            "名胜区",
            "步行街",
            "商业街",
            "美食街",
            "古镇",
            "古城",
            "古村",
            "大峡谷",
            "大教堂",
            "清真寺",
            "大学",
            "学院",
            "中学",
            "小学",
            "道观",
            "滑雪场",
            "文化园",
            "生态园",
            "遗址",
            "乐园",
            "景区",
            "景点",
            "名胜",
            "寺庙",
            "禅寺",
            "教堂",
            "园林",
            "公园",
            "广场",
            "胡同",
            "峡谷",
            "溶洞",
            "瀑布",
            "山脉",
            "高原",
            "草原",
            "沙漠",
            "湿地",
            "运河",
            "大桥",
            "码头",
            "港口",
            "航站楼",
            "陵园",
            "陵墓",
            "陵寝",
            "塔",
            "楼",
            "阁",
            "亭",
            "台",
            "桥",
            "隧",
            "洞",
            "宫",
            "殿",
            "院",
            "苑",
            "园",
            "寺",
            "庙",
            "庵",
            "观",
            "祠",
            "坛",
            "坊",
            "门",
            "廊",
            "街",
            "巷",
            "弄",
            "屯",
            "寨",
            "村",
            "庄",
            "堡",
            "城",
            "镇",
            "乡",
            "岛",
            "礁",
            "滩",
            "湾",
            "港",
            "池",
            "潭",
            "泉",
            "瀑",
            "溪",
            "江",
            "河",
            "湖",
            "海",
            "洋",
            "泊",
            "山",
            "岭",
            "峰",
            "岳",
            "丘",
            "岗",
            "坡",
            "峡",
            "谷",
            "坪",
            "峪",
            "关",
            "洲",
            "中心",
            "大道",
            "路",
        },
        key=len,
        reverse=True,
    )
)
# 少数无法靠后缀识别的全国级地标昵称/专名（体量固定，不是按城市穷举）。
POI_EXTRA_FIXED_NAMES = frozenset(
    {
        "兵马俑",
        "小蛮腰",
        "橘子洲",
        "什刹海",
        "环球影城",
    }
)
# 景点名片段若以这些字开头，多为「买/去/单」等口语动词链上的误切，应排除。
POI_LEADING_FORBIDDEN_FIRST = frozenset("买卖再去就又把给单叫让")
# 常见高校/机构二字简称（用于无「园山湖」等后缀时的弱补充，仍配合否定规则使用）。
POI_SHORT_ABBREVS = frozenset(
    {
        "北大",
        "清华",
        "复旦",
        "交大",
        "浙大",
        "武大",
        "厦大",
        "中大",
        "南大",
        "南开",
        "华科",
        "人大",
    }
)
# 并列连接词：游经常写「A和B」，整段滑窗会把「北大和圆明园」误收成一条，需在扫描时拆段。
POI_LINKER_PATTERN = re.compile(r"[和与及]")


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


def is_plausible_poi_name(name: str) -> bool:
    """
    判断字符串是否像景点/地名：优先用语义后缀（山湖园宫等），辅以少量全国级专名与高校简称；
    不维护「按城市枚举」的 POI 库，避免数据量随城市爆炸。
    """
    name = normalize_text(name)
    if not name or len(name) < 2:
        return False
    if any(fragment in name for fragment in SPOT_NOISE_SUBSTRINGS):
        return False
    if any(ch.isdigit() for ch in name):
        return False
    # 正则最多抽 8 字；全文扫到的专名略放宽，防止半句被误收。
    if len(name) > 10:
        return False
    if name[0] in POI_LEADING_FORBIDDEN_FIRST:
        return False
    if any(link in name for link in ("和", "与", "及")):
        return False
    if name in POI_EXTRA_FIXED_NAMES or name in POI_SHORT_ABBREVS:
        return True
    return any(name.endswith(suffix) for suffix in POI_NAME_SUFFIXES)


def filter_spot_candidates(raw: list[str]) -> list[str]:
    """对正则抽到的片段做二次筛选，只保留像景点名的条目。"""
    return [item for item in raw if is_plausible_poi_name(item)]


def scan_text_for_poi_like_names(text: str) -> list[str]:
    """
    在纯中文连续片段上从左贪心取最长「像景点名」的子串，补全动词正则漏掉的正文内地名。
    遇到「和/与/及」先拆段，避免「北大和圆明园」被整段收成一条；规则与 is_plausible_poi_name 一致。
    """
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
                    if is_plausible_poi_name(window):
                        picked = window
                        break
                if picked:
                    results.append(picked)
                    index += len(picked)
                else:
                    index += 1
    return results


def extract_duration(text: str) -> str:
    """从融合文本中提取游玩时长，未命中时返回空串。"""
    match = DAY_PATTERN.search(text)
    if not match:
        return ""
    token = normalize_text(match.group(1) or match.group(2))
    if not token:
        return ""
    return f"{token}天"


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


def extract_spots(text: str, title: str) -> list[str]:
    """从标题与正文中抽取景点候选词。"""
    candidates = []
    combined = f"{title}\n{text}"
    for match in SPOT_PATTERN.findall(combined):
        candidates.append(match)
    candidates.extend(scan_text_for_poi_like_names(combined))
    # 少数昵称型全国地标无法单靠后缀命中，用小集合在正文里补扫（非按城市穷举）。
    for token in sorted(POI_EXTRA_FIXED_NAMES, key=len, reverse=True):
        if token in text or token in title:
            candidates.append(token)
    candidates = filter_spot_candidates(candidates)
    return unique_keep_order(candidates)[:12]


def extract_foods(text: str) -> list[str]:
    """从融合文本中抽取美食与餐饮关键词。"""
    foods = []
    for keyword in FOOD_KEYWORDS:
        if keyword in text:
            foods.append(keyword)
    return unique_keep_order(foods)[:12]


def extract_transport(text: str) -> list[str]:
    """从融合文本中抽取交通方式关键词。"""
    transports = []
    for keyword in TRANSPORT_KEYWORDS:
        if keyword in text:
            transports.append(keyword)
    return unique_keep_order(transports)[:12]


def extract_tags(text: str) -> list[str]:
    """根据内容语义打标签，便于后续按意图检索。"""
    # 使用标签-关键词映射统一处理，减少重复 if 分支。
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


def extract_itinerary_text(text: str) -> list[dict]:
    """提取按天信息，生成简化 itinerary 结构。"""
    itinerary = []
    day_matches = list(re.finditer(r"第\s*([一二三四五六七八九十\d]+)\s*天[:：]?", text))
    if not day_matches:
        return itinerary
    for index, day_match in enumerate(day_matches):
        start = day_match.end()
        end = day_matches[index + 1].start() if index + 1 < len(day_matches) else len(text)
        day_block = text[start:end]
        activities = extract_spots(day_block, "")
        itinerary.append(
            {
                "day": day_match.group(1),
                "activities": activities[:6],
            }
        )
    return itinerary[:7]


def build_structured_profile(city_name: str, title: str, desc: str, ocr_text: str) -> dict:
    """把 desc + OCR 融合后抽取结构化字段，支持后续按城市和意图检索。"""
    merged_text = f"{title}\n{desc}\n{ocr_text}".strip()
    spots = extract_spots(merged_text, title)
    foods = extract_foods(merged_text)
    transports = extract_transport(merged_text)
    tags = extract_tags(merged_text)
    itinerary = extract_itinerary_text(merged_text)
    raw_summary = merged_text[:220]
    return {
        "city": city_name,
        "spots": spots,
        "foods": foods,
        "transport": transports,
        "duration": extract_duration(merged_text),
        "budget_level": extract_budget_level(merged_text),
        "tags": tags,
        "travel_style": extract_travel_style(merged_text),
        "itinerary": itinerary,
        "raw_summary": raw_summary,
    }


def load_travel_cache_docs(data_path="data", city_name=None, allow_runtime_ocr=False):
    """
    从 data/cache 下读取旅游笔记缓存，按 note_id 去重并生成 Document（整篇不切分）。
    city_name 传值时只读取对应城市文件（如 北京 -> 北京.json），用于缩小检索域。
    allow_runtime_ocr 控制是否在查询阶段补跑 OCR：
    - False：只复用已有 OCR 缓存，不做实时识别（默认，保证查询低时延）
    - True：缓存缺失时允许识别图片并回写缓存（适合离线预处理）
    """
    cache_dir = Path(data_path) / "cache"
    if not cache_dir.is_dir():
        return []

    target_files = sorted(cache_dir.glob("*.json"))
    if city_name:
        city_file = cache_dir / f"{city_name}.json"
        if not city_file.is_file():
            return []
        target_files = [city_file]

    merged = {}
    for path in target_files:
        try:
            with open(path, "r", encoding="utf-8") as file:
                records = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, dict):
                continue
            note_id = str(item.get("note_id", "")).strip()
            if not note_id:
                continue
            # 记录来源城市：优先使用文件名（如 北京.json），便于后续城市检索提权。
            item_city = str(path.stem).strip()
            if item_city:
                item["city"] = item_city
            prev = merged.get(note_id)
            if prev is None:
                merged[note_id] = item
                continue
            has_images = bool(item.get("feed_images"))
            prev_has_images = bool(prev.get("feed_images"))
            if has_images and not prev_has_images:
                merged[note_id] = item

    ocr_cache_path = Path(data_path) / "cache_ocr_text.json"
    ocr_cache = load_ocr_cache(ocr_cache_path)
    cache_updated = False

    docs = []
    backend_root = Path(data_path).resolve().parent
    for note in merged.values():
        title = normalize_text(note.get("title"))
        desc = normalize_text(note.get("desc"))
        ocr_text, changed = get_or_build_note_ocr_text(
            note,
            backend_root,
            ocr_cache,
            allow_runtime_ocr=allow_runtime_ocr,
        )
        if changed:
            cache_updated = True
        ocr_text_for_embed = normalize_text(ocr_text)[:MAX_OCR_CHARS_FOR_EMBED]
        inferred_city = normalize_text(note.get("city") or city_name)
        structured_profile = build_structured_profile(
            city_name=inferred_city,
            title=title,
            desc=desc,
            ocr_text=ocr_text,
        )
        profile_text_for_embed = (
            f"城市:{structured_profile['city']}\n"
            f"景点:{' '.join(structured_profile['spots'])}\n"
            f"美食:{' '.join(structured_profile['foods'])}\n"
            f"交通:{' '.join(structured_profile['transport'])}\n"
            f"标签:{' '.join(structured_profile['tags'])}\n"
            f"旅行风格:{structured_profile['travel_style']}\n"
            f"预算:{structured_profile['budget_level']}\n"
            f"时长:{structured_profile['duration']}"
        ).strip()
        print("profile_text_for_embed===========结果不准确 \n", profile_text_for_embed, "\n")
        page_content = f"{title}\n{desc}\n{ocr_text_for_embed}\n{profile_text_for_embed}".strip()
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source_type": "travel_cache",
                    "note_id": note.get("note_id"),
                    "note_url": note.get("note_url"),
                    "title": title,
                    "desc": desc,
                    "ocr_text": ocr_text,
                    "city": structured_profile["city"],
                    "spots_text": " ".join(structured_profile["spots"]),
                    "foods_text": " ".join(structured_profile["foods"]),
                    "transport_text": " ".join(structured_profile["transport"]),
                    "duration": structured_profile["duration"],
                    "budget_level": structured_profile["budget_level"],
                    "tags_text": " ".join(structured_profile["tags"]),
                    "travel_style": structured_profile["travel_style"],
                    "raw_summary": structured_profile["raw_summary"],
                    "itinerary_json": json.dumps(structured_profile["itinerary"], ensure_ascii=False),
                    "structured_profile_json": json.dumps(structured_profile, ensure_ascii=False),
                    # Milvus metadata 字段不支持 list/dict，这里序列化成 JSON 字符串，使用侧再反序列化。
                    "feed_images_json": json.dumps(note.get("feed_images") or [], ensure_ascii=False),
                },
            )
        )
    if cache_updated:
        save_ocr_cache(ocr_cache_path, ocr_cache)
    return docs


def load_ocr_cache(cache_file: Path):
    """读取 OCR 文本缓存，避免重复识别相同图片。"""
    if not cache_file.is_file():
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def save_ocr_cache(cache_file: Path, cache_data: dict):
    """持久化 OCR 缓存到 data 目录，重启后仍可复用。"""
    try:
        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(cache_data, file, ensure_ascii=False)
    except OSError:
        return


def resolve_local_image_paths(note: dict, backend_root: Path):
    """解析笔记图片本地路径，优先 local_path，否则按 note_id 目录扫描。"""
    paths = []
    seen = set()
    for image_info in note.get("feed_images") or []:
        if not isinstance(image_info, dict):
            continue
        local_path = image_info.get("local_path")
        if not local_path:
            continue
        image_path = (backend_root / local_path).resolve()
        if image_path.is_file() and str(image_path) not in seen:
            seen.add(str(image_path))
            paths.append(image_path)
    if paths:
        return paths[:MAX_IMAGES_PER_NOTE_FOR_OCR]

    note_id = str(note.get("note_id") or "").strip()
    if not note_id:
        return []
    image_folder = backend_root / "data" / "note_images" / note_id
    if not image_folder.is_dir():
        return []
    for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png"):
        for image_path in sorted(image_folder.glob(ext)):
            if str(image_path) in seen:
                continue
            seen.add(str(image_path))
            paths.append(image_path)
            if len(paths) >= MAX_IMAGES_PER_NOTE_FOR_OCR:
                return paths
    return paths


def get_ocr_engine():
    """
    懒加载 OCR 引擎，只有首次需要 OCR 时才初始化。
    未安装或 Python 版本无可用包时返回 None，避免整条加载链路因 ImportError 中断。
    """
    global ocr_engine
    if ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            ocr_engine = RapidOCR()
        except ImportError:
            ocr_engine = False
    if ocr_engine is False:
        return None
    return ocr_engine


def ocr_image_text(image_path: Path) -> str:
    """识别单张图片文字，失败返回空字符串。"""
    engine = get_ocr_engine()
    if engine is None:
        return ""
    try:
        result, _elapsed = engine(str(image_path))
    except Exception:
        return ""
    if not result:
        return ""
    lines = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
            line = item[1].strip()
            if line:
                lines.append(line)
    return "\n".join(lines).strip()


def build_image_signature(image_paths):
    """根据图片路径、修改时间和大小构建签名，判断缓存是否失效。"""
    signature = []
    for image_path in image_paths:
        stat = image_path.stat()
        signature.append(f"{str(image_path)}|{stat.st_mtime}|{stat.st_size}")
    return signature


def get_or_build_note_ocr_text(
    note: dict,
    backend_root: Path,
    ocr_cache: dict,
    allow_runtime_ocr: bool = False,
):
    """
    获取单条笔记 OCR 文本：签名一致且已有非空结果时直接返回；
    签名一致但 ocr_text 为空时不视为有效缓存（此前可能识别失败或未跑过），会重新识别；
    若重试后仍为空则写入 empty_verified，避免每条笔记每次建库都重复全量 OCR。
    allow_runtime_ocr=False 时只读缓存，不触发实时 OCR，确保在线查询链路稳定低延迟。
    """
    note_id = normalize_text(note.get("note_id"))
    if not note_id:
        return "", False
    image_paths = resolve_local_image_paths(note, backend_root)
    if not image_paths:
        return "", False
    signature = build_image_signature(image_paths)
    cached_item = ocr_cache.get(note_id)
    if isinstance(cached_item, dict) and cached_item.get("signature") == signature:
        cached_ocr = str(cached_item.get("ocr_text") or "").strip()
        if cached_ocr:
            return cached_ocr, False
        # 已确认「确实无字」的笔记不再反复识别，防止建库阶段被空串拖死。
        if cached_item.get("empty_verified"):
            return "", False
        if not allow_runtime_ocr:
            return "", False

    # 在线查询默认不补跑 OCR：缺缓存就直接空串返回，避免单次检索触发上百张图识别。
    if not allow_runtime_ocr:
        return "", False

    ocr_texts = []
    for image_path in image_paths:
        text = ocr_image_text(image_path)
        if text:
            ocr_texts.append(text)
    merged_ocr_text = "\n\n".join(ocr_texts).strip()
    has_text = bool(merged_ocr_text)
    ocr_cache[note_id] = {
        "signature": signature,
        "ocr_text": merged_ocr_text,
        "empty_verified": not has_text,
    }
    return merged_ocr_text, True
