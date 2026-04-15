"""
旅游攻略工具：调用 rag 层旅游缓存检索 + 本地配图 OCR，组装返回给模型。
检索逻辑见 rag/travel_cache_retriever.py。
"""
import json
import re
from pathlib import Path
from typing import List

from langchain.tools import tool

from rag.travel_cache_retriever import is_food_focus_query, retrieve_travel_docs

# backend 根目录（含 data/note_images）
BACKEND_ROOT = Path(__file__).resolve().parent.parent
NOTE_IMAGES_ROOT = BACKEND_ROOT / "data" / "note_images"

TOP_K_NOTES = 4
# 美食类问题需要覆盖同城多条笔记；具体上限仍受 retrieve_travel_docs 与数据量约束。
TOP_K_FOOD = 20
MAX_IMAGES_PER_NOTE = 8

# 查询阶段不对配图做实时 OCR：否则「美食检索约 20 条 × 每条约 8 张图」会触发上百次 RapidOCR，单次搜索可达数分钟。
# 配图文字应来自建库阶段 travel_loader 写入 metadata 的 ocr_text；若为空则仅用语义正文 desc。
MAX_RUNTIME_OCR_IMAGES_IN_TOOL = 0

# 用于从正文/OCR 里截取「和吃喝更相关」的一小段，供美食类紧凑摘要使用（非穷举，覆盖常见口语）。
FOOD_SNIPPET_HINTS = (
    "美食",
    "小吃",
    "餐厅",
    "饭馆",
    "火锅",
    "烤鸭",
    "奶茶",
    "咖啡",
    "早茶",
    "早餐",
    "午餐",
    "晚餐",
    "夜宵",
    "必吃",
    "好吃",
    "铜锅",
    "涮肉",
    "豆汁",
    "米其林",
)

ocr_engine = None


def pick_food_related_snippet(desc: str, ocr_text: str, max_len: int = 220) -> str:
    """
    从正文 desc 与配图 OCR 中优先截取包含餐饮线索的片段；若无命中则退回正文前若干字。
    """
    blob = (desc or "").strip().replace("\n", " ")
    for hint in FOOD_SNIPPET_HINTS:
        if hint in blob:
            idx = blob.index(hint)
            start = max(0, idx - 36)
            end = min(len(blob), start + max_len)
            piece = blob[start:end]
            return piece + ("..." if end < len(blob) else "")
    ocr = (ocr_text or "").strip().replace("\n", " ")
    for hint in FOOD_SNIPPET_HINTS:
        if hint in ocr:
            idx = ocr.index(hint)
            start = max(0, idx - 24)
            end = min(len(ocr), start + min(max_len, 180))
            piece = ocr[start:end]
            return piece + ("..." if end < len(ocr) else "")
    if blob:
        return blob[:max_len] + ("..." if len(blob) > max_len else "")
    if ocr:
        return ocr[:max_len] + ("..." if len(ocr) > max_len else "")
    return "（暂无正文与配图文字摘要）"


def resolve_local_image_paths(note: dict) -> List[Path]:
    """
    解析笔记对应的本地配图路径：优先 feed_images[].local_path，否则扫描 note_images/<note_id>/。
    """
    paths: List[Path] = []
    seen = set()
    for img in note.get("feed_images") or []:
        if not isinstance(img, dict):
            continue
        lp = img.get("local_path")
        if not lp:
            continue
        p = (BACKEND_ROOT / lp).resolve()
        if p.is_file() and str(p) not in seen:
            seen.add(str(p))
            paths.append(p)
    if paths:
        return paths[:MAX_IMAGES_PER_NOTE]

    note_id = note.get("note_id")
    if not note_id:
        return []
    folder = NOTE_IMAGES_ROOT / str(note_id)
    if not folder.is_dir():
        return []
    for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png"):
        for p in sorted(folder.glob(ext)):
            if str(p) not in seen:
                seen.add(str(p))
                paths.append(p)
            if len(paths) >= MAX_IMAGES_PER_NOTE:
                return paths
    return paths


def get_ocr_engine():
    """
    懒加载 RapidOCR，避免 import 工具时就初始化模型。
    未安装或当前 Python 版本无可用 wheel（如 3.14）时返回 None，调用方按“无 OCR”降级。
    """
    global ocr_engine
    if ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            ocr_engine = RapidOCR()
        except ImportError:
            # PyPI 上 rapidocr-onnxruntime 目前最高约支持到 3.12，3.14 等版本会装不上。
            ocr_engine = False
    if ocr_engine is False:
        return None
    return ocr_engine


def ocr_image_file(image_path: Path) -> str:
    """
    对单张本地图片做中文 OCR，返回识别文本；失败时返回空串（单张失败不影响其他图）。
    """
    engine = get_ocr_engine()
    if engine is None:
        return ""
    try:
        result, _elapsed = engine(str(image_path))
    except Exception:
        return ""
    if not result:
        return ""
    lines: List[str] = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
            lines.append(item[1].strip())
    return "\n".join(lines).strip()


def build_note_block(note: dict) -> str:
    """
    组装单条笔记输出：标题、链接、正文 desc、配图 OCR 汇总。
    """
    title = (note.get("title") or "").strip() or "(无标题)"
    # url = (note.get("note_url") or "").strip()
    desc = (note.get("desc") or "").strip()

    parts = [f"【标题】{title}"]
    # if url:
    #     parts.append(f"【链接】{url}")

    prebuilt_ocr_text = (note.get("ocr_text") or "").strip()
    ocr_chunks: List[str] = []
    if prebuilt_ocr_text:
        # 命中建库阶段的 OCR 缓存时，直接复用，避免查询阶段重复 OCR。
        ocr_chunks.append(prebuilt_ocr_text)
    elif MAX_RUNTIME_OCR_IMAGES_IN_TOOL > 0:
        # 仅当允许实时 OCR 且建库未写入 ocr_text 时，才按张识别（默认关闭以保证响应时间）。
        img_paths = resolve_local_image_paths(note)[:MAX_RUNTIME_OCR_IMAGES_IN_TOOL]
        for idx, p in enumerate(img_paths):
            text = ocr_image_file(p)
            if text:
                ocr_chunks.append(f"— 图{idx + 1} ({p.name}) —\n{text}")
    # 按用户要求融合素材：
    # - OCR 成功：desc + OCR 合并，给模型更多可用事实
    # - OCR 失败：仅使用 desc，避免空 OCR 噪声影响路线生成
    merged_content_parts: List[str] = []
    if desc:
        merged_content_parts.append(f"【正文 desc】\n{desc}")
    if ocr_chunks:
        merged_content_parts.append("【配图 OCR 文字】\n" + "\n\n".join(ocr_chunks))
    if not merged_content_parts:
        merged_content_parts.append("【正文素材】（该条暂无可用正文与 OCR 文字）")
    parts.append("\n\n".join(merged_content_parts))

    return "\n\n".join(parts)


def build_compact_material_summary(docs, query: str = "") -> str:
    """
    把检索文档压成紧凑摘要，确保模型在长文本被截断时仍能拿到核心事实。
    美食类问题时优先展示结构化餐饮字段（foods_text、标签）与正文/OCR 中与吃喝相关的片段，避免只显示标题+泛化攻略前几句。
    """
    food_focus = is_food_focus_query((query or "").strip())
    lines: List[str] = []
    for rank, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        title = (metadata.get("title") or "").strip() or "(无标题)"
        desc = (metadata.get("desc") or "").strip().replace("\n", " ")
        note_id = str(metadata.get("note_id") or "")
        if not food_focus:
            short_desc = desc[:180] + ("..." if len(desc) > 180 else "")
            lines.append(f"- 结果{rank} | note_id={note_id} | 标题={title} | 摘要={short_desc}")
            continue
        foods_text = (metadata.get("foods_text") or "").strip()
        tags_text = (metadata.get("tags_text") or "").strip()
        ocr_text = (metadata.get("ocr_text") or "").strip()
        foods_short = (
            foods_text[:100] + ("..." if len(foods_text) > 100 else "")
            if foods_text
            else "（建库未抽到餐饮关键词，请看下方正文/配图摘要）"
        )
        tags_short = tags_text[:72] + ("..." if len(tags_text) > 72 else "") if tags_text else ""
        snippet = pick_food_related_snippet(desc, ocr_text, max_len=220)
        extra_tags = f" | 标签={tags_short}" if tags_short else ""
        lines.append(
            f"- 结果{rank} | note_id={note_id} | 标题={title} | 餐饮关键词={foods_short}{extra_tags}"
            f" | 正文或配图摘要={snippet}"
        )
    return "\n".join(lines)


def parse_travel_days(query: str):
    """
    从用户问题里解析旅游天数：
    - 命中“X日游/玩X天”时返回对应天数
    - 未命中时返回 None，交给模型结合城市规模和检索材料判断建议天数
    """
    q = (query or "").strip()
    if not q:
        return None

    day_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    match = re.search(r"([一二三四五六七八九十\d]+)\s*(?:日游|天)", q)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return day_map.get(token)


def build_itinerary_format_instruction(query: str) -> str:
    """
    生成旅游攻略输出格式约束，要求回答按天给出可执行路线与餐饮建议。
    """
    # 仅当用户明确在问吃喝/美食时使用美食模板；不要用「query 里没出现旅游」误伤纯行程问题。
    if is_food_focus_query(query):
        # 用户明确要搜索美食时，直接返回美食输出要求
        return (
            "【美食输出要求（必须遵守）】\n"
            "你现在是“美食推荐博主”，请严格执行以下规则，不要输出与规则无关的泛化攻略。\n\n"
            "请先根据目的地与下方参考材料整理当地美食。\n"
            "把菜品分类，例如当地特色菜、京菜、川菜、粤菜等（按材料里实际出现的内容选，不要空泛罗列菜系名）。\n"
            "然后根据口味、价格区间、地理位置、口碑等把材料里出现的美食尽量列全，并做推荐度排序（⭐️ 越多越推荐）。\n"
            "若有具体餐厅名请写出餐厅名；若无则给可执行的美食建议。\n"
            "若有具体菜名请写出菜名；若无则给品类或方向建议。\n"
        )
    # 用户未指定天数时，先让模型给出推荐总天数，再按天展开。
    return (
            "你现在是“旅游路线编排助手”，请严格执行以下规则，不要输出与规则无关的泛化攻略。\n\n"
            "【行程输出要求（必须遵守）】\n"
            "请先根据目的地规模、景点密度和检索材料，给出“游玩天数”（例如 2-3 天 / 4-5 天），"
            "根据游玩路线选择1-2个最适合住宿的地方，如果有具体的酒店名称请给出酒店名称，如果没有请给出住宿建议。给出参考价格区间"
            "并给大致游玩概览；然后按这个建议天数输出逐日路线。\n\n"
            "【每日结构（每一天都必须包含）】\n"
            "- 早餐：给出 2 家，说明推荐理由、是否常排队、建议到店时间。\n"
            "- 饭后安排：列出上午景点与顺序，标注每个点建议游玩时长，并说明这样安排是否顺路。\n"
            "- 午餐：先判断上午结束后是否来得及正常午饭。\n"
            "  - 若来不及：建议 14:00-16:00 去网红餐厅错峰，并说明为什么这个时段更合适。\n"
            "  - 若来得及：给出 1-2 家午餐店并说明理由、预估排队时长。\n"
            "- 下午规划：给出 1-2 个景点，说明游玩时长、交通衔接和体力节奏是否合理。\n"
            "- 晚餐：给出 2 家，说明理由、是否需要排队、最佳到店时间。\n"
            "- 出片点提醒：至少给 1-2 个适合拍照出片的位置和推荐时间段。\n\n"
            "- 每天都要覆盖早餐、上午、午餐策略、下午、晚餐、出片点、时间合理性说明。\n"
            "【全局要求】\n"
            "- 行程规划不要太早以8:30-9点出门为最佳 早餐不要早于8:30 如果10点后出门 建议直接吃早午餐\n"
            "- 若要必须早出门请说明理由\n"
            "- 每天的餐食要做到不重复 比如第一天吃烤鸭第二天的推荐里面就不要再有烤鸭了\n"
            "- 每一天末尾补充：说明景点数量、步行/交通时长、就餐节奏是否过紧。\n"
            "- 餐厅推荐必须解释“为什么选它”（口味、位置、口碑、与路线顺路程度）。\n"
            "- 如果某餐厅大概率排队，给出明确错峰建议或备选方案。\n"
            "- 不要只给景点清单，必须给可执行的时间段与顺序。"
    )


@tool
def search_trivel(query: str) -> str:
    """旅游行程专用工具：从本地 data/cache 语义检索笔记并返回“强约束格式指令”。当用户问旅游攻略/行程路线/景点美食时必须优先调用本工具，并严格按工具返回的“行程输出要求（必须遵守）”组织答案，不得改写为泛泛介绍。"""
    q = (query or "").strip()
    if not q:
        return "请提供具体的旅游攻略检索问题。"

    top_k = TOP_K_FOOD if is_food_focus_query(q) else TOP_K_NOTES
    docs = retrieve_travel_docs(q, top_k=top_k)
    print("search_trivel 命中文档数:", len(docs))
    if not docs:
        return "本地暂无缓存笔记（data/cache 为空或无法读取），请先通过其它方式导入缓存数据。"

    # days = parse_travel_days(q)
    format_instruction = build_itinerary_format_instruction(q)
    print("format_instruction===========format_instruction \n", format_instruction, "\n")
    compact_summary = build_compact_material_summary(docs, q)
    print("compact_summary===========compact_summary \n", compact_summary, "\n")
    blocks = []
    for rank, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        feed_images_json = metadata.get("feed_images_json") or "[]"
        try:
            feed_images = json.loads(feed_images_json)
        except (TypeError, json.JSONDecodeError):
            feed_images = []
        note = {
            "note_id": metadata.get("note_id"),
            "note_url": metadata.get("note_url"),
            "title": metadata.get("title"),
            "desc": metadata.get("desc"),
            "ocr_text": metadata.get("ocr_text"),
            "feed_images": feed_images,
        }
        try:
            note_block = build_note_block(note)
        except Exception as error:
            # 单条素材构建失败时降级为简版，避免整次工具返回失败。
            note_block = (
                f"【标题】{(note.get('title') or '').strip() or '(无标题)'}\n\n"
                f"【正文 desc】\n{(note.get('desc') or '').strip()}\n\n"
                f"【配图 OCR 文字】（该条素材处理失败：{error}）"
            )
        blocks.append(f"========== 结果 {rank}（相似度排序） ==========\n{note_block}")
    
    return (
        format_instruction
        + "\n\n【执行提醒】\n"
        "- 优先使用下面的参考材料事实，不够再做合理补充，但不要偏离用户问题。\n\n"
        "- 语言可以稍微幽默风趣一点，不要过于正式\n"
        "- 回答的时候需要多用表情包或者图标来增加趣味性 比如地点可以用📍 火锅可以用🍲 景点可以用🎈 交通可以用🚗 住宿可以用🏠 等等\n"
        "- 回答格式可以多元化，不要过于单一，文字可以加点颜色\n"
        + "【素材紧凑摘要（优先使用）】但不要出现“素材”、“参考材料”等字眼\n"
        + compact_summary
        + "\n\n".join(blocks)
    )
