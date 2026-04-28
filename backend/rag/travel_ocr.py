from pathlib import Path
import json

MAX_IMAGES_PER_NOTE_FOR_OCR = 8
MAX_OCR_CHARS_FOR_EMBED = 1200
ocr_engine = None


def normalize_text(value) -> str:
    """统一清洗文本：把 None 等值转换成去空白字符串。"""
    return str(value or "").strip()


def load_ocr_cache(cache_file: Path):
    """读取 OCR 文本缓存，避免重复识别相同图片。"""
    if not cache_file.is_file():
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_ocr_cache(cache_file: Path, cache_data: dict):
    """持久化 OCR 缓存到 data 目录。"""
    try:
        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(cache_data, file, ensure_ascii=False)
    except OSError:
        pass


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
    """懒加载 OCR 引擎，失败时返回 None。"""
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


def get_or_build_note_ocr_text(
    note: dict,
    backend_root: Path,
    ocr_cache: dict,
    allow_runtime_ocr: bool = False,
):
    """获取单条笔记 OCR 文本，支持签名缓存与空值确认。"""
    note_id = normalize_text(note.get("note_id"))
    if not note_id:
        return "", False
    image_paths = resolve_local_image_paths(note, backend_root)
    if not image_paths:
        return "", False
    signature = []
    for image_path in image_paths:
        stat = image_path.stat()
        signature.append(f"{str(image_path)}|{stat.st_mtime}|{stat.st_size}")
    cached_item = ocr_cache.get(note_id)
    if isinstance(cached_item, dict) and cached_item.get("signature") == signature:
        cached_ocr = str(cached_item.get("ocr_text") or "").strip()
        if cached_ocr:
            return cached_ocr, False
        if cached_item.get("empty_verified") or not allow_runtime_ocr:
            return "", False
    if not allow_runtime_ocr:
        return "", False

    ocr_texts = []
    for image_path in image_paths:
        text = ocr_image_text(image_path)
        if text:
            ocr_texts.append(text)
    merged_ocr_text = "\n\n".join(ocr_texts).strip()
    ocr_cache[note_id] = {"signature": signature, "ocr_text": merged_ocr_text, "empty_verified": not bool(merged_ocr_text)}
    return merged_ocr_text, True
