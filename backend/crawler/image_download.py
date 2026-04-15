"""从 feed 的 note_card.image_list[].url_default 下载图片到本地。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


def download_note_images_from_url_defaults(
    note_id: str,
    urls: list[str],
    root_dir: str = "data/note_images",
) -> list[dict[str, Any]]:
    """
    按 url_default 列表顺序下载，保存为 img_0.webp、img_1.webp …

    不做后缀猜测：feed 里 url_default 已是完整可请求地址，笔记配图为 webp 流，统一落盘为 .webp。

    多图时并行拉取，避免串行 HTTP 把单条笔记耗时拉到十几秒。

    返回每条：url、local_path（失败则 local_path 为空字符串）。
    """
    base = Path(root_dir) / note_id
    base.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.xiaohongshu.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }

    def download_one(pair: tuple[int, str]) -> dict[str, Any]:
        """按索引下载单张图；executor.map 保证结果顺序与 urls 一致。"""
        index, raw_url = pair
        url = str(raw_url).strip()
        file_path = base / f"img_{index}.webp"
        try:
            response = requests.get(url, headers=headers, timeout=45, stream=True)
            response.raise_for_status()
            with file_path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        file_obj.write(chunk)
            rel = str(file_path).replace("\\", "/")
            return {"url": url, "local_path": rel}
        except requests.RequestException:
            return {"url": url, "local_path": ""}

    if not urls:
        return []

    workers = min(8, len(urls))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(download_one, enumerate(urls)))
