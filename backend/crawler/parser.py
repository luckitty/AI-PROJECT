"""Transform raw API payloads into structured records."""

from typing import Any


class NoteParser:
    """Parser for search and note detail data."""

    @staticmethod
    def xsec_from_search_item(item: dict[str, Any], note_card: dict[str, Any]) -> tuple[str, str]:
        """从搜索 item / note_card 解析 xsec_token、xsec_source（字段可能只在某一层出现）。"""
        user = note_card.get("user")
        if not isinstance(user, dict):
            user = {}
        token = item.get("xsec_token") or note_card.get("xsec_token") or user.get("xsec_token") or ""
        source = item.get("xsec_source") or note_card.get("xsec_source") or "pc_search"
        return str(token), str(source)

    @staticmethod
    def title_from_note_card(note_card: dict[str, Any]) -> str:
        """note_card 标题可能在 display_title 或 title（搜索与 feed 结构类似）。"""
        return str(note_card.get("display_title") or note_card.get("title") or "").strip()

    def parse_search_result(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """解析旧版搜索接口列表。"""
        items = payload["data"]["items"]
        return [{"note_id": str(item["id"]), "title": item["title"]} for item in items]

    def parse_note_detail(self, payload: dict[str, Any]) -> dict[str, Any]:
        """解析旧版详情接口。"""
        data = payload["data"]
        return {"note_id": str(data["id"]), "title": data["title"], "desc": data["desc"]}

    def extract_note_id_from_url(self, note_url: str) -> str:
        """从帖子链接中提取 note_id（explore / discovery 路径）。"""
        normalized_url = note_url.split("?", 1)[0].rstrip("/")
        if "/explore/" in normalized_url:
            return normalized_url.split("/explore/")[-1]
        if "/discovery/item/" in normalized_url:
            return normalized_url.split("/discovery/item/")[-1]
        return normalized_url.split("/")[-1]

    def extract_feed_image_urls_default(self, note_card: dict[str, Any]) -> list[str]:
        """从 note_card.image_list[] 收集 url_default。"""
        return [str(img["url_default"]).strip() for img in note_card["image_list"]]

    def build_explore_url(self, note_id: str, xsec_token: str = "", xsec_source: str = "pc_search") -> str:
        """根据 note_id 与 xsec_token 构造详情页链接。"""
        if xsec_token:
            return (
                f"https://www.xiaohongshu.com/explore/{note_id}"
                f"?xsec_token={xsec_token}&xsec_source={xsec_source}&source=web_explore_feed"
            )
        return f"https://www.xiaohongshu.com/explore/{note_id}"

    def parse_card(self, card: dict[str, Any]) -> dict[str, str]:
        """标准化页面抓取的列表卡片数据。"""
        note_url = str(card["note_url"])
        note_id = self.extract_note_id_from_url(note_url)
        xsec_token = str(card["xsec_token"])
        xsec_source = str(card["xsec_source"])
        normalized_note_url = self.build_explore_url(
            note_id=note_id, xsec_token=xsec_token, xsec_source=xsec_source
        )
        return {
            "note_id": note_id,
            "note_url": normalized_note_url,
            "title": str(card["title"]),
            "xsec_token": xsec_token,
            "xsec_source": xsec_source,
            "desc": str(card["desc"]),
            "feed_images": card["feed_images"],
        }

    def parse_detail(self, detail: dict[str, Any], note_id: str, note_url: str) -> dict[str, str]:
        """Normalize detail page extraction."""
        return {
            "note_id": note_id,
            "note_url": note_url,
            "title": str(detail["title"]),
            "desc": str(detail["desc"]),
        }

    def parse_search_api_cards(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        """解析搜索接口卡片列表。"""
        items = payload["data"]["items"]
        records: list[dict[str, str]] = []
        for item in items:
            if item["model_type"] != "note":
                continue
            note_card = item["note_card"]
            # 搜索接口里 id 有时只在 item 顶层，不在 note_card 里。
            note_id = str(note_card.get("note_id") or item.get("id", ""))
            if not note_id:
                continue
            xsec_token, _ = NoteParser.xsec_from_search_item(item, note_card)
            tag_names = [str(tag["name"]) for tag in note_card["tag_list"]]
            records.append(
                {
                    "note_id": note_id,
                    "xsec_token": xsec_token,
                    "title": NoteParser.title_from_note_card(note_card),
                    "desc": " ".join(tag_names),
                }
            )
        return records

    def parse_feed_detail(self, payload: dict[str, Any]) -> dict[str, str]:
        """解析 feed 接口：取首条 note 的 note_card。"""
        items = payload["data"]["items"]
        note_item = next(i for i in items if i["model_type"] == "note")
        note_card = note_item["note_card"]
        return {
            "note_id": str(note_card.get("note_id") or note_item.get("id", "")),
            "title": NoteParser.title_from_note_card(note_card),
            "desc": str(note_card["desc"]),
        }
