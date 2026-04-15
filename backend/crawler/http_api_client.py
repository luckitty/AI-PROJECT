"""HTTP API client for Xiaohongshu web endpoints."""

from typing import Any

import requests

from .config import CrawlerConfig


class XhsHttpApiClient:
    """Use captured XHS web API requests directly."""

    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.http_common_headers)
        if config.http_cookie:
            self.session.headers["Cookie"] = config.http_cookie

    def search_notes(self, keyword: str, page: int = 1) -> dict[str, Any]:
        """Call /api/sns/web/v1/search/notes."""
        payload = {
            "keyword": keyword,
            "page": page,
            "page_size": self.config.http_page_size,
            "search_id": self.config.http_search_id,
            "sort": "general",
            "note_type": 0,
            "ext_flags": [],
            "filters": [
                {"tags": ["general"], "type": "sort_type"},
                {"tags": ["不限"], "type": "filter_note_type"},
                {"tags": ["不限"], "type": "filter_note_time"},
                {"tags": ["不限"], "type": "filter_note_range"},
                {"tags": ["不限"], "type": "filter_pos_distance"},
            ],
            "geo": "",
            "image_formats": ["jpg", "webp", "avif"],
        }
        response = self.session.post(
            self.config.http_search_url,
            json=payload,
            timeout=20,
        )
        print("search_notes===========response \n", response, "\n")
        response.raise_for_status()
        print("search_notes===========response.json() \n", response.json(), "\n")
        return response.json()

    def fetch_feed_detail(self, note_id: str, xsec_token: str = "") -> dict[str, Any]:
        """Call /api/sns/web/v1/feed with source_note_id."""
        payload = {
            "source_note_id": note_id,
            "image_formats": ["jpg", "webp", "avif"],
            "extra": {"need_body_topic": "1"},
            "xsec_source": "pc_search",
            "xsec_token": xsec_token,
        }
        response = self.session.post(
            self.config.http_feed_url,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
