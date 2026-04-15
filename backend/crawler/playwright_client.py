"""Playwright based crawler for Xiaohongshu pages."""

import random
from pathlib import Path
from typing import Any
from urllib.parse import quote

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from .config import CrawlerConfig
from .image_download import download_note_images_from_url_defaults
from .parser import NoteParser


class PlaywrightCrawlerClient:
    """Collect notes from web pages through browser automation."""

    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self.storage_state_file = Path(config.storage_state_path)
        self.storage_state_file.parent.mkdir(parents=True, exist_ok=True)

    def create_context(self, browser):
        """Create browser context with optional persisted state."""
        if self.config.use_cdp and browser.contexts:
            return browser.contexts[0]
        if self.storage_state_file.exists():
            return browser.new_context(storage_state=str(self.storage_state_file))
        return browser.new_context()

    def open_browser_and_context(self, playwright):
        """Open browser and context for current config."""
        if self.config.use_cdp:
            browser = playwright.chromium.connect_over_cdp(self.config.cdp_url)
            context = self.create_context(browser)
            return browser, context
        browser = playwright.chromium.launch(headless=self.config.headless)
        context = self.create_context(browser)
        return browser, context

    def search_note_cards(self, keyword: str) -> list[dict[str, str]]:
        """打开搜索页并从 search/notes 接口提取卡片。"""
        encoded_keyword = quote(keyword)
        search_url = f"https://www.xiaohongshu.com/search_result?keyword={encoded_keyword}&source=web_explore_feed"
        with sync_playwright() as p:
            browser, context = self.open_browser_and_context(p)
            page = context.new_page()
            api_cards: list[dict[str, str]] = []
            api_seen_note_ids: set[str] = set()
            note_parser = NoteParser()

            def collect_search_api_response(response) -> None:
                """解析 search/notes 响应并沉淀 note_id 与 xsec_token。"""
                if "/api/sns/web/v1/search/notes" not in response.url:
                    return
                try:
                    payload = response.json()
                except PlaywrightError:
                    return
                items = payload["data"]["items"]
                for item in items:
                    if item["model_type"] != "note":
                        continue
                    note_card = item["note_card"]
                    # 搜索接口里 id 有时只在 item 顶层，不在 note_card 里。
                    note_id = str(note_card.get("note_id") or item.get("id", ""))
                    if not note_id or note_id in api_seen_note_ids:
                        continue
                    api_seen_note_ids.add(note_id)
                    xsec_token, xsec_source = note_parser.xsec_from_search_item(item, note_card)
                    note_url = (
                        f"https://www.xiaohongshu.com/explore/{note_id}"
                        f"?xsec_token={xsec_token}&xsec_source={xsec_source}&source=web_explore_feed"
                    )
                    api_cards.append(
                        {
                            "note_url": note_url,
                            "title": note_parser.title_from_note_card(note_card),
                            "xsec_token": xsec_token,
                            "xsec_source": xsec_source,
                        }
                    )

            page.on("response", collect_search_api_response)
            page.goto(search_url, wait_until="domcontentloaded")
            page.wait_for_event(
                "response",
                predicate=lambda response: "/api/sns/web/v1/search/notes" in response.url,
                timeout=8000,
            )
            page.wait_for_timeout(2500)

            for _ in range(self.config.max_scroll_rounds):
                page.mouse.wheel(0, 3500)
                page.evaluate("window.scrollBy(0, 1800)")
                page.wait_for_timeout(self.config.scroll_wait_ms)

            if not self.config.use_cdp:
                context.storage_state(path=str(self.storage_state_file))
            browser.close()
        return self.deduplicate_cards(api_cards)

    def search_note_cards_and_fetch_details(self, keyword: str) -> list[dict[str, Any]]:
        """在同一浏览器、同一 Tab 内依次进入 explore 并监听 feed。"""
        encoded_keyword = quote(keyword)
        search_url = f"https://www.xiaohongshu.com/search_result?keyword={encoded_keyword}&source=web_explore_feed"
        merged: list[dict[str, Any]] = []
        with sync_playwright() as p:
            browser, context = self.open_browser_and_context(p)
            page = context.new_page()
            api_cards: list[dict[str, str]] = []
            api_seen_note_ids: set[str] = set()
            note_parser = NoteParser()

            def collect_search_api_response(response) -> None:
                """解析 search/notes 响应并沉淀 note_id 与 xsec_token。"""
                if "/api/sns/web/v1/search/notes" not in response.url:
                    return
                try:
                    payload = response.json()
                except PlaywrightError:
                    return
                items = payload["data"]["items"]
                for item in items:
                    if item["model_type"] != "note":
                        continue
                    note_card = item["note_card"]
                    # 搜索接口里 id 有时只在 item 顶层，不在 note_card 里。
                    note_id = str(note_card.get("note_id") or item.get("id", ""))
                    if not note_id or note_id in api_seen_note_ids:
                        continue
                    api_seen_note_ids.add(note_id)
                    xsec_token, xsec_source = note_parser.xsec_from_search_item(item, note_card)
                    note_url = (
                        f"https://www.xiaohongshu.com/explore/{note_id}"
                        f"?xsec_token={xsec_token}&xsec_source={xsec_source}&source=web_explore_feed"
                    )
                    api_cards.append(
                        {
                            "note_url": note_url,
                            "title": note_parser.title_from_note_card(note_card),
                            "xsec_token": xsec_token,
                            "xsec_source": xsec_source,
                        }
                    )

            page.on("response", collect_search_api_response)
            page.goto(search_url, wait_until="domcontentloaded")
            page.wait_for_event(
                "response",
                predicate=lambda response: "/api/sns/web/v1/search/notes" in response.url,
                timeout=8000,
            )
            page.wait_for_timeout(2500)

            for _ in range(self.config.max_scroll_rounds):
                page.mouse.wheel(0, 3500)
                page.evaluate("window.scrollBy(0, 1800)")
                page.wait_for_timeout(self.config.scroll_wait_ms)

            cards = self.deduplicate_cards(api_cards)
            for index, card in enumerate(cards):
                if index > 0:
                    page.wait_for_timeout(
                        random.randint(
                            self.config.detail_note_delay_ms_min,
                            self.config.detail_note_delay_ms_max,
                        )
                    )
                detail_url = str(card["note_url"])
                detail = self.fetch_single_detail_with_retry(page=page, note_url=detail_url)
                note_id = str(detail["note_id"])
                image_urls = detail["feed_image_urls"]
                feed_images = download_note_images_from_url_defaults(note_id, image_urls)
                merged.append(
                    {
                        "note_url": detail_url,
                        "title": detail["title"],
                        "desc": detail["desc"],
                        "xsec_token": str(card["xsec_token"]),
                        "xsec_source": str(card["xsec_source"]),
                        "feed_images": feed_images,
                    }
                )
                print("fetch_single_detail_with_retry===========desc \n", detail["desc"], "\n")


            if not self.config.use_cdp:
                context.storage_state(path=str(self.storage_state_file))
            browser.close()
        return merged

    def fetch_note_details(self, note_urls: list[str]) -> list[dict[str, Any]]:
        """批量抓取帖子详情并返回 feed 提取结果（含本地下载后的 feed 图片列表）。"""
        details: list[dict[str, Any]] = []
        with sync_playwright() as p:
            browser, context = self.open_browser_and_context(p)
            page = context.new_page()
            for index, note_url in enumerate(note_urls):
                if index > 0:
                    page.wait_for_timeout(
                        random.randint(
                            self.config.detail_note_delay_ms_min,
                            self.config.detail_note_delay_ms_max,
                        )
                    )
                detail = self.fetch_single_detail_with_retry(page=page, note_url=note_url)
                print("批量抓取帖子详情并返回feed 提取结果 \n", detail, "\n")
                nid = str(detail["note_id"])
                image_urls = detail["feed_image_urls"]
                feed_images = download_note_images_from_url_defaults(nid, image_urls)
                details.append(
                    {
                        "note_url": note_url,
                        "title": detail["title"],
                        "desc": detail["desc"],
                        "feed_images": feed_images,
                    }
                )
            if not self.config.use_cdp:
                context.storage_state(path=str(self.storage_state_file))
            browser.close()
        return details

    def fetch_single_detail_with_retry(self, page, note_url: str) -> dict[str, Any]:
        """抓取单条详情并直接解析 feed 接口响应。"""
        candidate_urls = self.build_detail_url_candidates(note_url)
        print("candidate_urls===========candidate_urls \n", candidate_urls, "\n")
        candidate_url = candidate_urls[0]
        try:
            with page.expect_response(
                lambda response: "/api/sns/web/v1/feed" in response.url and response.request.method == "POST",
                timeout=25000,
            ) as feed_response_info:
                page.evaluate(
                    """
                    (targetUrl) => {
                        const u = new URL(targetUrl);
                        const nextPath = u.pathname + u.search;
                        window.history.replaceState(window.history.state, '', nextPath);
                        window.dispatchEvent(new PopStateEvent('popstate', { state: window.history.state }));
                    }
                    """,
                    candidate_url,
                )
                # 过短易漏 feed，过长会线性叠在每条详情上（用户体感「十几秒才切一次」多来自此 + 配图下载）。
                page.wait_for_timeout(self.config.feed_popstate_wait_ms)
        except PlaywrightError as error:
            raise RuntimeError(f"抓取 feed 接口失败: {candidate_url}; 原因: {error}") from error
        try:
            feed_payload = feed_response_info.value.json()
        except Exception as error:
            raise RuntimeError(f"解析 feed 响应失败: {candidate_url}; 原因: {error}") from error
        return self.extract_detail_from_feed_payload(feed_payload)

    def extract_detail_from_feed_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """从 feed 响应中提取标题、正文与配图 url_default 列表。"""
        items = payload["data"]["items"]
        note_item = next(i for i in items if i["model_type"] == "note")
        note_card = note_item["note_card"]
        note_parser = NoteParser()
        image_urls = note_parser.extract_feed_image_urls_default(note_card)
        # feed 里 note_id 也可能只在 item 顶层。
        note_id = str(note_card.get("note_id") or note_item.get("id", ""))
        return {
            "note_id": note_id,
            "title": note_parser.title_from_note_card(note_card),
            "desc": str(note_card["desc"]).strip(),
            "feed_image_urls": image_urls,
        }

    def build_detail_url_candidates(self, note_url: str) -> list[str]:
        """构造详情页候选链接列表，当前仅返回原始链接。"""
        return [note_url]

    def deduplicate_cards(self, cards: list[dict[str, Any]]) -> list[dict[str, str]]:
        """按 URL 去重并限制条数。"""
        seen_urls: set[str] = set()
        result: list[dict[str, str]] = []
        for card in cards:
            note_url = str(card["note_url"])
            if note_url in seen_urls:
                continue
            seen_urls.add(note_url)
            result.append(
                {
                    "note_url": note_url,
                    "title": str(card["title"]),
                    "xsec_token": str(card["xsec_token"]),
                    "xsec_source": str(card["xsec_source"]),
                }
            )
            if len(result) >= self.config.max_note_count:
                break
        return result
