"""Crawler run entry."""

import argparse
import hashlib
import random
import time

from .config import CrawlerConfig
from .http_api_client import XhsHttpApiClient
from .parser import NoteParser
from .planner import TravelRoutePlanner
from .playwright_client import PlaywrightCrawlerClient
from .pipeline import DataPipeline
from .storage import CrawlerStorage


def run_crawler(
    keyword: str,
    output_path: str = "data/xhs_notes.json",
    headless: bool = False,
    max_scroll_rounds: int = 8,
    max_note_count: int = 50,
    use_cache: bool = True,
    force_refresh: bool = False,
    use_cdp: bool = False,
    cdp_url: str = "http://127.0.0.1:9222",
    use_http_api: bool = False,
) -> list[dict]:
    """Run Playwright crawler and return cleaned notes."""
    config = CrawlerConfig()
    config.headless = headless
    config.max_scroll_rounds = max_scroll_rounds
    config.max_note_count = max_note_count
    config.use_cdp = use_cdp
    config.cdp_url = cdp_url
    config.use_http_api = use_http_api
    client = PlaywrightCrawlerClient(config)
    http_client = XhsHttpApiClient(config)
    parser = NoteParser()
    pipeline = DataPipeline()
    storage = CrawlerStorage()
    keyword_digest = hashlib.md5(keyword.encode("utf-8")).hexdigest()[:12]
    cache_path = f"data/cache/xhs_notes_{keyword_digest}.json"

    if use_cache and not force_refresh:
        cached = storage.load_json(cache_path)
        if isinstance(cached, list) and cached:
            storage.save_json(cached, output_path=output_path)
            return cached

    if use_http_api:
        search_payload = http_client.search_notes(keyword=keyword, page=1)
        print("search_payload===========search_payload \n", search_payload, "\n")
        search_records = parser.parse_search_api_cards(search_payload)
        detail_records: list[dict] = []
        for index, record in enumerate(search_records[:max_note_count]):
            if index > 0:
                time.sleep(random.uniform(1.5, 3))
            feed_payload = http_client.fetch_feed_detail(
                note_id=record["note_id"], xsec_token=record["xsec_token"]
            )
            detail_records.append(parser.parse_feed_detail(feed_payload))
    else:
        # 必须在同一 Tab 的搜索会话内跳转 explore（带 source=web_explore_feed），新开页面往往不会触发 feed。
        cards = client.search_note_cards_and_fetch_details(keyword=keyword)
        card_records = [parser.parse_card(card) for card in cards]
        unique_search_records = pipeline.deduplicate(pipeline.clean(card_records))
        detail_records = []
        for record in unique_search_records:
            detail_records.append(
                {
                    "note_id": record["note_id"],
                    "note_url": record["note_url"],
                    "title": record["title"],
                    "desc": record["desc"],
                    "feed_images": record["feed_images"],
                }
            )

    final_records = pipeline.deduplicate(pipeline.clean(detail_records))
    storage.save_json(final_records, output_path=cache_path)
    storage.save_json(final_records, output_path=output_path)
    return final_records


def run_travel_route_agent(
    keyword: str,
    destination: str,
    days: int,
    notes_output_path: str = "data/xhs_notes.json",
    route_output_path: str = "data/xhs_route_plan.json",
    headless: bool = False,
    max_scroll_rounds: int = 8,
    max_note_count: int = 50,
    use_cache: bool = True,
    force_refresh: bool = False,
    use_cdp: bool = False,
    cdp_url: str = "http://127.0.0.1:9222",
    use_http_api: bool = False,
) -> dict:
    """Run note crawl + travel route planning in one entry."""
    records = run_crawler(
        keyword=keyword,
        output_path=notes_output_path,
        headless=headless,
        max_scroll_rounds=max_scroll_rounds,
        max_note_count=max_note_count,
        use_cache=use_cache,
        force_refresh=force_refresh,
        use_cdp=use_cdp,
        cdp_url=cdp_url,
        use_http_api=use_http_api,
    )
    print("run_travel_route_agent===========records \n", records, "\n")
    planner = TravelRoutePlanner()
    storage = CrawlerStorage()
    route_plan = planner.build_route(records=records, destination=destination, days=days)
    storage.save_route_plan(route_plan, output_path=route_output_path)
    return route_plan


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for quick local execution."""
    parser = argparse.ArgumentParser(description="XHS crawler for travel route planning")
    parser.add_argument("--keyword", type=str, default="杭州 旅游 攻略")
    parser.add_argument("--destination", type=str, default="杭州")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-scroll-rounds", type=int, default=8)
    parser.add_argument("--max-note-count", type=int, default=50)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--use-cdp", action="store_true")
    parser.add_argument("--cdp-url", type=str, default="http://127.0.0.1:9222")
    parser.add_argument("--use-http-api", action="store_true")
    parser.add_argument("--notes-output", type=str, default="data/xhs_notes.json")
    parser.add_argument("--route-output", type=str, default="data/xhs_route_plan.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_travel_route_agent(
        keyword=args.keyword,
        destination=args.destination,
        days=args.days,
        notes_output_path=args.notes_output,
        route_output_path=args.route_output,
        headless=args.headless,
        max_scroll_rounds=args.max_scroll_rounds,
        max_note_count=args.max_note_count,
        use_cache=not args.no_cache,
        force_refresh=args.force_refresh,
        use_cdp=args.use_cdp,
        cdp_url=args.cdp_url,
        use_http_api=args.use_http_api,
    )
    print(
        "Travel route generated, destination="
        f'{result["destination"]}, days={result["days"]}, route_days={len(result["route"])}'
    )
