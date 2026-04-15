"""Crawler configuration objects."""

from dataclasses import dataclass, field


@dataclass
class CrawlerConfig:
    """Playwright crawler settings."""

    # 无头模式默认关闭，便于首次手动确认登录态与验证码状态。
    headless: bool = False
    # 复用登录态文件，可减少频繁登录导致的风险。
    storage_state_path: str = "data/xhs_storage_state.json"
    # 每次滚动等待的毫秒数，适当放慢可降低请求突发。
    scroll_wait_ms: int = 1500
    # 搜索页滚动次数上限。
    max_scroll_rounds: int = 8
    # 每次任务最多抓取多少篇帖子详情，避免过量访问。
    max_note_count: int = 50
    # 相邻两条详情之间随机等待（毫秒），降低固定节奏；过大会让「切换 URL」体感很慢。
    detail_note_delay_ms_min: int = 800
    detail_note_delay_ms_max: int = 2200
    # popstate 触发后给 SPA 发 feed 的缓冲（毫秒）；原 1200ms 会实打实叠在每条详情上。
    feed_popstate_wait_ms: int = 450
    # 可选：连接已开启远程调试端口的本机 Chrome（复用你已登录会话）。
    use_cdp: bool = False
    cdp_url: str = "http://127.0.0.1:9222"
    # 可选：直接使用小红书 web 接口，绕过页面 DOM 抓取。
    use_http_api: bool = False
    http_search_url: str = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"
    http_feed_url: str = "https://edith.xiaohongshu.com/api/sns/web/v1/feed"
    http_page_size: int = 20
    http_search_id: str = ""
    # 不在代码中固化任何个人 cookie，避免泄露和过期问题。
    http_cookie: str = ""
    # 仅保留基础请求头；页面抓取模式不依赖这些字段。
    http_common_headers: dict[str, str] = field(
        default_factory=lambda: {
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "content-type": "application/json;charset=UTF-8",
            "origin": "https://www.xiaohongshu.com",
            "referer": "https://www.xiaohongshu.com/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        }
    )
