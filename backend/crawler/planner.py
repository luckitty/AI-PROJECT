"""Travel route planner based on crawled notes."""

from collections import Counter


class TravelRoutePlanner:
    """Generate a lightweight route plan from note records."""

    def extract_spot_candidates(self, records: list[dict]) -> list[str]:
        """从标题和正文中提取候选景点词。"""
        spot_candidates: list[str] = []
        noise_words = {
            "攻略",
            "酒店",
            "民宿",
            "机票",
            "高铁",
            "地铁",
            "拍照",
            "vlog",
            "Vlog",
            "citywalk",
            "Citywalk",
            "路线",
            "避坑",
            "预算",
            "人均",
            "天",
            "日游",
        }
        spot_suffixes = ["公园", "古镇", "博物馆", "景区", "寺", "山", "湖", "街", "广场", "码头", "塔", "桥"]
        for record in records:
            text = f'{record["title"]} {record["desc"]}'
            parts = (
                text.replace("，", " ")
                .replace("。", " ")
                .replace("！", " ")
                .replace("?", " ")
                .replace("？", " ")
                .replace("｜", " ")
                .replace("|", " ")
                .replace("/", " ")
                .split()
            )
            for part in parts:
                if len(part) < 2:
                    continue
                if any(word in part for word in noise_words):
                    continue
                if any(tag in part for tag in spot_suffixes):
                    spot_candidates.append(part[:12])
        return spot_candidates

    def build_route(self, records: list[dict], destination: str, days: int) -> dict:
        """根据提取到的景点构建按天路线草案。"""
        counter = Counter(self.extract_spot_candidates(records))
        ranked_spots = [spot for spot, _ in counter.most_common(max(days * 3, 6))]
        route_days: list[dict] = []

        for day_index in range(days):
            day_spots = ranked_spots[day_index * 3 : day_index * 3 + 3]
            route_days.append(
                {
                    "day": day_index + 1,
                    "spots": day_spots,
                    "notes": "根据小红书高频内容自动生成，建议出发前二次确认开放时间。",
                }
            )

        return {
            "destination": destination,
            "days": days,
            "top_spots": ranked_spots[:10],
            "route": route_days,
        }
