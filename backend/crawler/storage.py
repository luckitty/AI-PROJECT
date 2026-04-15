"""Storage backends for crawler output."""

import json
from pathlib import Path


class CrawlerStorage:
    """Provide JSON storage first, leave extension hooks for DB/Milvus."""

    def save_json(self, records: list[dict], output_path: str) -> None:
        """Persist records to a UTF-8 JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(records, file_obj, ensure_ascii=False, indent=2)

    def load_json(self, input_path: str):
        """Load json content if file exists."""
        path = Path(input_path)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def save_route_plan(self, plan: dict, output_path: str) -> None:
        """Persist route plan json for frontend or agent use."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(plan, file_obj, ensure_ascii=False, indent=2)

    def save_milvus(self, records: list[dict]) -> None:
        """Placeholder for Milvus integration."""
        # 后续在这里加入向量化与 Milvus 入库流程。
        print(f"[CrawlerStorage] save_milvus pending, records={len(records)}")

    def save_db(self, records: list[dict]) -> None:
        """Placeholder for relational or document DB integration."""
        # 后续在这里加入数据库连接和批量写入逻辑。
        print(f"[CrawlerStorage] save_db pending, records={len(records)}")
