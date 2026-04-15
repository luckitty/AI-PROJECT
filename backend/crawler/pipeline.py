"""Data cleaning and deduplication pipeline."""


class DataPipeline:
    """Pipeline for cleaning and de-duplicating crawler results."""

    def clean(self, records: list[dict]) -> list[dict]:
        """去除标题与正文首尾空格。"""
        for record in records:
            record["title"] = str(record["title"]).strip()
            record["desc"] = str(record["desc"]).strip()
        return records

    def deduplicate(self, records: list[dict]) -> list[dict]:
        """按 note_id 去重。"""
        seen_ids: set[str] = set()
        unique_records: list[dict] = []
        for record in records:
            note_id = str(record["note_id"])
            if note_id in seen_ids:
                continue
            seen_ids.add(note_id)
            unique_records.append(record)
        return unique_records
